"""Firul de executie care ruleaza lotul in spatele ferestrei.

Nu contine logica de conversie. Ruleaza **exact** `Runner.run()` — acelasi cod
pe care il foloseste linia de comanda — si traduce apelurile lui de progres in
semnale Qt.

De ce conteaza separarea: incarcarea modelului si transcrierea dureaza secunde
bune per pagina. Facute pe firul interfetei, fereastra ar parea inghetata tot
lotul, iar butonul de oprire nu ar mai raspunde.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from app.batch.runner import Progress, Runner, enqueue
from app.batch.store import Job, Store
from app.recognition.engine import Kind
from app.recognition.smt_engine import engine_for, greutati_produs


class EnqueueWorker(QObject):
    """Pune fisiere in coada, pe alt fir.

    Pare o operatie instantanee, dar nu e: fiecare fisier e citit integral ca
    sa i se calculeze amprenta, iar pentru un folder cu 500 de scanuri asta
    inseamna un gigaoctet de citiri. Pe firul interfetei ar ingheta fereastra.
    """

    finished = Signal(int)  # cate unitati de lucru asteapta
    failed = Signal(str)

    def __init__(self, db_path: Path, kind: Kind, sources: list[Path]) -> None:
        super().__init__()
        self.db_path = Path(db_path)
        # `Kind` mosteneste din `str`, iar Qt il aplatizeaza la un `str` simplu
        # cand trece prin `QComboBox.currentData()`. Fara conversia asta,
        # primul `kind.value` de mai jos crapa.
        self.kind = Kind(kind)
        self.sources = sources

    def run(self) -> None:
        try:
            with Store(self.db_path) as store:
                self.finished.emit(enqueue(store, self.sources, self.kind))
        except Exception:  # noqa: BLE001 - orice esec ajunge in fereastra
            self.failed.emit(traceback.format_exc())


class BatchWorker(QObject):
    """Ruleaza un lot si raporteaza inaintarea lui."""

    progressed = Signal(object)  # Progress
    finished = Signal(list)  # list[Job]
    failed = Signal(str)
    started_loading = Signal()

    def __init__(self, db_path: Path, out_dir: Path, kind: Kind,
                 sources: list[Path] | None = None) -> None:
        super().__init__()
        self.db_path = Path(db_path)
        self.out_dir = Path(out_dir)
        self.kind = Kind(kind)  # vezi nota din `EnqueueWorker.__init__`
        self.sources = sources or []
        self.cancel = threading.Event()

    def stop(self) -> None:
        """Opreste dupa pagina curenta.

        Nu intrerupem la mijlocul unei pagini: am ramane cu un job `running` in
        baza si cu iesiri scrise pe jumatate.
        """
        self.cancel.set()

    def run(self) -> None:
        try:
            self.started_loading.emit()
            # Curatarea pornita pe scris de mana, fiindca fereastra n-are de
            # unde s-o ceara — si e o valoare **masurata**, nu o preferinta:
            # §15.1 a aratat ca dubleaza notele corecte, 5,5% -> 11,0%. Fara ea,
            # cine converteste din interfata primeste tocmai setarea care s-a
            # dovedit mai proasta pe materialul asta.
            #
            # Gruparea portativelor nu se poate face implicita in acelasi fel:
            # variaza de la pagina la pagina (§15.8), deci ramane pentru cand
            # fereastra va avea controalele ei.
            #
            # Greutatile antrenate, cand au fost publicate (`greutati_produs`), cate
            # unele pe fel de pagina: pe tiparit, `pagina3` trece de la 32,7% cu
            # modelul publicat la ~90% (PROJECT.md §15.60-§15.67). Curatarea ramane
            # oprita pe tiparit: cu greutatile antrenate nu schimba nimic (489 fara, 484 cu).
            engine = engine_for(
                self.kind, clean=self.kind is Kind.HANDWRITTEN,
                weights=greutati_produs(self.kind),
            )
            with Store(self.db_path) as store:
                runner = Runner(store=store, engine=engine, out_dir=self.out_dir)
                if self.sources:
                    runner.enqueue(self.sources, self.kind)
                jobs = runner.run(progress=self.progressed.emit, cancel=self.cancel)
            self.finished.emit(jobs)
        except Exception:  # noqa: BLE001 - orice esec ajunge in fereastra, nu in consola
            self.failed.emit(traceback.format_exc())


def start(worker: BatchWorker) -> QThread:
    """Porneste un lucrator pe firul lui si intoarce firul.

    Apelantul trebuie sa pastreze o referinta atat la `worker` cat si la firul
    intors: daca le lasa sa fie colectate, Qt inchide firul sub picioarele lui.
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.start()
    return thread
