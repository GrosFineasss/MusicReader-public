"""Fereastra principala.

Nu contine **nicio** logica de recunoastere. Trage fisiere, porneste lotul intr-un
fir separat, si afiseaza ce raporteaza el. Daca schimbam modelul, fisierul asta
nu se atinge.

Corectarea se face in MuseScore, nu aici — vezi PROJECT.md §1, "Ce NU construim".
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.batch import inputs, report
from app.batch.store import Job, State, Store
from app.recognition.engine import Kind
from app.ui import musescore
from app.ui.worker import BatchWorker, EnqueueWorker, start

COLUMNS = ("Fisier", "Tip", "Stare", "Portative", "Sisteme", "Erori", "Secunde", "")

STATE_LABELS = {
    State.PENDING: "in asteptare",
    State.RUNNING: "in lucru",
    State.DONE: "gata",
    State.FAILED: "esuat",
}


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f} secunde"
    return f"{seconds / 60:.0f} minute"


class DropArea(QLabel):
    """Zona in care se trag fisierele si folderele."""

    def __init__(self, on_drop) -> None:
        super().__init__("Trage aici poze, PDF-uri sau foldere\n(sau apasa „Alege…”)")
        self.on_drop = on_drop
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(90)
        self.setStyleSheet(
            "border: 2px dashed palette(mid); border-radius: 8px; color: palette(mid);"
        )

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        self.on_drop([p for p in paths if p.exists()])
        event.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self, out_dir: Path) -> None:
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.out_dir / "batch.db"
        # Un `QThread` care ruleaza si isi pierde ultima referinta Python e
        # distrus sub picioarele lui si opreste procesul. De aceea referintele
        # se elibereaza abia pe semnalul `finished` al firului, niciodata pe
        # cel al lucratorului.
        self.batch_worker: BatchWorker | None = None
        self.batch_thread = None
        self.enqueue_worker: EnqueueWorker | None = None
        self.enqueue_thread = None

        self.setWindowTitle("OMR Cor — conversie in MusicXML")
        self.resize(980, 620)
        self._build()
        self._reload_table()
        self._check_musescore()

    # -- constructie ----------------------------------------------------

    def _build(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        controls = QHBoxLayout()
        self.kind_box = QComboBox()
        self.kind_box.addItem("Scrise de mana", Kind.HANDWRITTEN)
        self.kind_box.addItem("Tiparite / scanate", Kind.PRINTED)
        controls.addWidget(QLabel("Tipul partiturilor:"))
        controls.addWidget(self.kind_box)

        self.choose_files_button = QPushButton("Alege fisiere…")
        self.choose_files_button.clicked.connect(self._choose_files)
        controls.addWidget(self.choose_files_button)

        self.choose_folder_button = QPushButton("Alege folder…")
        self.choose_folder_button.clicked.connect(self._choose_folder)
        controls.addWidget(self.choose_folder_button)
        controls.addStretch()

        self.start_button = QPushButton("Porneste conversia")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("Opreste")
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setEnabled(False)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)

        layout.addWidget(DropArea(self._add_paths))

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("Niciun lot pornit.")
        layout.addWidget(self.status)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        footer = QHBoxLayout()
        self.report_button = QPushButton("Deschide raportul")
        self.report_button.clicked.connect(self._open_report)
        self.folder_button = QPushButton("Deschide folderul cu rezultate")
        self.folder_button.clicked.connect(self._open_folder)
        footer.addWidget(self.report_button)
        footer.addWidget(self.folder_button)
        footer.addStretch()
        layout.addLayout(footer)

        self.setCentralWidget(central)

    # -- verificari de mediu --------------------------------------------

    def _check_musescore(self) -> None:
        if musescore.find() is None:
            self.status.setText(
                "MuseScore nu a fost gasit — conversia merge, dar butonul de "
                "deschidere nu va functiona."
            )
        elif not musescore.is_configured():
            self.status.setText(
                "MuseScore nu a fost pornit niciodata. Porneste-l o data manual: "
                "pana treci de expertul initial, se blocheaza la deschidere."
            )

    # -- surse ----------------------------------------------------------

    def _choose_files(self) -> None:
        """Alege una sau mai multe partituri.

        Filtrul se construieste din `inputs.SUPPORTED`, nu dintr-o lista scrisa
        aici: daca adaugam un format nou in lot, dialogul il accepta automat.
        """
        patterns = " ".join(f"*{suffix}" for suffix in sorted(inputs.SUPPORTED))
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Alege partituri",
            "",
            f"Partituri ({patterns});;Toate fisierele (*)",
        )
        if paths:
            self._add_paths([Path(p) for p in paths])

    def _choose_folder(self) -> None:
        """Alege un folder intreg — se parcurge recursiv."""
        folder = QFileDialog.getExistingDirectory(self, "Alege folderul cu partituri")
        if folder:
            self._add_paths([Path(folder)])

    def _add_paths(self, paths: list[Path]) -> None:
        """Pune fisierele in coada imediat si le arata in tabel.

        Nu le tinem in memorie pana la apasarea butonului de start: asa,
        fisierele adaugate se vad, supravietuiesc inchiderii aplicatiei, iar
        „Porneste conversia" nu face decat sa proceseze o coada deja vizibila.
        """
        if not paths or self.enqueue_worker is not None:
            return

        self.status.setText(f"Se adauga {len(paths)} elemente in coada…")
        self._set_inputs_enabled(False)

        self.enqueue_worker = EnqueueWorker(
            db_path=self.db_path, kind=self.kind_box.currentData(), sources=list(paths)
        )
        self.enqueue_worker.finished.connect(self._on_enqueued)
        self.enqueue_worker.failed.connect(self._on_failed)
        self.enqueue_thread = start(self.enqueue_worker)
        self.enqueue_thread.finished.connect(self._release_enqueue)

    def _on_enqueued(self, waiting: int) -> None:
        self._set_inputs_enabled(True)
        self._reload_table()
        if waiting:
            self.status.setText(
                f"{waiting} in asteptare. Apasa butonul „Porneste conversia”."
            )
        else:
            self.status.setText("Nimic nou de convertit — totul e deja procesat.")

    def _set_inputs_enabled(self, enabled: bool) -> None:
        for widget in (self.choose_files_button, self.choose_folder_button,
                       self.kind_box, self.start_button):
            widget.setEnabled(enabled)

    # -- lot ------------------------------------------------------------

    def _start(self) -> None:
        if self.batch_worker is not None or self.enqueue_worker is not None:
            return

        # Coada e deja in baza de date: fisierele au fost puse acolo cand le-ai
        # ales. Butonul asta nu adauga nimic, doar proceseaza ce asteapta.
        self.batch_worker = BatchWorker(
            db_path=self.db_path,
            out_dir=self.out_dir,
            kind=self.kind_box.currentData(),
        )
        self.batch_worker.started_loading.connect(
            lambda: self.status.setText("Se incarca modelul… (o singura data pentru tot lotul)")
        )
        self.batch_worker.progressed.connect(self._on_progress)
        self.batch_worker.finished.connect(self._on_finished)
        self.batch_worker.failed.connect(self._on_failed)

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # nedeterminat pana la prima pagina
        self._set_inputs_enabled(False)
        self.stop_button.setEnabled(True)

        self.batch_thread = start(self.batch_worker)
        self.batch_thread.finished.connect(self._release_batch)

    def _stop(self) -> None:
        if self.batch_worker is not None:
            self.batch_worker.stop()
            self.stop_button.setEnabled(False)
            self.status.setText("Se opreste dupa pagina curenta…")

    def _on_progress(self, progress) -> None:
        self.progress.setRange(0, progress.total)
        self.progress.setValue(progress.done)
        remaining = _format_duration(progress.seconds_remaining)
        self.status.setText(
            f"{progress.done}/{progress.total} · {progress.current} · "
            f"aproximativ {remaining} ramase"
        )

    def _on_finished(self, jobs: list) -> None:
        """Rezultatele lotului. **Nu** elibereaza firul — vezi `_release_batch`."""
        report.write(jobs, self.out_dir / "raport.html")
        self._fill_table(jobs)
        counts: dict[str, int] = {}
        for job in jobs:
            counts[job.state.value] = counts.get(job.state.value, 0) + 1
        self.status.setText("Gata: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))

    def _on_failed(self, message: str) -> None:
        self.status.setText("A esuat.")
        QMessageBox.critical(self, "Eroare", message)

    def _release_batch(self) -> None:
        """Elibereaza firul de lot, dupa ce chiar s-a oprit.

        Legata de `QThread.finished`, nu de `worker.finished`: al doilea se
        emite in timp ce firul inca ruleaza, iar stergerea referintelor in acel
        moment distruge un `QThread` in executie si opreste procesul.
        """
        self.batch_worker = None
        self.batch_thread = None
        self.progress.setVisible(False)
        self._set_inputs_enabled(True)
        self.stop_button.setEnabled(False)

    def _release_enqueue(self) -> None:
        self.enqueue_worker = None
        self.enqueue_thread = None

    def closeEvent(self, event) -> None:
        """Nu inchide fereastra peste un fir care inca lucreaza.

        Fara asta, inchiderea in mijlocul unui lot distruge firul in executie
        si aplicatia moare cu o eroare in loc sa se inchida curat.
        """
        for worker, thread in (
            (self.batch_worker, self.batch_thread),
            (self.enqueue_worker, self.enqueue_thread),
        ):
            if thread is None:
                continue
            if worker is not None and hasattr(worker, "stop"):
                worker.stop()
            thread.quit()
            if not thread.wait(30000):
                thread.terminate()
                thread.wait()
        event.accept()

    # -- tabel ----------------------------------------------------------

    def _reload_table(self) -> None:
        if not self.db_path.exists():
            return
        with Store(self.db_path) as store:
            self._fill_table(store.all())

    def _fill_table(self, jobs: list[Job]) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            values = (
                job.label,
                job.kind,
                STATE_LABELS.get(job.state, job.state.value),
                job.staves,
                job.systems,
                job.errors,
                f"{job.seconds:.1f}" if job.seconds else None,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem("—" if value is None else str(value))
                if column >= 3:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if job.state is State.FAILED or (job.errors or 0) > 0 or job.message:
                    item.setToolTip(job.message or "are erori de validare")
                self.table.setItem(row, column, item)
            self.table.setCellWidget(row, len(COLUMNS) - 1, self._actions_for(job))
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

    def _actions_for(self, job: Job) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)

        if job.musicxml:
            button = QPushButton("MuseScore")
            button.clicked.connect(lambda _, p=job.musicxml: self._open_in_musescore(Path(p)))
            row.addWidget(button)
            # Corectarea sta langa MuseScore, nu in locul lui: MuseScore e pentru
            # editat partitura, fereastra asta e pentru **legat corectia de poza
            # din care a iesit** — adica pentru a face din ea material de invatat.
            corectare = QPushButton("Corecteaza")
            corectare.clicked.connect(lambda _, j=job: self._corecteaza(j))
            row.addWidget(corectare)
        elif job.detail:
            button = QPushButton("De ce?")
            button.clicked.connect(lambda _, j=job: self._explain(j))
            row.addWidget(button)
        return holder

    def _corecteaza(self, job: Job) -> None:
        """Deschide fereastra de corectare pentru pagina asta."""
        from app.ui.corectare import FereastraCorectare

        if not job.musicxml or not Path(job.musicxml).exists():
            QMessageBox.warning(self, job.label, "Nu exista o transcriere de corectat.")
            return
        try:
            fereastra = FereastraCorectare(job.path, Path(job.musicxml), self)
        except Exception as eroare:
            QMessageBox.critical(
                self, job.label,
                f"Nu s-a putut deschide transcrierea: "
                f"{type(eroare).__name__}: {eroare}",
            )
            return
        fereastra.exec()

    def _explain(self, job: Job) -> None:
        """Arata de ce nu a iesit o partitura din pagina asta."""
        lines = [job.message or ""]
        try:
            detail = json.loads(job.detail or "{}")
        except json.JSONDecodeError:
            detail = {}
        for label, items in (("Probleme", detail.get("problems")),
                             ("Erori", detail.get("errors"))):
            if items:
                lines.append(f"\n{label}:")
                lines.extend(f"  · {item}" for item in items[:15])
        if detail.get("crops"):
            lines.append(f"\nDecupajele portativelor: {Path(detail['crops'][0]).parent}")
        QMessageBox.information(self, job.label, "\n".join(line for line in lines if line))

    # -- actiuni --------------------------------------------------------

    def _open_in_musescore(self, path: Path) -> None:
        try:
            musescore.open_score(path)
        except FileNotFoundError as error:
            QMessageBox.warning(self, "MuseScore", str(error))

    def _open_report(self) -> None:
        path = self.out_dir / "raport.html"
        if not path.exists():
            QMessageBox.information(self, "Raport", "Nu exista inca un raport.")
            return
        webbrowser.open(path.as_uri())

    def _open_folder(self) -> None:
        from PySide6.QtCore import QUrl

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.out_dir)))
