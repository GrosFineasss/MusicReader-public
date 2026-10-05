"""Bucla de procesare a lotului.

Doua reguli care dau forma acestui modul:

* **Modelul se incarca o singura data.** La 500 de pagini, reincarcarea per
  fisier ar insemna ~40 de minute irosite.
* **Un fisier stricat nu opreste lotul.** Orice exceptie il marcheaza `failed`
  cu mesajul ei si trece mai departe. Un lot de peste noapte nu are voie sa
  moara la fisierul 7 si sa te trezesti dimineata cu nimic.

`progress` si `cancel` exista de la inceput, desi acum ruleaza din linia de
comanda: interfata grafica din etapa 2 ruleaza **exact acest cod** intr-un fir
de executie separat, fara sa-l modifice.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.batch import inputs
from app.batch.store import Job, State, Store, digest_of
from app.recognition.engine import Engine, Kind, PageImage, PageResult

ProgressCallback = Callable[["Progress"], None]


def enqueue(store: Store, roots: list[Path], kind: Kind) -> int:
    """Adauga fisiere in coada. Intoarce cate unitati de lucru sunt de facut.

    Separata de `Runner` fiindca nu are nevoie de model: interfata grafica pune
    fisierele in coada imediat ce le alegi, cu mult inainte sa existe un motor.

    Aceeasi regula ca la procesare: **un fisier stricat nu opreste lotul.** Un
    PDF criptat sau trunchiat, ori unul pe care nu-l putem citi, se inregistreaza
    ca esuat si se trece mai departe. Altfel un singur fisier prost ar ucide un
    lot de peste noapte inainte sa converteasca vreo pagina.
    """
    for root in roots:
        try:
            pages = inputs.collect([root], kind)
        except Exception as error:  # noqa: BLE001 - esecul ramane local
            _record_unreadable(store, Path(root), kind, error)
            continue

        # Amprenta se calculeaza o data pe fisier, nu o data pe pagina: un PDF
        # scanat de 200 de pagini si 100 MB ar fi citit integral de 200 de ori.
        digests: dict[Path, str] = {}
        for page in pages:
            try:
                if page.path not in digests:
                    digests[page.path] = digest_of(page.path)
                store.add(page.path, page.page, kind.value, digests[page.path])
            except Exception as error:  # noqa: BLE001
                _record_unreadable(store, page.path, kind, error, page.page)

    return len(store.pending())


def _record_unreadable(
    store: Store, path: Path, kind: Kind, error: Exception, page: int = 0
) -> None:
    """Retine ca fisierul nu a putut fi nici macar pus in coada."""
    try:
        store.add_failed(path, page, kind.value, f"{type(error).__name__}: {error}")
    except Exception:  # noqa: BLE001 - daca nici asta nu merge, nu avem ce face
        pass


def output_paths(page: PageImage, out_dir: Path) -> tuple[Path, Path, Path]:
    """Unde ajung MusicXML-ul, `**kern`-ul brut si folderul de decupaje."""
    out_dir = Path(out_dir)
    return (
        out_dir / f"{page.stem}.musicxml",
        out_dir / f"{page.stem}.krn",
        out_dir / "crops",
    )


def write_outputs(
    page: PageImage, result: PageResult, out_dir: Path, save_crops: bool = True
) -> list[Path]:
    """Scrie tot ce ajuta la diagnostic, nu doar rezultatul final.

    `**kern`-ul brut si decupajele se salveaza **si cand asamblarea a esuat** —
    tocmai atunci sunt cele mai utile. Intoarce caile decupajelor scrise.

    Folosita si de lot, si de `scripts/transcribe.py`, ca sa nu ajungem cu doua
    variante ale acelorasi iesiri.
    """
    xml_path, kern_path, crop_dir = output_paths(page, out_dir)
    kern_path.parent.mkdir(parents=True, exist_ok=True)
    kern_path.write_text(result.kern, encoding="utf-8")

    written: list[Path] = []
    if save_crops and result.staves:
        import cv2

        crop_dir.mkdir(parents=True, exist_ok=True)
        for staff in result.staves:
            if staff.image is None:
                continue
            target = crop_dir / f"{page.stem}_staff{staff.index:02d}.png"
            cv2.imwrite(str(target), staff.image)
            written.append(target)

    if result.score is not None:
        from app.musicxml import export

        export.write(result.score, xml_path)

    return written


@dataclass
class Progress:
    done: int
    total: int
    current: str
    seconds_elapsed: float
    last: PageResult | None = None

    @property
    def seconds_remaining(self) -> float | None:
        """Estimare din media de pana acum. `None` la primul fisier."""
        if self.done == 0:
            return None
        return self.seconds_elapsed / self.done * (self.total - self.done)


class Runner:
    def __init__(
        self,
        store: Store,
        engine: Engine,
        out_dir: Path,
        save_crops: bool = True,
    ) -> None:
        self.store = store
        self.engine = engine
        self.out_dir = Path(out_dir)
        self.save_crops = save_crops
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(self, roots: list[Path], kind: Kind) -> int:
        return enqueue(self.store, roots, kind)

    def run(
        self,
        progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
    ) -> list[Job]:
        jobs = self.store.pending()
        started = time.perf_counter()

        for index, job in enumerate(jobs):
            if cancel is not None and cancel.is_set():
                break

            page = PageImage(path=job.path, kind=Kind(job.kind), page=job.page)
            self.store.mark_running(job.path, job.page)
            result = self._process(page)

            if progress is not None:
                progress(
                    Progress(
                        done=index + 1,
                        total=len(jobs),
                        current=str(page),
                        seconds_elapsed=time.perf_counter() - started,
                        last=result,
                    )
                )

        return self.store.all()

    def _process(self, page: PageImage) -> PageResult | None:
        try:
            image_path = inputs.resolve(page, self.out_dir)
            resolved = PageImage(path=image_path, kind=page.kind, page=0)
            result = self.engine.transcribe(resolved)
        except Exception as error:  # noqa: BLE001 - orice esec ramane local
            self.store.mark_failed(page.path, page.page, f"{type(error).__name__}: {error}")
            return None

        crops = write_outputs(page, result, self.out_dir, self.save_crops)
        xml_path, _, _ = output_paths(page, self.out_dir)
        self.store.mark_done(
            page.path,
            page.page,
            systems=result.systems,
            staves=len(result.staves),
            errors=len(result.errors),
            warnings=len(result.warnings),
            seconds=result.seconds,
            message="; ".join(result.problems) or None,
            musicxml=str(xml_path) if result.score is not None else None,
            detail=json.dumps(
                {
                    "problems": result.problems,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    # Doar decupajele scrise efectiv: o cale catre un fisier
                    # inexistent ar trimite raportul dupa imagini fantoma.
                    "crops": [str(path) for path in crops],
                },
                ensure_ascii=False,
            ),
        )
        return result
