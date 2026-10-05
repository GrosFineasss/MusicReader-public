"""Motorul SMT: detectie YOLO de portative + decodare incrementala + asamblare.

Nu contine logica noua de recunoastere. Aduna la un loc ce era imprastiat in
`scripts/transcribe.py` si il pune in spatele interfetei `Engine`, ca lotul si
interfata grafica sa nu depinda de ce model rulam.

Modelul si detectorul se incarca **o singura data**, la prima transcriere.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from app.recognition.engine import Kind, PageImage, PageResult, StaffCrop
from app.staff_detection.staves import detect_boxes

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "ISMIR-Jazzmus"
YOLO_WEIGHTS = VENDOR / "yolo_weigths" / "yolov11s_20241108.pt"

# Greutatile cu care ruleaza aplicatia, cand exista. Le pune aici
# `scripts/publica_greutati.py`, dintr-un checkpoint masurat; fara fisier,
# aplicatia ramane pe modelul publicat, ca inainte de 2026-10-04.
PRODUS = ROOT / "models" / "produs" / "greutati.pt"
# Pentru scris de mana, separat: antrenarea pe manuscris care ajuta mana strica tiparitul dupa
# ~1.500 de pasi (PROJECT.md §15.66-§15.67), iar greutatile antrenate doar pe tiparit citeau mana mai
# prost decat modelul publicat. Fara fisierul asta, scrisul de mana ramane pe modelul publicat.
PRODUS_MANA = ROOT / "models" / "produs" / "greutati-mana.pt"

STAFF_HEIGHT = 128  # impus de model
MAX_STAFF_WIDTH = 1000

# Peste cat de multa comprimare orizontala merita sa ne plangem. Un portativ
# lung, redus la latimea maxima, e turtit — notele se apropie fata de cum a
# vazut modelul la antrenare.
SQUASH_WARNING_RATIO = 1.15


def _ensure_vendor_on_path() -> None:
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))


class SmtEngine:
    """Recunoastere cu SMT. Potrivit pentru scris de mana; vezi PROJECT.md §3.7
    pentru limitele vocabularului lui `jazzmus-model` pe muzica de cor."""

    name = "smt"

    def __init__(self, threads: int = 1, max_tokens: int | None = None) -> None:
        # Un singur thread e de ~5x mai rapid aici: operatiile modelului sunt
        # mici, iar sincronizarea intre thread-uri costa mai mult decat calculul.
        self.threads = threads
        self.max_tokens = max_tokens
        self._model = None
        self._decoder = None
        self._yolo = None

    # -- incarcare lenesa, o singura data --------------------------------

    def load(self) -> None:
        if self._model is not None:
            return

        _ensure_vendor_on_path()
        import torch
        from ultralytics import YOLO

        from app.recognition.incremental import IncrementalSMT
        from scripts.benchmark_smt import load_model

        torch.set_num_threads(self.threads)
        model, _ = load_model(torch.device("cpu"))
        self._model = model
        self._decoder = IncrementalSMT(model)
        self._yolo = YOLO(str(YOLO_WEIGHTS))

    # -- etapele --------------------------------------------------------

    def detect_staves(self, image_path: Path) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        """Decupeaza portativele, de sus in jos.

        Ordinea verticala conteaza: intr-un sistem coral in partitura stransa,
        portativul de sus poarta S+A si cel de jos T+B.
        """
        from PIL import Image

        image = Image.open(image_path)
        return [
            (box, np.array(image.crop(box).convert("L")))
            for box in detect_boxes(self._yolo, image_path)
        ]

    def _preprocess(self, staff: np.ndarray) -> tuple[object, np.ndarray, float]:
        """Normalizeaza un portativ la inaltimea ceruta de model.

        Pastreaza intentionat aritmetica din `predict.py` al repo-ului de
        referinta, inclusiv taierea latimii la `MAX_STAFF_WIDTH`, care **turteste
        orizontal** portativele lungi in loc sa le scurteze. Nu o schimbam pe
        ghicite: probabil e chiar preprocesarea cu care a fost antrenat modelul,
        iar o abatere „logica" ar putea inrautati transcrierea.

        Intoarcem insa cat de mult s-a turtit, ca sa se vada in raport si sa
        putem masura efectul cand avem partituri reale.
        """
        import cv2

        from jazzmus.dataset.data_preprocessing import convert_img_to_tensor

        natural = int(np.ceil(staff.shape[1] * STAFF_HEIGHT / staff.shape[0]))
        width = min(natural, MAX_STAFF_WIDTH)
        resized = cv2.resize(staff, (width, STAFF_HEIGHT))
        return convert_img_to_tensor(resized), resized, natural / max(width, 1)

    # -- interfata Engine ------------------------------------------------

    def transcribe(self, page: PageImage) -> PageResult:
        # `load()` inainte de import: el pune `vendor/` pe `sys.path`, deci
        # `jazzmus` nu e importabil pana atunci.
        self.load()
        from jazzmus.dataset.tokenizer import untokenize

        started = time.perf_counter()
        result = PageResult(page=page)

        squashed = 0
        for index, (box, staff) in enumerate(self.detect_staves(page.path), 1):
            tensor, resized, squash = self._preprocess(staff)
            if squash >= SQUASH_WARNING_RATIO:
                squashed += 1
            t0 = time.perf_counter()
            tokens = self._decoder.generate(tensor, max_tokens=self.max_tokens)
            result.staves.append(
                StaffCrop(
                    index=index,
                    box=box,
                    image=resized,
                    kern=untokenize(tokens),
                    tokens=len(tokens),
                    taiat=len(tokens) >= self._decoder.plafon(self.max_tokens),
                    seconds=round(time.perf_counter() - t0, 2),
                )
            )

        if result.taiate:
            result.problems.append(
                f"{len(result.taiate)} portative au atins plafonul de decodare "
                f"({self._decoder.plafon(self.max_tokens)} tokeni) si sunt taiate "
                f"la mijloc: {', '.join(str(i) for i in result.taiate)}. "
                "Ce s-a citit din ele nu e o pagina scurta, e o pagina rupta"
            )

        if squashed:
            result.problems.append(
                f"{squashed} portative au fost comprimate orizontal ca sa incapa "
                f"in {MAX_STAFF_WIDTH} px — transcrierea lor poate fi mai slaba"
            )

        if not result.staves:
            result.problems.append("nu s-a detectat niciun portativ")
            result.seconds = round(time.perf_counter() - started, 2)
            return result

        self._assemble(result)
        result.seconds = round(time.perf_counter() - started, 2)
        return result

    def _assemble(self, result: PageResult) -> None:
        """Portative -> partitura SATB -> validare.

        Esecul de aici nu arunca: `**kern`-ul brut si decupajele raman in
        rezultat si sunt exact ce trebuie ca sa intelegem de ce a esuat.
        """
        from app.recognition.page import assemble_page
        from app.validation.rules import Severity, validate

        try:
            page = assemble_page([staff.kern for staff in result.staves],
                                 title=result.page.stem)
        except ValueError as error:
            result.problems.append(str(error))
            return

        result.score = page.score
        result.systems = page.systems
        result.problems.extend(page.problems)

        findings = validate(page.score)
        result.errors = [str(f) for f in findings if f.severity is Severity.ERROR]
        result.warnings = [str(f) for f in findings if f.severity is not Severity.ERROR]


def engine_for(kind: Kind, **kwargs):
    """Alege motorul potrivit tipului de partitura. Singurul loc care ramifica.

    **Amandoua tipurile merg pe `smt-grandstaff`**, si asta nu mai e o ipoteza:
    §12.4 a masurat-o pe o pagina corala reala. `jazzmus-model` nu a emis `*^`
    pe niciunul dintre cele sapte portative, adica a aruncat tacut una din cele
    doua voci ale fiecaruia. Pe partitura stransa — forma in care sunt scrise
    toate partiturile noastre, si tiparite si de mana — asta il face nepotrivit,
    oricat de bine ar citi altfel.

    `SmtEngine` ramane in proiect, dar ca **instrument de masura**, nu ca piesa
    de produs: e singurul model handwritten-native la indemana, deci spune cat
    de greu e scrisul nostru pentru un model deja adaptat la scris de mana
    (§3.9, punctul 4).

    Importul e inauntru fiindca `grandstaff_engine` incarca `vendor/SMT`, si
    nu vrem sa se intample doar pentru ca cineva a importat modulul asta.
    """
    from app.recognition.grandstaff_engine import GrandstaffEngine

    return GrandstaffEngine(**kwargs)


def greutati_produs(kind: Kind = Kind.PRINTED, cale: Path | None = None) -> Path | None:
    """Greutatile aplicatiei pentru felul paginii, daca au fost publicate; altfel `None`.

    `None` = modelul publicat. Tiparitul ia `PRODUS`, scrisul de mana `PRODUS_MANA` — **nu** cade
    pe greutatile de tiparit cand lipsesc ale lui, fiindca pe mana acelea erau mai slabe decat
    modelul publicat (code review, 2026-10-04). Doar aplicatia le ia implicit; uneltele de masura
    (`masoara.py`, `evaluate.py`) raman pe modelul publicat cand nu li se dau greutati, fiindca
    reperele din PROJECT.md sunt masurate asa.
    """
    if cale is None:
        cale = PRODUS_MANA if kind is Kind.HANDWRITTEN else PRODUS
    return cale if cale.is_file() else None
