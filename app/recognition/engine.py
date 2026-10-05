"""Interfata comuna a motoarelor de recunoastere.

Rostul acestui modul e sa fie **singurul loc** care stie ce model ruleaza.
Motorul de lot, raportul si interfata grafica primesc `PageResult` si nu au de
unde sti daca in spate e SMT, alt model, sau o unealta externa.

De ce conteaza: recunoasterea e partea nevalidata a proiectului. Cand schimbam
modelul — si e probabil sa-l schimbam, vezi PROJECT.md §3.9 — vrem sa se schimbe
o clasa, nu tot lantul de deasupra.

Incarcarea modelului e lenesa si se face **o singura data per motor**. Pentru un
lot de 500 de pagini, reincarcarea la fiecare fisier ar insemna zeci de minute
irosite.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np


class Kind(str, Enum):
    """Ce fel de partitura e — decide pe ce conducta merge.

    Se stabileste explicit, din eticheta folderului de intrare. Un clasificator
    automat care greseste trimite partitura pe conducta gresita si strica
    rezultatul in tacere; o eticheta gresita se vede si se corecteaza.
    """

    HANDWRITTEN = "handwritten"
    PRINTED = "printed"


@dataclass(frozen=True)
class PageImage:
    """O pagina de procesat.

    `page` exista pentru ca un PDF scanat are mai multe pagini in acelasi
    fisier: identitatea unei unitati de lucru e (cale, pagina), nu doar calea.
    """

    path: Path
    kind: Kind
    page: int = 0

    @property
    def stem(self) -> str:
        """Nume de fisier unic pentru aceasta pagina.

        Numele simplu nu ajunge: `collect()` parcurge recursiv, deci
        `piesa1/scan.pdf` si `piesa2/scan.pdf` ar produce aceleasi iesiri, iar a
        doua le-ar suprascrie pe ale primei — in tacere, cu ambele randuri din
        raport aratand spre acelasi fisier. Adaugam o amprenta scurta a caii,
        care le desparte fara sa faca numele ilizibil.
        """
        base = self.path.stem if self.page == 0 else f"{self.path.stem}_p{self.page + 1}"
        return f"{base}_{self.path_key}"

    @property
    def path_key(self) -> str:
        """Sase caractere derivate din calea completa. Stabile intre rulari."""
        return hashlib.sha256(str(self.path.resolve()).encode("utf-8")).hexdigest()[:6]

    def __str__(self) -> str:
        return self.path.name if self.page == 0 else f"{self.path.name}#{self.page + 1}"


@dataclass
class StaffCrop:
    """Un portativ decupat, cu locul lui pe pagina.

    Se pastreaza pentru raport: cand transcrierea iese prost, primul lucru de
    verificat e daca detectia de portative a lucrat corect.
    """

    index: int
    box: tuple[int, int, int, int]
    image: np.ndarray
    kern: str = ""
    tokens: int = 0
    seconds: float = 0.0
    #: decodarea s-a oprit fiindca a atins plafonul, nu fiindca a terminat.
    #: Portativul e taiat la mijloc, iar `kern`-ul lui se termina oriunde
    #: (§15.36). Fara steag, o pagina taiata arata exact ca una citita.
    taiat: bool = False


@dataclass
class PageResult:
    """Ce a iesit dintr-o pagina.

    `score` poate lipsi chiar daca portativele au fost citite: asamblarea in
    partitura SATB poate esua separat (de exemplu daca modelul nu a emis `*^`).
    Pastram si atunci decupajele si `**kern`-ul brut — tocmai ele spun de ce.
    """

    page: PageImage
    staves: list[StaffCrop] = field(default_factory=list)
    score: object | None = None
    systems: int = 0
    problems: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.score is not None and not self.errors

    @property
    def kern(self) -> str:
        return "!!linebreak\n".join(staff.kern for staff in self.staves)

    @property
    def taiate(self) -> list[int]:
        """Portativele oprite de plafonul de decodare, nu de sfarsitul muzicii.

        §15.36: o iesire care atinge plafonul arata exact ca una scurta si
        corecta. Lista asta e singurul loc unde diferenta se poate citi.
        """
        return [staff.index for staff in self.staves if staff.taiat]


class Engine(Protocol):
    """Contractul pe care se sprijina tot ce e deasupra recunoasterii."""

    name: str

    def transcribe(self, page: PageImage) -> PageResult:
        """Citeste o pagina. Nu ridica exceptii pentru partituri gresite —
        acelea se raporteaza in `PageResult.problems` si `.errors`."""
        ...
