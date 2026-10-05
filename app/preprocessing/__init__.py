"""Etapa 6: pagina bruta -> pagina din care modelul poate citi note.

Trei pasi, in ordinea asta, si ordinea conteaza:

1. **indreptare** — pe scanul din §14, pagina 1 e inclinata cu 2,3°, ceea ce
   inseamna ~90 px de deriva pe latimea paginii, mai mult decat inaltimea unui
   portativ. Fara ea, pasul 2 masoara zgomot;
2. **gasirea liniilor** — de aici stim unde e muzica, la scara paginii, si nu
   din cutii umflate de text;
3. **stergerea textului** — ce nu atinge un portativ nu e muzica.

Toate lucreaza pe imagini si nu stiu nimic despre modelul muzical, conform
regulii de separare din §4.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.preprocessing.deskew import binarize, deskew, estimate_skew, rotate
from app.preprocessing.staff_lines import Band, line_metrics, staff_bands
from app.preprocessing.text import MARGIN_SPACES, Suppression, suppress_text

__all__ = [
    "Band",
    "CleanPage",
    "Suppression",
    "binarize",
    "clean_page",
    "deskew",
    "estimate_skew",
    "line_metrics",
    "rotate",
    "staff_bands",
    "suppress_text",
    "MARGIN_SPACES",
]


@dataclass(frozen=True)
class CleanPage:
    """Pagina curatata, plus tot ce s-a masurat pe drum.

    Masuratorile se intorc, nu se arunca: cand o transcriere iese prost, prima
    intrebare e ce i-am dat modelului sa citeasca (§12.4), iar unghiul si
    numarul de portative gasite sunt exact ce raspunde la ea.
    """

    image: np.ndarray  # cerneala inchisa pe fond deschis, ca pagina originala
    angle: float
    bands: list[Band]
    suppression: Suppression

    @property
    def staves(self) -> int:
        return len(self.bands)


def clean_page(grey: np.ndarray, margin_spaces: float = MARGIN_SPACES) -> CleanPage:
    """Indreapta pagina si sterge tot ce nu tine de un portativ.

    Intoarce imaginea in aceeasi conventie in care a venit — cerneala inchisa
    pe fond deschis — ca restul conductei sa n-o trateze altfel.
    """
    binary = binarize(grey)
    straight, angle = deskew(binary)
    bands = staff_bands(straight)
    result = suppress_text(straight, bands, margin_spaces)
    return CleanPage(
        image=cv2.bitwise_not(result.image),
        angle=angle,
        bands=bands,
        suppression=result,
    )
