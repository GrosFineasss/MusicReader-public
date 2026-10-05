"""Indreptarea paginii.

De ce e primul pas si nu un detaliu cosmetic: pe scanul din §14, pagina 1 e
inclinata cu 2,25°. Peste o latime de 2300 px asta inseamna ~90 px de deriva
verticala — mai mult decat inaltimea unui portativ intreg (~95 px). O linie de
portativ ajunge deci sa treaca prin randuri diferite la stanga si la dreapta
paginii, iar orice masuratoare care se sprijina pe proiectia orizontala — adica
gasirea liniilor, si prin ea tot restul — devine zgomot.

Metoda: se roteste pagina prin unghiuri si se alege cel la care **proiectia
orizontala are varfurile cele mai ascutite**. Cand liniile sunt drepte, fiecare
cade intr-un singur rand si profilul are varfuri inalte si inguste; cand sunt
inclinate, cerneala se imprastie pe zeci de randuri si profilul se aplatizeaza.
Varianta profilului masoara exact asta.

Modulul nu stie nimic despre muzica: primeste o imagine si intoarce alta.
"""

from __future__ import annotations

import cv2
import numpy as np

# Peste atat nu mai e vorba de o pagina asezata strambe pe scanner, ci de alta
# problema — si o rotatie mare ar strica mai mult decat repara.
MAX_ANGLE = 5.0

# Cautarea se face in doua treceri: intai grosier pe tot intervalul, apoi fin
# in jurul castigatorului. O singura trecere fina ar fi de ~10 ori mai lenta,
# fara sa dea alt raspuns.
COARSE_STEP = 0.5
FINE_STEP = 0.05

# Imaginea se micsoreaza inainte de cautare. Unghiul se masoara la fel de bine
# pe o pagina de patru ori mai mica, iar fiecare rotatie devine de 16 ori mai
# ieftina — iar rotatii se fac zeci.
SEARCH_SCALE = 4


def binarize(grey: np.ndarray) -> np.ndarray:
    """Alb-negru cu pragul lui Otsu, cerneala = 255.

    Otsu isi alege pragul din histograma imaginii, deci se descurca si cu
    fotocopiile cenusii din §14.2, unde un prag fix ar manca notele slabe sau
    ar pastra fondul.
    """
    if grey.ndim == 3:
        grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _sharpness(binary: np.ndarray, angle: float) -> float:
    """Cat de ascutite sunt varfurile proiectiei orizontale, la unghiul dat."""
    height, width = binary.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(binary, matrix, (width, height), borderValue=0)
    return float(np.var(rotated.sum(axis=1).astype(np.float64)))


def estimate_skew(binary: np.ndarray, max_angle: float = MAX_ANGLE) -> float:
    """Unghiul cu care trebuie rotita pagina ca sa iasa dreapta, in grade."""
    small = cv2.resize(
        binary,
        (max(binary.shape[1] // SEARCH_SCALE, 1), max(binary.shape[0] // SEARCH_SCALE, 1)),
        interpolation=cv2.INTER_AREA,
    )

    coarse = max(
        np.arange(-max_angle, max_angle + COARSE_STEP / 2, COARSE_STEP),
        key=lambda angle: _sharpness(small, float(angle)),
    )
    fine = max(
        np.arange(coarse - COARSE_STEP, coarse + COARSE_STEP + FINE_STEP / 2, FINE_STEP),
        key=lambda angle: _sharpness(small, float(angle)),
    )
    return round(float(fine), 2)


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Roteste pastrand dimensiunea. Marginile se umplu cu fond, nu cu cerneala.

    `INTER_NEAREST` pentru ca imaginea e deja alb-negru: o interpolare care
    face medii ar introduce griuri, iar pasii de dupa presupun doua valori.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0
    )


def deskew(binary: np.ndarray) -> tuple[np.ndarray, float]:
    """Pagina indreptata, plus unghiul folosit."""
    angle = estimate_skew(binary)
    return rotate(binary, angle), angle
