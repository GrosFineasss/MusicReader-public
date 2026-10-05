"""Scoate versurile din imagine, ca modelul sa vada doar muzica.

Cerinta, stabilita pe 2026-08-29: **masina scrie notele, tu scrii versurile.**
Nu trebuie sa citim textul cantat — trebuie ca el sa nu strice citirea notelor.
Pe manuscrisele din §14 textul e scris intre portativele sistemului si sub ele,
deci intra in decupajul dat modelului, iar `smt-grandstaff` e antrenat pe grand
staff de pian tiparit si n-a vazut niciodata text sub un portativ.

Regula e simpla si se sprijina pe geometria paginii, nu pe recunoasterea
scrisului: **cerneala legata de un portativ ramane, restul se sterge.** Notele,
coditele, liniile suplimentare si barele de masura ating portativul sau ies
putin din el; versurile stau in banda dintre portative si nu-l ating.

Ce se pierde odata cu textul: indicatii de tempo, numere de masura, titlul —
adica exact lucrurile pe care oricum nu le citim in v1. Ce **nu** avem voie sa
pierdem sunt notele, si de asta banda pastrata e mai larga decat portativul.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.preprocessing.staff_lines import Band

# Cat se largeste portativul, in spatii de portativ, inainte de a decide ce e
# muzica. Notele cu linii suplimentare si coditele ies din portativ; cu o
# margine prea stransa le-am taia, cu una prea larga am pastra versurile.
MARGIN_SPACES = 1.0

# Marginea de mai sus nu ajunge pentru notele cu doua-trei linii suplimentare,
# si asta nu e un caz rar: in partitura stransa corala, **tenorul** sta deasupra
# portativului de jos (Mi4-Sol4 = doua-trei linii), altistul si basul coboara
# sub portativele lor. Gasit pe 2026-10-02 la verificarea paginilor Bach
# randate: pe `bwv104.6`, cinci note de tenor sterse dintr-un singur sistem,
# ramasa doar linia suplimentara. Aceeasi stergere se intampla la inferenta, pe
# orice pagina citita cu `--curata`.
#
# Deosebirea de versuri nu e distanta, ci **forma**: o nota cu codita are o
# trasatura verticala dreapta, pe care literele n-o au (tiparite, au ~1,5 spatii
# cu tot cu codite). Deci o componenta care nu atinge banda e pastrata daca e
# inalta cat o nota, are o asemenea trasatura si sta la cel mult `REACH_SPACES`
# de un portativ.
#
# Codita nu e de 3,5 spatii, cum spun manualele: cu doua voci pe portativ,
# MuseScore le scurteaza. Masurat pe `bwv10.7`: 34 px la un spatiu de 13,75,
# adica 2,47 spatii — sub un prag de 2,5, deci nota tot se stergea.
STEM_SPACES = 2.0
NOTE_HEIGHT_SPACES = 2.5
REACH_SPACES = 6.0
# Ce insoteste o nota pastrata asa nu atinge nici el portativul, si se pastreaza
# odata cu ea: alteratia din stanga (ingusta, inalta de doua spatii sau mai
# mult), punctul de prelungire din dreapta, liniile suplimentare de sub ea.
ACCIDENTAL_GAP_SPACES = 2.0
DOT_GAP_SPACES = 1.0


@dataclass(frozen=True)
class Suppression:
    """Ce a ramas si ce s-a sters — raportat, nu facut in tacere."""

    image: np.ndarray
    removed: int
    kept: int

    @property
    def ratio(self) -> float:
        total = self.removed + self.kept
        return self.removed / total if total else 0.0


def music_zones(bands: list[Band], margin: float, height: int) -> list[tuple[int, int]]:
    """Benzile in care avem voie sa presupunem ca e muzica."""
    zones = []
    for band in bands:
        top = max(int(band.top - margin), 0)
        bottom = min(int(band.bottom + margin), height)
        zones.append((top, bottom))
    return zones


def _overlaps(top: int, bottom: int, zones: list[tuple[int, int]]) -> bool:
    return any(top < zone_bottom and bottom > zone_top for zone_top, zone_bottom in zones)


def longest_vertical_run(mask: np.ndarray) -> int:
    """Cel mai lung sir de pixeli aprinsi pe o coloana — lungimea unei codite."""
    run = np.zeros(mask.shape[1], dtype=np.int32)
    best = 0
    for row in mask:
        run = (run + 1) * row
        best = max(best, int(run.max(initial=0)))
    return best


def _ink_in(integral: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> bool:
    """E cerneala pastrata in dreptunghiul dat? Pe imaginea integrala, deci O(1)."""
    height, width = integral.shape[0] - 1, integral.shape[1] - 1
    x1, x2 = max(int(x1), 0), min(int(x2), width)
    y1, y2 = max(int(y1), 0), min(int(y2), height)
    if x1 >= x2 or y1 >= y2:
        return False
    total = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    return total > 0


def _companion(box, integral: np.ndarray, pitch: float) -> bool:
    """Componenta e alteratia, punctul sau linia suplimentara a unei note pastrate?

    Nota poate fi pastrata de banda portativului sau salvata de codita ei; ce o
    insoteste sta oricum in afara benzii — punctul unei note de alto de sub
    portativ, de pilda — si s-ar sterge singur, desi nota ramane.
    """
    x, y, w, h = box
    # Alteratia: ingusta, inalta de peste doua spatii, cu nota imediat in
    # dreapta. Latimea e strajuita strans: pe pagina 3, un „A" scris de mana in
    # versuri (1,5 spatii lat) trecea drept alteratie la o margine de 1,5.
    if h >= 2.2 * pitch and w <= 1.2 * pitch and _ink_in(
        integral, x + w, y, x + w + ACCIDENTAL_GAP_SPACES * pitch, y + h
    ):
        return True
    # Punctul de prelungire: mic, aproape rotund, lipit de capul notei din
    # stanga. Fara conditia de forma, praful unei fotocopii trecea drept puncte.
    round_dot = 0.2 * pitch <= min(w, h) and max(w, h) <= 0.6 * pitch and max(w, h) <= 1.6 * min(w, h)
    if round_dot and _ink_in(
        integral, x - DOT_GAP_SPACES * pitch, y - pitch / 4, x, y + h + pitch / 4
    ):
        return True
    # Linia suplimentara: subtire, cel putin cat un cap de nota, cu capul pe ea
    # sau langa ea. Liniutele din versuri sunt mai scurte de un spatiu.
    if h <= 0.4 * pitch and pitch <= w <= 3.5 * pitch and _ink_in(
        integral, x, y - pitch, x + w, y + h + pitch
    ):
        return True
    return False


def _rescued(
    binary: np.ndarray, stats, labels, doomed, pitch: float, reach: list[tuple[int, int]]
) -> np.ndarray:
    """Ce componente condamnate sunt de fapt note departe de portativ, cu ce le insoteste."""
    rescued = np.zeros_like(doomed)
    for label in np.flatnonzero(doomed):
        x, y, w, h = (int(v) for v in stats[label, :4])
        if h < NOTE_HEIGHT_SPACES * pitch or not _overlaps(y, y + h, reach):
            continue
        mask = labels[y:y + h, x:x + w] == label
        if longest_vertical_run(mask) >= STEM_SPACES * pitch:
            rescued[label] = True

    kept = (binary > 0) & ~(doomed & ~rescued)[labels]
    integral = cv2.integral(kept.astype(np.uint8))
    for label in np.flatnonzero(doomed & ~rescued):
        x, y, w, h = (int(v) for v in stats[label, :4])
        if _overlaps(y, y + h, reach) and _companion((x, y, w, h), integral, pitch):
            rescued[label] = True
    return rescued


def suppress_text(
    binary: np.ndarray, bands: list[Band], margin_spaces: float = MARGIN_SPACES
) -> Suppression:
    """Sterge cerneala care nu atinge niciun portativ.

    Daca nu s-a gasit niciun portativ, imaginea se intoarce neatinsa. A sterge
    tot pe baza unei detectii care a esuat ar fi cea mai proasta iesire
    posibila: o pagina goala din care modelul n-ar avea ce citi, fara ca nimeni
    sa stie de ce.
    """
    if not bands:
        return Suppression(binary, removed=0, kept=0)

    pitch = max(band.spacing for band in bands) or 1.0
    zones = music_zones(bands, margin_spaces * pitch, binary.shape[0])

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    doomed = np.zeros(count, dtype=bool)
    for label in range(1, count):
        top = stats[label, cv2.CC_STAT_TOP]
        bottom = top + stats[label, cv2.CC_STAT_HEIGHT]
        if not _overlaps(int(top), int(bottom), zones):
            doomed[label] = True

    reach = music_zones(bands, REACH_SPACES * pitch, binary.shape[0])
    doomed &= ~_rescued(binary, stats, labels, doomed, pitch, reach)
    removed = int(doomed.sum())

    cleaned = binary.copy()
    cleaned[doomed[labels]] = 0
    return Suppression(cleaned, removed=removed, kept=count - 1 - removed)
