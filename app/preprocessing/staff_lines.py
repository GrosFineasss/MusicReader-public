"""Unde sunt liniile de portativ, masurate din imagine.

Detectorul YOLO din `app/staff_detection/` gaseste *portative*, dar cutiile lui
sunt umflate de versuri — pe scanul din §14.5, un portativ cu intinderea reala
de ~95 px primeste o cutie de 222-285 px, fiindca inghite textul de dedesubt.
Pentru a sti unde se termina muzica si unde incepe textul, avem nevoie de
liniile insesi, nu de cutii.

Se masoara direct din imagine, deci independent de detector si de scara paginii.
Cifrele masurate pe scan: linii de ~4 px, spatii de ~20 px, portativ de 93-101.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

LINES_PER_STAFF = 5

# Un portativ caruia ii lipsesc linii e tot un portativ: pe o fotocopie se
# pierd des. Masurat pe scan: **portativul de sus al paginii 3 da 3 linii din 5,
# iar ultimul portativ al paginii 1 la fel** — si amandoua erau aruncate cand
# pragul era 4. Consecinta nu era cosmetica: pagina 1 iesea cu 11 portative in
# loc de 12, deci gruparea declarata `--portative-pe-sistem 2` producea 5
# sisteme plus un rest orfan, iar toata pagina se citea strambe.
#
# Ce face sigure trei linii nu e numarul lor, ci **regularitatea**: distanta
# dintre ele era 25,5 px pe o pagina cu pasul 25, si 24,5 pe una cu pasul 24.
# Doua bare de masura alaturate — cazul de care ne temeam — nu cad niciodata la
# exact pasul paginii, si nici nu vin cate trei egal distantate. Deci pragul
# scade la trei, iar in schimb un portativ incomplet e cantarit cu o toleranta
# mult mai stransa (vezi `PARTIAL_SPACING_TOLERANCE`).
MIN_LINES = 3

# Cat de mult poate sa varieze distanta dintre doua linii ale aceluiasi
# portativ. Larg, fiindca liniile sunt trase de mana.
SPACING_TOLERANCE = 0.55

# Cat de mult poate sa varieze pasul unui portativ **incomplet**, caruia i s-au
# gasit mai putin de cinci linii. Mult mai strans decat pentru unul intreg, si
# asta e tot rostul: cinci linii sunt prin ele insele dovada ca e un portativ,
# trei nu sunt, deci trebuie sa cada aproape exact pe pasul paginii ca sa fie
# crezute. Pe scan, cele doua cazuri reale cad la 2% de pas — incap comod.
PARTIAL_SPACING_TOLERANCE = 0.20

# Nucleul orizontal cu care se sterge tot ce nu e linie lunga, exprimat in
# spatii de portativ. Destul de lung cat sa nu supravietuiasca o codita sau o
# litera, destul de scurt cat sa supravietuiasca o linie trasa strambe.
OPENING_SPACES = 8

# Cat de tare trebuie sa iasa un rand in evidenta ca sa fie socotit linie.
# Raportat la percentila 99, nu la maxim: un singur rand foarte incarcat — o
# bara groasa, o pata — ar ridica maximul si ar ascunde liniile adevarate.
ROW_THRESHOLD = 0.30

# Cate coloane se citesc cand se masoara grosimea si spatiul. Fix, nu
# proportional: costul masuratorii nu trebuie sa creasca cu latimea paginii.
SAMPLE_COLUMNS = 200


@dataclass(frozen=True)
class Band:
    """Un portativ: de la linia de sus la cea de jos, plus liniile gasite."""

    top: int
    bottom: int
    lines: tuple[int, ...]

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def spacing(self) -> float:
        """Distanta medie intre linii — unitatea naturala de scara a paginii."""
        if len(self.lines) < 2:
            return 0.0
        return (self.lines[-1] - self.lines[0]) / (len(self.lines) - 1)


def _runs(column: np.ndarray) -> list[tuple[bool, int]]:
    """Sirurile de aceeasi valoare dintr-o coloana, ca (e_cerneala, lungime)."""
    if column.size == 0:
        return []
    changes = np.flatnonzero(np.diff(column.astype(np.int8)))
    starts = np.r_[0, changes + 1]
    ends = np.r_[changes + 1, column.size]
    return [(bool(column[s]), int(e - s)) for s, e in zip(starts, ends)]


def line_metrics(binary: np.ndarray, columns: int = SAMPLE_COLUMNS) -> tuple[int, int]:
    """(grosimea liniei, distanta dintre linii), in pixeli.

    Se citesc din histograma sirurilor verticale: cel mai des intalnit sir de
    cerneala e grosimea liniei, iar cel mai des intalnit sir de fond e spatiul
    dintre linii. Metoda e clasica in OMR si are avantajul ca **nu depinde de
    inclinare** — o linie inclinata cu doua grade tot subtire ramane.

    Se citesc doar `columns` coloane, imprastiate pe toata latimea. O histograma
    are nevoie de esantion, nu de recensamant: cateva sute de coloane dau
    acelasi mod ca toate, iar costul nu mai creste cu latimea paginii — ceea ce
    conteaza, fiindca paginile reale au 2300-2400 px.
    """
    ink: dict[int, int] = {}
    gap: dict[int, int] = {}
    ink_mask = binary > 0
    sample_every = max(ink_mask.shape[1] // max(columns, 1), 1)
    for x in range(0, ink_mask.shape[1], sample_every):
        for is_ink, length in _runs(ink_mask[:, x]):
            target = ink if is_ink else gap
            target[length] = target.get(length, 0) + 1

    if not ink or not gap:
        raise ValueError("pagina nu contine destula cerneala pentru a masura portativele")

    thickness = max(ink, key=lambda length: ink[length])
    # Spatiile foarte mari sunt marginile paginii si golurile dintre sisteme;
    # ne intereseaza modul distributiei, care e spatiul dintre linii.
    spacing = max(gap, key=lambda length: gap[length])
    return thickness, spacing


def line_rows(binary: np.ndarray, spacing: int) -> list[int]:
    """Randurile pe care trece o linie de portativ, cate unul per linie."""
    width = max(int(OPENING_SPACES * spacing), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
    only_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    profile = (only_lines > 0).sum(axis=1).astype(np.float64)
    if not profile.any():
        return []
    strong = profile > ROW_THRESHOLD * np.percentile(profile, 99)

    centres: list[int] = []
    start = None
    for y, on in enumerate(strong):
        if on and start is None:
            start = y
        elif not on and start is not None:
            centres.append((start + y - 1) // 2)
            start = None
    if start is not None:
        centres.append((start + len(strong) - 1) // 2)
    return centres


def merge_split_lines(centres: list[int], pitch: int) -> list[int]:
    """Uneste randurile care sunt de fapt aceeasi linie, gasita in bucati.

    O linie groasa sau intrerupta — obisnuit pe o fotocopie — face profilul sa
    coboare la mijlocul ei, si atunci apare ca doua randuri la 2-8 px distanta.
    Masurat pe pagina 1: din 88 de randuri gasite, o buna parte erau astfel de
    perechi. Un portativ adevarat nu poate avea doua linii mai apropiate decat
    jumatate din distanta lui normala, deci despartirea e sigura.
    """
    if not centres:
        return []
    minimum = max(pitch // 2, 1)
    merged: list[list[int]] = [[centres[0]]]
    for centre in centres[1:]:
        if centre - merged[-1][-1] < minimum:
            merged[-1].append(centre)
        else:
            merged.append([centre])
    return [sum(group) // len(group) for group in merged]


def group_lines(centres: list[int], pitch: int) -> list[Band]:
    """Aduna liniile in portative de cate cinci, dupa distanta dintre ele.

    `pitch` e distanta asteptata **centru-la-centru**, nu spatiul alb dintre
    linii: intre doua centre incap si spatiul, si grosimea unei linii.
    """
    centres = merge_split_lines(centres, pitch)
    if not centres:
        return []

    limit = pitch * (1 + SPACING_TOLERANCE)
    groups: list[list[int]] = [[centres[0]]]
    for previous, current in zip(centres, centres[1:]):
        if current - previous <= limit and len(groups[-1]) < LINES_PER_STAFF:
            groups[-1].append(current)
        else:
            groups.append([current])

    # Un grup a carui distanta intre linii nu seamana cu a paginii nu e un
    # portativ, ci resturi adunate intamplator — text ingramadit, o pata, doua
    # bare alaturate. Se arunca aici, nu mai departe: un portativ inventat ar
    # muta si banda de text, si odata cu ea ce se sterge din imagine.
    bands = []
    for group in groups:
        if len(group) < MIN_LINES:
            continue
        band = _completed(group, pitch)
        complete = len(group) == LINES_PER_STAFF
        tolerance = SPACING_TOLERANCE if complete else PARTIAL_SPACING_TOLERANCE
        if abs(band.spacing - pitch) <= tolerance * pitch:
            bands.append(band)
    return bands


def _completed(lines: list[int], pitch: int) -> Band:
    """Banda unui grup de linii, largita cu cele care lipsesc.

    `top` si `bottom` nu sunt aici o descriere a ce s-a gasit, ci **zona in care
    are voie sa fie muzica**: `suppress_text` sterge cerneala care nu atinge
    nicio banda. Pentru un portativ caruia i s-au gasit doar trei linii,
    marginile liniilor gasite ar lasa pe dinafara exact partea in care liniile
    s-au pierdut — adica si notele de acolo.

    Pe ce parte lipsesc nu se poate sti din grup, deci se lasa loc de amandoua.
    Asimetria costurilor decide: cerneala pastrata in plus e cel mult un rand de
    versuri care ajunge la model, pe cand cerneala stearsa in minus e o nota
    care nu mai exista pentru nimeni.
    """
    missing = max(LINES_PER_STAFF - len(lines), 0)
    return Band(
        top=lines[0] - missing * pitch,
        bottom=lines[-1] + missing * pitch,
        lines=tuple(lines),
    )


def staff_bands(binary: np.ndarray) -> list[Band]:
    """Portativele paginii, de sus in jos. Pagina trebuie sa fie deja dreapta."""
    thickness, spacing = line_metrics(binary)
    pitch = thickness + spacing
    return group_lines(line_rows(binary, spacing), pitch)
