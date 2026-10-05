"""Pagina -> decupaje de citit cu ochiul, sistem cu sistem.

Nu e o unealta de conducta: nimic din recunoastere n-o foloseste. Serveste
scrierii referintelor (PROJECT.md §15.41). O pagina de 2400x3400 px vazuta
intreaga e nedescifrabila — capetele de nota au 20 de pixeli si nu se poate
spune daca sunt pe linie sau intre linii. Taiata pe sisteme si apoi pe felii
orizontale, aceeasi pagina se citeste la rezolutia ei adevarata.

    .venv/Scripts/python.exe scripts/decupaje.py data/raw/scris-de-mana/pagina1.png \
        --portative-pe-sistem 2 --out data/processed/citire/pagina1

Ce iese: `sistem-01.png` (sistemul intreg, pentru context si numarul de masuri)
si `sistem-01-a.png`, `-b.png`, ... (felii care se suprapun, pentru inaltimi).
Suprapunerea nu e cosmetica: o nota taiata exact de marginea feliei ar fi citita
o data pe jumatate si o data deloc.

Coordonatele se scriu in `decupaje.json`, in sistemul de coordonate al paginii
**indreptate**, ca sa se poata da inapoi la poza originala orice observatie.

Detectia portativelor e cea din `preprocessing/staff_lines.py` — proiectie
orizontala, fara model. Deci ruleaza in doua secunde si nu depinde de greutati
descarcate; pe o pagina scrisa de mana greseste rar, iar cand greseste se vede
imediat pe decupaj, care e tocmai ce urmeaza sa fie privit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.preprocessing.deskew import binarize, estimate_skew  # noqa: E402
from app.preprocessing.staff_lines import Band, staff_bands  # noqa: E402
from app.staff_detection.staves import group_fixed, parse_grouping  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# Cate spatii de portativ se lasa deasupra si dedesubtul sistemului. Trei,
# fiindca deasupra stau liniile suplimentare si triolete, iar dedesubt versurile
# — si versurile spun unde incepe fiecare silaba, deci unde cade fiecare nota.
MARGIN_SPACINGS = 3.0

# Latimea unei felii, in pixeli de pagina. ~1200 fiindca atat se vede fara
# micsorare intr-o citire; mai mult si feliile se micsoreaza, adica se pierde
# exact ce s-a castigat prin taiere.
SLICE_WIDTH = 1200

# Cat se suprapun doua felii vecine, ca fractiune din latimea feliei.
OVERLAP = 0.15

# Cu cat se mareste felia inainte de scriere. Spatierea portativului pe paginile
# astea e de ~24 px: la marime naturala nu se poate spune daca un cap de nota
# sta **pe** linie sau **intre** linii, si toata inaltimea atarna de asta.
# Marirea nu adauga informatie, dar o face vizibila — capul de nota ajunge de
# 48 px, cat sa se vada unde-i trece linia.
MAGNIFY = 2.0

# Decuparea pe portativ, pentru inaltimi. Un portativ are ~100 px inaltime, deci
# incape marit de trei ori intr-o citire — si abia acolo se vede fara ezitare
# daca un cap de nota e pe linia a treia sau in spatiul de deasupra ei.
STAFF_SLICE_WIDTH = 500
STAFF_MAGNIFY = 3.0

# Deasupra si dedesubt: liniile suplimentare urca pana la doua spatii si jumatate
# peste portativ, iar versurile stau imediat dedesubt si spun unde cade silaba.
STAFF_MARGIN_SPACINGS = 2.5

# Latimea benzii albe adaugate in stanga feliei de portativ, pentru numele
# inaltimilor. Se adauga, nu se scrie peste muzica.
LABEL_STRIP = 110

# Cate trepte de grila se deseneaza peste linia de sus si sub cea de jos.
GRILA_PESTE = 4
GRILA_SUB = 4

# Inaltimile, de sus in jos, pornind de la linia de sus a portativului. Cheia
# spune doar de unde se numara: pe cheia Sol linia de sus e F5, pe cheia Fa e A3.
# Restul e alfabetul muzical, coborat cate o treapta la fiecare jumatate de spatiu.
SCARA = "CDEFGAB"
LINIA_DE_SUS = {"G": ("F", 5), "F": ("A", 3), "C": ("G", 4)}


def inaltimi(cheie: str, trepte_sus: int, trepte_jos: int) -> list[tuple[str, float]]:
    """Numele inaltimilor si pozitia lor, in jumatati de spatiu fata de linia de sus.

    Se intoarce de sus in jos. `trepte_sus` numara pozitiile de deasupra liniei
    de sus (linii suplimentare si spatiile dintre ele), `trepte_jos` pe cele de
    sub linia de jos a portativului.
    """
    litera, octava = LINIA_DE_SUS[cheie]
    start = SCARA.index(litera)
    rezultat = []
    # `9`, nu `8`: portativul are cinci linii, adica pozitiile 0..4, care cer
    # pasii 0..8. Cu `8`, `trepte_jos=0` lasa linia de jos pe dinafara — nu se
    # vedea, fiindca valoarea folosita e 4.
    for pas in range(-trepte_sus, 9 + trepte_jos):
        index = start - pas
        nume = SCARA[index % 7]
        oct_curenta = octava + (index // 7)
        rezultat.append((f"{nume}{oct_curenta}", pas / 2.0))
    return rezultat


def rotate_grey(image: np.ndarray, angle: float) -> np.ndarray:
    """Indreapta o imagine in tonuri de gri, cu fond alb la margini.

    `deskew.rotate` umple marginea cu 0 — negru pe o imagine gri, adica o rama
    care inghite notele de la capat. Aici fondul trebuie sa ramana hartie.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255
    )


def complete_bands(bands: list[Band], expected: int) -> tuple[list[Band], list[int]]:
    """Adauga portativele pe care proiectia le-a pierdut, la pasul median.

    Pe pagina 1 si pagina 2 lipseste exact ultimul portativ, si din acelasi
    motiv: liniile de jos sunt inclinate fata de restul paginii (fotocopie de
    manuscris), iar o proiectie pe toata latimea le intinde pana sub prag.
    O indreptare globala nu le poate aduce pe toate deodata.

    Aici nu se ghiceste muzica, ci **unde se taie o poza**. Pasul dintre
    portative e vizibil constant (median ~244 px pe pagina 1, abatere sub 5%),
    iar o taietura gresita se vede pe decupaj din prima privire — spre deosebire
    de o nota ghicita, care nu se vede niciodata. Benzile adaugate sunt marcate
    ca atare in `decupaje.json`.
    """
    if expected <= len(bands) or len(bands) < 2:
        return bands, []
    tops = [band.top for band in bands]
    pitch = float(np.median(np.diff(tops)))
    height = int(np.median([band.bottom - band.top for band in bands]))
    lines = bands[-1].lines
    completed = list(bands)
    added: list[int] = []
    while len(completed) < expected:
        top = round(completed[-1].top + pitch)
        shift = top - lines[0]
        completed.append(
            Band(top=top, bottom=top + height, lines=tuple(line + shift for line in lines))
        )
        added.append(len(completed) - 1)
    return completed, added


def grila(strip, band, cheie: str, top: int):
    """Felia de portativ cu grila de inaltimi desenata, plus numele si pozitiile lor.

    De ce exista: masurat pe pagina 3, citind fara grila am nimerit portativul de
    sus si am gresit tot portativul de jos — dar cu o abatere **constanta** de
    ~1,5 trepte. Nu capetele de nota erau problema, ci asezarea grilei in minte.
    Coordonatele liniilor sunt cunoscute (`staff_bands` le da), deci se deseneaza;
    nu se ghiceste ce se stie deja. Aceeasi hotarare ca la casutele foii de glife.
    """
    inalt, lat = strip.shape[:2]
    panza = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    # `band.top`, nu `band.lines[0]`: cand o linie de sus lipseste din detectie,
    # `_completed` o extrapoleaza in `top`, iar `lines[0]` e a doua linie. Masurat
    # pe portativul 1 din pagina 3 — patru linii gasite din cinci — grila iesea cu
    # o linie mai sus, adica **toata pagina citita cu o terta mai inalt**. Nu s-ar
    # fi vazut fara grila: o transcriere transpusa arata perfect coerenta.
    spacing = band.spacing or 10.0
    linie_sus = band.top - top
    etichete = []
    # Doua trepte peste portativ, nu sase: pe un sistem stans, grilele celor doua
    # portative se suprapuneau in banda de etichete, si acolo un nume putea fi
    # citit ca apartinand celuilalt portativ. Doua trepte acopera liniile
    # suplimentare care apar de fapt in corala.
    for nume, pas in inaltimi(cheie, GRILA_PESTE, GRILA_SUB):
        y = round(linie_sus + pas * spacing)
        if not (0 <= y < inalt):
            continue
        pe_linie = abs(pas - round(pas)) < 0.01
        culoare = (215, 165, 110) if pe_linie else (230, 210, 185)
        cv2.line(panza, (0, y), (lat, y), culoare, 1)
        etichete.append((y, nume, pe_linie))
    return panza, etichete


def eticheteaza(felie, etichete, magnify: float):
    """Adauga in stanga feliei o banda alba cu numele inaltimilor.

    Se adauga, nu se scrie peste muzica: un nume asezat peste portativ ar acoperi
    exact ce trebuie citit. Si se pune pe **fiecare** felie, fiindcă felia `d` se
    citeste la fel de des ca felia `a`.
    """
    panza = cv2.copyMakeBorder(
        felie, 0, 0, LABEL_STRIP, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    for y, nume, pe_linie in etichete:
        y_marit = round(y * magnify)
        if not (0 <= y_marit < panza.shape[0]):
            continue
        cv2.line(panza, (LABEL_STRIP - 12, y_marit), (LABEL_STRIP, y_marit),
                 (215, 165, 110) if pe_linie else (230, 210, 185), 1)
        cv2.putText(panza, nume, (6, y_marit + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (170, 0, 0) if pe_linie else (150, 150, 150), 1, cv2.LINE_AA)
    return panza


def systems_from_bands(bands: list[Band], per_system: int | list[int]) -> list[list[Band]]:
    """Grupeaza portativele in sisteme, refolosind gruparea declarata."""
    boxes = [(0, band.top, 1, band.bottom) for band in bands]
    grouped = group_fixed(boxes, per_system)
    result: list[list[Band]] = []
    index = 0
    for group in grouped:
        result.append(bands[index : index + len(group)])
        index += len(group)
    return result


def slice_columns(width: int, slice_width: int, overlap: float) -> list[tuple[int, int]]:
    """Marginile feliilor orizontale, cu suprapunere.

    Ultima felie se lipeste de marginea din dreapta in loc sa iasa din pagina:
    o felie de 80 px la capat n-ar arata nimic si ar cere o citire in plus.
    """
    if width <= slice_width:
        return [(0, width)]
    step = max(1, round(slice_width * (1 - overlap)))
    starts = list(range(0, width - slice_width + step, step))
    return [(start, min(start + slice_width, width)) for start in starts if start < width]


# Cat de plina trebuie sa fie o coloana ca sa fie bara de masura, ca fractiune
# din inaltimea portativului. 0,95 si nu 0,80: la 0,80 intra si codite lungi cu
# steag. Masurat pe pagina 3 — la 0,80 ies 8 „bare" intr-un portativ cu 4.
BARLINE_FILL = 0.95

# Sub atatia pixeli intre doua coloane pline, e aceeasi bara. 25, nu 8: bara de
# repetitie e groasa+subtire+puncte si iesea numarata de trei ori (masurat pe
# sistemul 2 din pagina 3: 2026, 2040, 2057 — aceeasi bara).
BARLINE_MERGE = 25

# Cat de aproape trebuie sa fie doua bare de pe portative diferite ca sa fie
# aceeasi bara. Manuscrisul nu le trage perfect aliniate.
BARLINE_ALIGN = 20

# O masura mai ingusta de atatea spatii de portativ nu e masura, ci o bara dubla
# sau o bara de repetitie numarata de doua ori.
MIN_MEASURE_SPACINGS = 2.0


def barlines(binary, band, fill: float = BARLINE_FILL) -> list[int]:
    """Coloanele care traverseaza portativul de sus pana jos.

    Se cauta pe **portativ**, nu pe sistem: pe partiturile astea bara e trasa
    separat pe fiecare portativ si nu se continua prin spatiul cu versuri, deci
    o cautare pe toata inaltimea sistemului nu gaseste nimic. Masurat pe pagina
    3: zero bare cautate pe sistem, patru pe portativ.
    """
    strip = binary[band.top : band.bottom, :]
    inalt = max(1, band.bottom - band.top)
    coloane = np.where(strip.sum(axis=0) / 255.0 >= fill * inalt)[0]
    grupuri: list[list[int]] = []
    for x in coloane:
        if grupuri and x - grupuri[-1][-1] <= BARLINE_MERGE:
            grupuri[-1].append(int(x))
        else:
            grupuri.append([int(x)])
    return [int(round(sum(g) / len(g))) for g in grupuri]


def measure_spans(binary, group, width: int, fill: float = BARLINE_FILL):
    """Intervalele de masura ale unui **sistem**, de la o bara la alta.

    Barele se cauta pe fiecare portativ si se reunesc: pe pagina 3, bara
    anacruzei e trasa doar pe portativul de jos, deci o cautare pe portativul de
    sus da 4 masuri acolo unde referinta are 5. Reuniunea le prinde pe amandoua.

    Filtrul de latime scoate ce nu e bara: o codita lunga cu steag umple si ea o
    coloana, dar lasa in urma o „masura" de doua spatii, care nu exista.

    Prima masura incepe de la marginea din stanga (acolo stau cheia si masura),
    iar ultima se termina la marginea din dreapta — care pe pagina 3 **taie**
    muzica, fiindca scanul n-are marja dreapta deloc.
    """
    pe_portativ = [barlines(binary, band, fill) for band in group]
    unite, singure = _incruciseaza(pe_portativ)
    spacing = max((b.spacing for b in group), default=0.0) or 10.0
    taieturi = [0] + unite + [width]
    spans = []
    for st, dr in zip(taieturi, taieturi[1:]):
        if dr - st >= MIN_MEASURE_SPACINGS * spacing:
            spans.append((st, dr))
        elif spans:
            spans[-1] = (spans[-1][0], dr)
    return spans, singure


def _incruciseaza(pe_portativ: list[list[int]]) -> tuple[list[int], list[int]]:
    """Barele gasite pe **toate** portativele sistemului, si cele gasite pe unul.

    O codita lunga cu steag umple si ea o coloana intreaga, dar numai pe
    portativul ei; o bara de masura e trasa pe amandoua. Incrucisarea le
    desparte. Masurat pe pagina 3, sistemul 3: reuniunea da 7 taieturi acolo
    unde referinta are 3 masuri, incrucisarea da exact 3.

    Cele gasite pe un singur portativ **nu se arunca** — bara anacruzei de pe
    pagina 3 e trasa doar pe portativul de jos. Se intorc separat, ca „posibile",
    si se spun in raport, ca sa se poata uita cineva la ele.
    """
    if len(pe_portativ) < 2:
        return (pe_portativ[0] if pe_portativ else []), []
    prima, restul = pe_portativ[0], pe_portativ[1:]
    sigure, singure = [], []
    for x in prima:
        potriviri = [
            min(altul, key=lambda y: abs(y - x), default=None) for altul in restul
        ]
        if all(p is not None and abs(p - x) <= BARLINE_ALIGN for p in potriviri):
            sigure.append((x + sum(potriviri)) // (1 + len(potriviri)))
        else:
            singure.append(x)
    for altul in restul:
        for x in altul:
            if all(abs(x - s) > BARLINE_ALIGN for s in sigure):
                singure.append(x)
    sigure.sort()
    unite: list[int] = []
    for x in sigure:
        if unite and x - unite[-1] <= BARLINE_MERGE:
            unite[-1] = (unite[-1] + x) // 2
        else:
            unite.append(x)
    return unite, sorted(set(singure))


def _cheia(clefs: list[str] | None, index: int) -> str | None:
    return clefs[index] if clefs and 0 <= index < len(clefs) else None


def parse_chei(text: str | None, per_system: int | list[int]) -> list[str] | None:
    """Cheile portativelor, de sus in jos.

    `G` = la fel peste tot. `G/F` = tiparul unui sistem, repetat. `G,F,G,F,...` =
    lista completa. Nimic ghicit: cheia hotaraste ce inseamna fiecare linie, iar
    o cheie gresita muta toata pagina cu o terta fara sa se vada.
    """
    if not text:
        return None
    if "/" in text:
        tipar = [c.strip().upper() for c in text.split("/") if c.strip()]
        return tipar * 40
    litere = [c.strip().upper() for c in text.split(",") if c.strip()]
    if len(litere) == 1:
        return litere * 40
    return litere


def cut(
    image_path: Path,
    out_dir: Path,
    per_system: int | list[int],
    margin_spacings: float = MARGIN_SPACINGS,
    slice_width: int = SLICE_WIDTH,
    overlap: float = OVERLAP,
    expected: int = 0,
    magnify: float = MAGNIFY,
    per_staff: bool = False,
    staff_slice_width: int = STAFF_SLICE_WIDTH,
    staff_magnify: float = STAFF_MAGNIFY,
    staff_margin_spacings: float = STAFF_MARGIN_SPACINGS,
    clefs: list[str] | None = None,
    per_measure: bool = False,
    barline_fill: float = BARLINE_FILL,
) -> dict:
    grey = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if grey is None:
        raise SystemExit(f"nu pot citi imaginea: {image_path}")

    angle = estimate_skew(binarize(grey))
    straight = rotate_grey(grey, angle)
    bands = staff_bands(binarize(straight))
    if not bands:
        raise SystemExit("niciun portativ gasit; pagina e prea slaba sau nu e o partitura")

    gasite = len(bands)
    bands, adaugate = complete_bands(bands, expected)
    systems = systems_from_bands(bands, per_system)
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = straight.shape[:2]

    raport = {
        "imagine": str(image_path),
        "unghi": round(angle, 3),
        "portative": len(bands),
        "portative_gasite": gasite,
        "portative_extrapolate": adaugate,
        "sisteme": [],
    }

    for number, group in enumerate(systems, start=1):
        spacing = max((band.spacing for band in group), default=0.0) or 10.0
        margin = round(margin_spacings * spacing)
        top = max(0, min(band.top for band in group) - margin)
        bottom = min(height, max(band.bottom for band in group) + margin)
        system = straight[top:bottom, 0:width]
        name = f"sistem-{number:02d}"
        cv2.imwrite(str(out_dir / f"{name}.png"), system)

        felii = []
        for letter, (x1, x2) in zip("abcdefghij", slice_columns(width, slice_width, overlap)):
            bucata = system[:, x1:x2]
            etichete_felie: list[tuple[int, str, bool]] = []
            if clefs:
                # Aceeasi grila ca pe decupajele de masura. Fara ea, feliile de
                # sistem sunt singurele care raman de citit cu ochiul liber — si
                # exact ele se folosesc pe sistemele unde barele nu se gasesc.
                colorata = cv2.cvtColor(bucata, cv2.COLOR_GRAY2BGR)
                for band in group:
                    cheie = _cheia(clefs, bands.index(band))
                    if not cheie:
                        continue
                    intreg, ale_lui = grila(bucata, band, cheie, top)
                    colorata = cv2.min(colorata, intreg)
                    etichete_felie.extend(ale_lui)
                bucata = colorata
            if magnify != 1.0:
                bucata = cv2.resize(
                    bucata, None, fx=magnify, fy=magnify, interpolation=cv2.INTER_CUBIC
                )
            if etichete_felie:
                bucata = eticheteaza(bucata, etichete_felie, magnify)
            cv2.imwrite(str(out_dir / f"{name}-{letter}.png"), bucata)
            felii.append({"fisier": f"{name}-{letter}.png", "x1": x1, "x2": x2})

        if per_measure:
            spans, posibile = measure_spans(binarize(straight), group, width, barline_fill)
            masuri = []
            for k, (x1, x2) in enumerate(spans, start=1):
                bucata = cv2.cvtColor(system[:, x1:x2], cv2.COLOR_GRAY2BGR)
                etichete: list[tuple[int, str, bool]] = []
                for pozitie, band in enumerate(group):
                    cheie = _cheia(clefs, bands.index(band))
                    if not cheie:
                        continue
                    intreg, ale_lui = grila(system[:, x1:x2], band, cheie, top)
                    bucata = cv2.min(bucata, intreg)
                    etichete.extend(ale_lui)
                if staff_magnify != 1.0:
                    bucata = cv2.resize(
                        bucata, None, fx=staff_magnify, fy=staff_magnify,
                        interpolation=cv2.INTER_CUBIC,
                    )
                if etichete:
                    bucata = eticheteaza(bucata, etichete, staff_magnify)
                fisier = f"{name}-m{k:02d}.png"
                cv2.imwrite(str(out_dir / fisier), bucata)
                masuri.append({"fisier": fisier, "x1": x1, "x2": x2})
            raport.setdefault("masuri", []).append(
                {"sistem": number, "masuri": masuri, "bare_posibile": posibile}
            )

        raport["sisteme"].append(
            {
                "numar": number,
                "portative": len(group),
                "y1": top,
                "y2": bottom,
                "spatiere": round(spacing, 2),
                "fisier": f"{name}.png",
                "felii": felii,
            }
        )

    if per_staff:
        raport["portative_decupate"] = []
        for index, band in enumerate(bands, start=1):
            spacing = band.spacing or 10.0
            margin = round(staff_margin_spacings * spacing)
            top = max(0, band.top - margin)
            bottom = min(height, band.bottom + margin)
            strip = straight[top:bottom, 0:width]
            cheie = _cheia(clefs, index - 1)
            etichete = []
            if cheie:
                strip, etichete = grila(strip, band, cheie, top)
            name = f"portativ-{index:02d}"
            felii = []
            for letter, (x1, x2) in zip(
                "abcdefghijklmn", slice_columns(width, staff_slice_width, overlap)
            ):
                felie = strip[:, x1:x2]
                if staff_magnify != 1.0:
                    felie = cv2.resize(
                        felie, None, fx=staff_magnify, fy=staff_magnify,
                        interpolation=cv2.INTER_CUBIC,
                    )
                if etichete:
                    felie = eticheteaza(felie, etichete, staff_magnify)
                cv2.imwrite(str(out_dir / f"{name}-{letter}.png"), felie)
                felii.append({"fisier": f"{name}-{letter}.png", "x1": x1, "x2": x2})
            raport["portative_decupate"].append(
                {"numar": index, "y1": top, "y2": bottom,
                 "extrapolat": index - 1 in adaugate, "felii": felii}
            )

    (out_dir / "decupaje.json").write_text(
        json.dumps(raport, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return raport


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Taie o pagina in sisteme si felii, pentru citit cu ochiul."
    )
    parser.add_argument("imagine", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="folderul de iesire")
    parser.add_argument(
        "--portative-pe-sistem", type=parse_grouping, default=2,
        help="un numar, sau o lista ca 2,2,3 cand pagina nu e uniforma",
    )
    parser.add_argument(
        "--margine", type=float, default=MARGIN_SPACINGS,
        help="cate spatii de portativ se lasa deasupra si dedesubt (implicit 3)",
    )
    parser.add_argument(
        "--asteptate", type=int, default=0, metavar="N",
        help="cate portative are pagina; ce lipseste se extrapoleaza si se spune",
    )
    parser.add_argument("--latime-felie", type=int, default=SLICE_WIDTH)
    parser.add_argument("--suprapunere", type=float, default=OVERLAP)
    parser.add_argument(
        "--marire", type=float, default=MAGNIFY,
        help="de cate ori se mareste felia la scriere (implicit 2)",
    )
    parser.add_argument(
        "--pe-portativ", action="store_true",
        help="scrie si decupaje portativ cu portativ, marite, pentru inaltimi",
    )
    parser.add_argument(
        "--prag-bara", type=float, default=BARLINE_FILL, metavar="F",
        help="cat de plina trebuie sa fie coloana ca sa fie bara (implicit 0,95). "
             "Pe pagini cu bare trase scurt, coboara-l.",
    )
    parser.add_argument(
        "--pe-masura", action="store_true",
        help="taie fiecare portativ si pe masuri, dupa barele gasite",
    )
    parser.add_argument(
        "--chei", default=None, metavar="G,F,G,F",
        help="cheia fiecarui portativ, de sus in jos; cu ea se deseneaza grila de "
             "inaltimi peste decupaje. O singura litera inseamna la fel peste tot, "
             "iar `G/F` se repeta pe sisteme.",
    )
    args = parser.parse_args()

    raport = cut(
        args.imagine, args.out, args.portative_pe_sistem,
        args.margine, args.latime_felie, args.suprapunere, args.asteptate, args.marire,
        args.pe_portativ, clefs=parse_chei(args.chei, args.portative_pe_sistem),
        per_measure=args.pe_masura, barline_fill=args.prag_bara,
    )
    print(f"unghi {raport['unghi']}° | {raport['portative']} portative | "
          f"{len(raport['sisteme'])} sisteme")
    for sistem in raport.get("masuri", []):
        rand = f"  sistem {sistem['sistem']:2d}: {len(sistem['masuri'])} masuri"
        if sistem["bare_posibile"]:
            rand += (f" | bare gasite pe un singur portativ, de privit: "
                     f"{sistem['bare_posibile']}")
        print(rand)
    if raport["portative_extrapolate"]:
        lipsa = len(raport["portative_extrapolate"])
        print(f"  ATENTIE: {lipsa} portativ(e) nu s-au gasit si s-au extrapolat la pasul median;")
        print(f"  uita-te pe decupajele lor inainte sa citesti ceva de acolo")
    for sistem in raport["sisteme"]:
        print(f"  sistem {sistem['numar']:2d}: {sistem['portative']} portative, "
              f"y {sistem['y1']}-{sistem['y2']}, {len(sistem['felii'])} felii")


if __name__ == "__main__":
    main()
