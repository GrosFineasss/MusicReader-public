"""Foaia de glife: o pagina de tiparit, pe care cineva scrie simbolurile de mana.

Etapa 11, partea de scriitor-independenta (PROJECT.md §15.12, §15.14). Cele ~500
de partituri sunt fiecare in alt scris, deci modelul trebuie sa citeasca **maini
pe care nu le-a vazut**. Singura sursa de scrisuri variate la scara, curata
comercial, e Smashcima alimentat cu glife proprii — iar glifele proprii incep
aici: fiecare om completeaza o foaie, si foaia lui devine un **stil** in
repository.

    python scripts/foaie_glife.py --out data/glife/foaie.png

Ce se scrie alaturi de imagine: un `foaie.json` cu **coordonatele exacte ale
fiecarei casute** si eticheta ei. Segmentarea de dupa nu ghiceste nimic — noi am
desenat grila, deci stim unde e fiecare simbol. E aceeasi hotarare ca la
`--portative-pe-sistem` si la eticheta de scriitor: informatia care se stie la
producere se **scrie**, nu se deduce mai tarziu.

Etichetele sunt cele pe care le cere Smashcima — nume SMuFL (`SmuflLabels`) plus
cateva proprii lui (`SmashcimaLabels`), verificate pe versiunea 1.1.2. Ce nu e in
scopul v1 (§5.2) nu apare pe foaie: fara articulatii, fara dinamici, fara
ornamente.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# A4 la 300 dpi. Se tipareste la marime reala, fara „fit to page": marimea
# scrisului fata de casuta e o informatie pe care o folosim mai tarziu.
DPI = 300
WIDTH = 2480
HEIGHT = 3508
MARGIN = 150

# Cate exemplare din fiecare simbol. Trei, fiindca variatia dintre doua scrieri
# ale aceleiasi persoane e tot informatie utila — repository-ul alege aleator
# dintre ele, deci un singur exemplar ar face fiecare pagina sintetica identica.
REPETITIONS = 3

# (eticheta pentru repository, numele pe care il vede omul care scrie)
#
# 36 de simboluri, nu doar cele 22 de la inceput: repository-ul nu poate
# sintetiza un simbol pentru care n-are nicio glifa scrisa de mana, deci orice
# lipsa de aici devine o gaura in datele de antrenare. S-au adaugat cifrele de
# masura complete (0-9, plus C si C taiat) fiindca o masura ca 6/8 sau 12/8 e
# curenta in corala si nu se poate compune din 2, 3 si 4; saisprezecimile
# (steaguri si pauza) fiindca apar in orice pasaj mai rapid; carligul de bara,
# fiindca ritmul punctat il cere; si legato-ul, care in cor leaga silabele.
# Restul scopului v1 (§5.2) ramane afara: fara articulatii, dinamici, ornamente.
SYMBOLS: list[tuple[str, str]] = [
    ("noteheadWhole", "cap de nota intreaga"),
    ("noteheadHalf", "cap de doime"),
    ("noteheadBlack", "cap de patrime (plin)"),
    ("stem", "codita"),
    ("augmentationDot", "punct de prelungire"),
    ("flag8thUp", "steag de optime, in sus"),
    ("flag8thDown", "steag de optime, in jos"),
    ("flag16thUp", "steag de saisprezecime, in sus"),
    ("flag16thDown", "steag de saisprezecime, in jos"),
    ("beam", "bara de grup"),
    ("beamHook", "carlig de bara (ritm punctat)"),
    ("restWhole", "pauza de intreaga"),
    ("restHalf", "pauza de doime"),
    ("restQuarter", "pauza de patrime"),
    ("rest8th", "pauza de optime"),
    ("rest16th", "pauza de saisprezecime"),
    ("gClef", "cheia Sol"),
    ("fClef", "cheia Fa"),
    ("accidentalSharp", "diez"),
    ("accidentalFlat", "bemol"),
    ("accidentalNatural", "becar"),
    ("legerLine", "linie suplimentara"),
    ("barlineSingle", "bara de masura"),
    ("slur", "legato / arc"),
    ("timeSigCommon", "masura comuna (C)"),
    ("timeSigCutCommon", "alla breve (C taiat)"),
    ("timeSig0", "cifra 0"),
    ("timeSig1", "cifra 1"),
    ("timeSig2", "cifra 2"),
    ("timeSig3", "cifra 3"),
    ("timeSig4", "cifra 4"),
    ("timeSig5", "cifra 5"),
    ("timeSig6", "cifra 6"),
    ("timeSig7", "cifra 7"),
    ("timeSig8", "cifra 8"),
    ("timeSig9", "cifra 9"),
]

INSTRUCTIONS = [
    "Scrie fiecare simbol de trei ori, cate unul in fiecare casuta.",
    "Scrie natural, cum ai scrie pe o partitura — nu caligrafic.",
    "Foloseste pix sau stilou negru ori albastru inchis. Nu creion.",
    "Nu iesi din casuta si nu atinge marginile ei.",
    "Daca gresesti, lasa casuta goala si treci mai departe.",
]


# Simbolurile pentru care **nu ajunge numele**. Prima foaie completata a
# aratat-o: „cap de doime" a fost citit firesc ca „nota doime", si s-au scris
# note intregi in loc de parti (§15.22). Sapte casute din 36.
#
# Ce le face diferite: sunt **bucati** dintr-o nota, nu simboluri de sine
# statatoare. O cheie Sol e o cheie Sol; un cap de patrime fara codita nu apare
# nicaieri intr-o partitura, deci nimeni n-are cum sa-l aiba in minte.
#
# Deci primesc o casuta de **exemplu tiparit**, cu piesa ceruta desenata plin si
# ce nu trebuie desenat punctat. Oamenii copiaza ce vad; o instructiune scrisa a
# fost deja inteleasa altfel o data.
GRI = 130
PUNCTAT = 200

# Fontul de referinta SMuFL. Licenta OFL, deci se poate livra; se descarca cu
# `scripts/ia_bravura.py`. Sta in `vendor/`, care nu e in git — de aceea tot ce
# foloseste fontul are revenire la primitivele PIL de mai jos. §15.22 spunea
# „pe masina asta nu exista niciun font muzical"; acum poate exista, legal.
BRAVURA = ROOT / "vendor" / "bravura" / "Bravura.otf"
SMUFL_NUME = ROOT / "vendor" / "bravura" / "glyphnames.json"

# In Bravura un spatiu de portativ e un sfert de em, deci marimea de font care
# da un spatiu de `n` pixeli e `4n`. Un cap de nota are inaltimea unui spatiu.
EM_PE_SPATIU = 4


def _bravura(marime: int):
    """Fontul, la marimea data, sau None daca nu e descarcat."""
    from PIL import ImageFont

    if not BRAVURA.exists():
        return None
    cheie = round(marime)
    if cheie not in _FONTURI:
        _FONTURI[cheie] = ImageFont.truetype(str(BRAVURA), cheie)
    return _FONTURI[cheie]


_FONTURI: dict[int, object] = {}
_CODEPOINTS: dict[str, str] | None = None


def codepoint(eticheta: str) -> str | None:
    """Caracterul SMuFL al etichetei, din tabelul oficial.

    Nu se scrie de mana o harta de 36 de coduri: tabelul e livrat cu fontul si
    are 2940 de nume, deci orice simbol adaugat mai tarziu merge de la sine.
    Etichetele proprii lui Smashcima (`beam`, `beamHook`, `slur`, `legerLine`)
    nu sunt in el, si se intoarce None — alea se deseneaza oricum cu linii.
    """
    global _CODEPOINTS
    if _CODEPOINTS is None:
        if not SMUFL_NUME.exists():
            _CODEPOINTS = {}
        else:
            tabel = json.loads(SMUFL_NUME.read_text(encoding="utf-8"))
            _CODEPOINTS = {
                nume: chr(int(date["codepoint"][2:], 16))
                for nume, date in tabel.items()
                if "codepoint" in date
            }
    return _CODEPOINTS.get(eticheta)


def glifa(draw, eticheta: str, x: float, y: float, spatiu: float,
          culoare: int, ancora: str = "mijloc") -> bool:
    """Deseneaza glifa SMuFL. Intoarce False daca fontul nu e descarcat.

    `spatiu` e inaltimea unui spatiu de portativ, in pixeli — unitatea in care
    SMuFL defineste toate marimile, deci singura care face desenele
    proportionale intre ele.

    `ancora` spune ce colt al **cutiei de cerneala** se aseaza in (x, y). Nu se
    poate folosi ancora lui PIL: aceea masoara cutia de text, iar la un font
    muzical linia de baza si ascendentul sunt departe de glifa, deci un steag
    ancorat „sus-stanga" ar ajunge la doua spatii de unde trebuie. Steagul se
    prinde de varful coditei, iar capul de nota se centreaza — sunt reguli
    diferite, si de aia ancora e un parametru.
    """
    caracter = codepoint(eticheta)
    font = _bravura(spatiu * EM_PE_SPATIU) if caracter else None
    if font is None:
        return False
    if ancora == "mijloc":
        draw.text((x, y), caracter, font=font, fill=culoare, anchor="mm")
        return True
    st, sus, dr, jos = draw.textbbox((0, 0), caracter, font=font)
    dx = x - st if ancora.startswith("stanga") else x - dr
    dy = y - sus if ancora.endswith("sus") else y - jos
    draw.text((dx, dy), caracter, font=font, fill=culoare)
    return True

# Desenele de mai jos sunt potrivite la mana pentru o casuta de latura asta — o
# bara de grup cu doua codite si doua capete umple aproape toata inaltimea, deci
# nu iese din niste proportii, ci din incercari — si apoi scalate cu `s` pentru
# orice alta latura. Constanta sta aici, nu ca implicit in semnatura lui `build`,
# fiindca e reperul fata de care s-au ales toate numerele din desene.
CELL_DEFAULT = 250
GAP_DEFAULT = 20


def _elipsa(draw, cx, cy, rx, ry, plin: bool, culoare: int, s: float = 1.0):
    """Cap de nota: elipsa, plina sau conturata."""
    cutie = [cx - rx * s, cy - ry * s, cx + rx * s, cy + ry * s]
    if plin:
        draw.ellipse(cutie, fill=culoare)
    else:
        draw.ellipse(cutie, outline=culoare, width=max(2, round(9 * s)))


def _codita_punctata(draw, x, y1, y2, s: float = 1.0):
    """Codita care **nu** se deseneaza — punctata, ca sa se vada ca lipseste."""
    pas, lung = 22 * s, 11 * s
    y, capat = min(y1, y2), max(y1, y2)
    while y < capat:
        draw.line([x, y, x, min(y + lung, capat)], fill=PUNCTAT,
                  width=max(2, round(7 * s)))
        y += pas


def _curba(p0, p1, p2, pasi: int = 18):
    """Bezier patratic esantionat in puncte.

    O polilinie de patru puncte iesea coltoasa, si la saisprezecime cele doua
    steaguri se atingeau la capat. Curba le tine paralele si arata a trasatura,
    nu a zigzag.
    """
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    puncte = []
    for i in range(pasi + 1):
        t = i / pasi
        u = 1 - t
        puncte.append((
            u * u * x0 + 2 * u * t * x1 + t * t * x2,
            u * u * y0 + 2 * u * t * y1 + t * t * y2,
        ))
    return puncte


def _steag(draw, x, y, jos: bool, dublu: bool, culoare: int, s: float = 1.0):
    """Steagul singur: una sau doua curbe pornind din varful coditei.

    `directie` e sensul in care **atarna** steagul, si e mereu inspre capul de
    nota: la codita in sus se agata de varful de sus si cade in jos, la codita in
    jos se agata de varful de jos si urca. Invers ar arunca desenul afara din
    casuta, peste legenda de dedesubt.
    """
    directie = -1 if jos else 1
    for i in range(2 if dublu else 1):
        pornire = y + directie * i * 54 * s
        draw.line(
            _curba(
                (x, pornire),
                (x + 96 * s, pornire + directie * 38 * s),
                (x + 56 * s, pornire + directie * 126 * s),
            ),
            fill=culoare, width=max(2, round(10 * s)), joint="curve",
        )


def _latime_glifei(draw, eticheta: str, spatiu: float) -> float:
    """Latimea cutiei de cerneala a glifei, sau 0 daca fontul lipseste."""
    caracter = codepoint(eticheta)
    font = _bravura(spatiu * EM_PE_SPATIU) if caracter else None
    if font is None:
        return 0.0
    st, _, dr, _ = draw.textbbox((0, 0), caracter, font=font)
    return float(dr - st)


def deseneaza_exemplu(draw, eticheta: str, box) -> None:
    """Piesa ceruta, plin; contextul care nu se scrie, punctat.

    Toate marimile sunt scrise pentru o casuta de `CELL_DEFAULT` si inmultite cu
    `s`, ca desenul sa incapa si intr-o casuta mai mica — foaia de supliment,
    care are putine simboluri, le vrea mai mici ca sa intre pe o singura pagina.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = (x2 - x1) / CELL_DEFAULT

    if eticheta in ("noteheadHalf", "noteheadBlack"):
        # Codita se lipeste de marginea din dreapta a capului desenat, nu la o
        # distanta fixa: capul din font n-are aceeasi latime ca elipsa veche, iar
        # o codita care nu atinge capul arata a doua lucruri, nu a unul.
        latime = _latime_glifei(draw, eticheta, 76 * s)
        if not glifa(draw, eticheta, cx, cy, 76 * s, GRI):
            _elipsa(draw, cx, cy, 52, 38, eticheta == "noteheadBlack", GRI, s)
            latime = 104 * s
        _codita_punctata(draw, cx + latime / 2 - 4 * s, y1 + 40 * s, cy, s)
    elif eticheta.startswith("flag"):
        jos = eticheta.endswith("Down")
        dublu = "16th" in eticheta
        varf = y1 + 45 * s if not jos else y2 - 45 * s
        codita = cx - 30 * s
        capat = cy + (70 * s if not jos else -70 * s)
        _codita_punctata(draw, codita, varf, capat, s)
        # Codita pleaca din marginea capului, nu din mijlocul lui: la stanga cand
        # coboara, la dreapta cand urca. Asa arata si in partitura de copiat.
        if not glifa(draw, "noteheadBlack",
                     codita + (-36 * s if not jos else 36 * s), capat, 60 * s, PUNCTAT):
            _elipsa(draw, codita + (-36 * s if not jos else 36 * s), capat, 40, 30,
                    True, PUNCTAT, s)
        # Steagul se **prinde de varful coditei**: marginea lui din stanga pe
        # codita, iar marginea de sus (la codita in sus) sau de jos (la codita in
        # jos) chiar in varf. Centrat pe varf, cum era inainte, la „in jos"
        # cadea peste capul de nota — adica desena exact ce nu trebuie copiat.
        if not glifa(draw, eticheta, codita, varf, 44 * s, GRI,
                     "stanga-sus" if not jos else "stanga-jos"):
            _steag(draw, codita, varf, jos, dublu, GRI, s)
    elif eticheta == "beam":
        # Asezat de la **marginea de sus** in jos, nu din centru: bara plus doua
        # codite plus doua capete masoara aproape toata inaltimea casutei, si din
        # centru ultimul cap iesea sub chenar, peste legenda.
        bara = y1 + 70 * s
        draw.polygon(
            [(x1 + 40 * s, bara + 30 * s), (x2 - 40 * s, bara - 30 * s),
             (x2 - 40 * s, bara + 12 * s), (x1 + 40 * s, bara + 72 * s)],
            fill=GRI,
        )
        for x, sus, cap in ((x1 + 46 * s, bara + 50 * s, y1 + 210 * s),
                            (x2 - 46 * s, bara - 10 * s, y1 + 180 * s)):
            _codita_punctata(draw, x, sus, cap - 10 * s, s)
            if not glifa(draw, "noteheadBlack", x, cap, 54 * s, PUNCTAT):
                _elipsa(draw, x, cap, 36, 27, True, PUNCTAT, s)
    elif eticheta == "beamHook":
        # Acelasi desen ca `beam`, dar **taiat**: o bucata scurta pornind dintr-o
        # singura codita. Asta e toata diferenta dintre cele doua etichete.
        draw.polygon(
            [(cx - 6 * s, y1 + 80 * s), (cx + 74 * s, y1 + 52 * s),
             (cx + 74 * s, y1 + 94 * s), (cx - 6 * s, y1 + 122 * s)],
            fill=GRI,
        )
        _codita_punctata(draw, cx - 2 * s, y1 + 70 * s, y1 + 190 * s, s)
        if not glifa(draw, "noteheadBlack", cx - 2 * s, y1 + 202 * s, 54 * s, PUNCTAT):
            _elipsa(draw, cx - 2 * s, y1 + 202 * s, 36, 27, True, PUNCTAT, s)


EXEMPLE = {
    "noteheadHalf", "noteheadBlack", "flag8thUp", "flag8thDown",
    "flag16thUp", "flag16thDown", "beam", "beamHook",
}


def load_font(size: int):
    """Un font de sistem, cu revenire la cel implicit daca nu se gaseste."""
    from PIL import ImageFont

    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build(cell: int = CELL_DEFAULT, gap: int = GAP_DEFAULT,
          exemple: frozenset[str] | set[str] = EXEMPLE,
          simboluri: list[tuple[str, str]] | None = None) -> list[tuple[object, dict]]:
    """Deseneaza foile si intoarce [(imagine, harta casutelor)].

    Sunt mai multe pagini (trei, la casuta implicita) fiindca cele 36 de
    simboluri x 3 exemplare nu incap pe una singura la o marime la care omul
    sa poata scrie natural. Casutele mici ar produce un scris inghesuit, adica
    exact ce nu vrem — glifele trebuie sa semene cu ce scrie omul pe o
    partitura, nu cu ce scrie cand n-are loc. De aceea paginarea se calculeaza
    aici din inaltimea ramasa, in loc sa fie fixata: daca lista mai creste,
    apare o pagina in plus, nu o casuta mai mica.
    """
    from PIL import Image, ImageDraw

    title_font = load_font(58)
    text_font = load_font(34)
    label_font = load_font(28)
    small_font = load_font(22)

    marker = 60
    corners = [
        (MARGIN, MARGIN),
        (WIDTH - MARGIN - marker, MARGIN),
        (MARGIN, HEIGHT - MARGIN - marker),
        (WIDTH - MARGIN - marker, HEIGHT - MARGIN - marker),
    ]
    group_width = REPETITIONS * (cell + gap)
    per_row = max(1, (WIDTH - 2 * MARGIN) // group_width)
    # Patru casute (exemplu + trei de scris) trebuie sa incapa pe latime.
    assert MARGIN + (REPETITIONS + 1) * (cell + gap) <= WIDTH - MARGIN, (
        "casuta e prea mare ca sa incapa exemplul plus cele trei de scris"
    )
    block = 38 + cell + 30  # eticheta + casuta + spatiu

    pages: list[tuple[object, dict]] = []
    remaining = list(SYMBOLS if simboluri is None else simboluri)
    page_number = 1

    while remaining:
        image = Image.new("L", (WIDTH, HEIGHT), 255)
        draw = ImageDraw.Draw(image)
        for x, y in corners:
            draw.rectangle([x, y, x + marker, y + marker], fill=0)

        y = MARGIN + marker + 36
        draw.text((MARGIN, y), f"Foaie de glife muzicale — pagina {page_number}",
                  font=title_font, fill=0)
        y += 74
        if page_number == 1:
            draw.text(
                (MARGIN, y),
                "Numele tau (sau orice nume scurt): ______________________________",
                font=text_font, fill=0,
            )
            y += 54
            for line in INSTRUCTIONS:
                draw.text((MARGIN, y), "- " + line, font=small_font, fill=0)
                y += 32
            y += 20
        else:
            draw.text((MARGIN, y), "Nume: ____________________", font=text_font, fill=0)
            y += 56

        # Instructiunea care conteaza cel mai mult pe foaia asta, si de asta e
        # separata de restul si scrisa mai mare: prima foaie completata s-a
        # intors cu note intregi in loc de bucati de nota, fiindca „cap de
        # doime" se citeste firesc drept „nota doime" (§15.22).
        if any(label in exemple for label, _ in remaining):
            draw.text(
                (MARGIN, y),
                "Unde vezi un desen gri: copiaza DOAR partea desenata plin.",
                font=text_font, fill=0,
            )
            y += 44
            draw.text(
                (MARGIN, y),
                "Ce e punctat (codita, capul de nota) e doar ca sa intelegi unde"
                " sta bucata — nu il scrie.",
                font=small_font, fill=0,
            )
            y += 46

        boxes = []
        while remaining and y + block <= HEIGHT - MARGIN - marker - 20:
            # Un simbol cu exemplu ia **randul intreg**: o casuta tiparita plus
            # cele trei de scris nu incap doua pe rand.
            for column in range(per_row):
                if not remaining:
                    break
                # Un simbol cu exemplu **incepe intotdeauna randul**: are patru
                # casute in loc de trei, si de la a doua coloana in colo ultima
                # ar iesi din pagina. Daca randul e deja inceput, se lasa pe
                # urmatorul. Asa, latimea de verificat e una singura.
                if remaining[0][0] in exemple and column > 0:
                    break
                label, human = remaining.pop(0)
                x0 = MARGIN + column * group_width
                draw.text((x0, y), human, font=label_font, fill=0)
                draw.text((x0, y + 30), label, font=small_font, fill=120)
                top = y + 60

                pornire = 0
                if label in exemple:
                    # Casuta de exemplu: desenata, **nu** trecuta in harta, deci
                    # segmentarea n-o citeste niciodata ca pe scrisul cuiva.
                    draw.rectangle([x0, top, x0 + cell, top + cell],
                                   outline=PUNCTAT, width=3)
                    deseneaza_exemplu(draw, label, (x0, top, x0 + cell, top + cell))
                    draw.text((x0 + 6, top + cell + 4), "exemplu — nu scrie aici",
                              font=small_font, fill=120)
                    pornire = 1

                for index in range(REPETITIONS):
                    bx = x0 + (index + pornire) * (cell + gap)
                    draw.rectangle([bx, top, bx + cell, top + cell], outline=0, width=3)
                    boxes.append(
                        {"label": label, "index": index,
                         "box": [bx, top, bx + cell, top + cell]}
                    )

                if pornire:
                    break  # randul e al lui, nu mai intra nimic langa el
            y += block + 30

        pages.append((image, {
            "page": page_number,
            "dpi": DPI,
            "size": [WIDTH, HEIGHT],
            "markers": [[x, y0, x + marker, y0 + marker] for x, y0 in corners],
            "cells": boxes,
        }))
        page_number += 1

    return pages


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="unde se scriu foile; numarul paginii se adauga la nume",
    )
    parser.add_argument(
        "--casuta", type=int, default=250, metavar="PX",
        help="latura casutei in pixeli la 300 dpi; mai mica inseamna mai multe "
             "simboluri pe pagina, dar si scris mai inghesuit",
    )
    parser.add_argument(
        "--doar", default=None, metavar="ETICHETE",
        help="doar simbolurile astea, separate prin virgula — o foaie de "
             "supliment pentru cine a sarit casute sau a inteles gresit una "
             "(raportul de acoperire din `glife_repository.py` spune care)",
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="scrie si un PDF de o singura bucata, la 300 dpi — ce dai la tiparit. "
             "Se tipareste la marime reala, fara `fit to page`: raportul dintre "
             "scris si casuta e o informatie folosita mai tarziu",
    )
    parser.add_argument(
        "--fara-exemple", action="store_true",
        help="foaia asa cum era inainte de desenele de referinta. Exista ca sa "
             "se poata **reface harta** unei foi deja completate: asezarea se "
             "muta cand un simbol capata casuta de exemplu, iar o foaie tiparita "
             "cu harta veche nu se mai poate decupa cu cea noua",
    )
    args = parser.parse_args()

    simboluri = SYMBOLS
    if args.doar:
        cerute = [e.strip() for e in args.doar.split(",") if e.strip()]
        cunoscute = {label for label, _ in SYMBOLS}
        necunoscute = [e for e in cerute if e not in cunoscute]
        if necunoscute:
            raise SystemExit(f"etichete necunoscute: {', '.join(necunoscute)}")
        simboluri = [(label, om) for label, om in SYMBOLS if label in set(cerute)]

    pages = build(
        cell=args.casuta,
        exemple=frozenset() if args.fara_exemple else EXEMPLE,
        simboluri=simboluri,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    plan = {"symbols": [label for label, _ in simboluri], "pages": []}
    for image, page in pages:
        path = Path(f"{stem}-{page['page']}.png")
        image.save(path, dpi=(DPI, DPI))
        page["image"] = path.name
        plan["pages"].append(page)
        print(f"  {path} — {len(page['cells'])} casute")

    if args.pdf:
        cale_pdf = Path(f"{stem}.pdf")
        imagini = [image.convert("L") for image, _ in pages]
        imagini[0].save(
            cale_pdf, save_all=True, append_images=imagini[1:], resolution=DPI
        )
        print(f"  {cale_pdf} — {len(imagini)} pagini, de tiparit la marime reala")

    plan_path = Path(f"{stem}.json")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    scrise = {cell["label"] for page in plan["pages"] for cell in page["cells"]}
    lipsa = [label for label, _ in simboluri if label not in scrise]
    print(f"{len(pages)} pagini, {len(scrise)} simboluri | harta: {plan_path}")
    if lipsa:
        raise SystemExit(f"n-au incaput: {', '.join(lipsa)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
