"""Portative gasite pe pagina, si gruparea lor in sisteme.

Modulul nu importa nici torch, nici ultralytics: primeste detectorul deja
construit si lucreaza cu dreptunghiuri. Asa, gruparea in sisteme — partea in
care se poate gresi — se testeaza in milisecunde, fara greutati descarcate.

De ce exista gruparea: `jazzmus-model` citeste **un portativ**, iar
`smt-grandstaff` citeste **un sistem intreg** (doua portative intr-o singura
imagine, cum e antrenat pe grand staff de pian). Aceeasi pagina trebuie deci
taiata in doua feluri.
"""

from __future__ import annotations

from argparse import ArgumentTypeError
from collections.abc import Sequence
from pathlib import PurePath

# Sub atat, dreptunghiul e o eroare de detectie, nu un portativ. O cutie cu
# coordonate trunchiate la acelasi pixel ar da un decupaj de inaltime zero, iar
# redimensionarea de dupa ar imparti la zero.
MIN_STAFF_SIZE = 4

# Distanta verticala, in pixeli, sub care doua portative apartin aceluiasi
# sistem. Intre sisteme spatiul e mult mai mare decat intre portativele
# aceluiasi sistem, deci pragul nu e delicat.
SYSTEM_GAP = 12

Box = tuple[int, int, int, int]  # x1, y1, x2, y2


def detect_boxes(yolo, source) -> list[Box]:
    """Portativele de pe o pagina, de sus in jos.

    Ordinea verticala conteaza: intr-un sistem coral in partitura stransa,
    portativul de sus poarta S+A si cel de jos T+B.

    `source` poate fi o cale sau o imagine deja incarcata. A doua varianta e
    necesara de cand pagina se indreapta inainte de detectie (etapa 6): cutiile
    trebuie sa fie in acelasi sistem de coordonate cu imaginea din care se taie
    decupajele, altfel decupam de langa muzica.
    """
    target = str(source) if isinstance(source, (str, PurePath)) else source
    boxes: list[Box] = []
    for result in yolo(target, verbose=False):
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if result.names[int(cls)].lower() != "staff":
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            if y2 - y1 < MIN_STAFF_SIZE or x2 - x1 < MIN_STAFF_SIZE:
                continue
            boxes.append((x1, y1, x2, y2))
    boxes.sort(key=lambda b: b[1])
    return boxes


def group_by_gap(boxes: list[Box], gap: int = SYSTEM_GAP) -> list[list[Box]]:
    """Grupeaza portativele in sisteme, dupa distanta verticala.

    Mai bun decat „cate doua": pe pagini reale un sistem poate avea un singur
    portativ, cand o voce nu canta inca. Exact cazul primului sistem din
    „Iata-L pe Hristos", unde T&B intra abia la masura 6.
    """
    systems: list[list[Box]] = []
    for box in boxes:
        if systems and box[1] - systems[-1][-1][3] <= gap:
            systems[-1].append(box)
        else:
            systems.append([box])
    return systems


def group_fixed(boxes: list[Box], per_system: int | Sequence[int]) -> list[list[Box]]:
    """Grupeaza portativele cate `per_system`, de sus in jos.

    Exista pentru ca pe partiturile scrise de mana cu versuri **spatierea nu
    poarta nicio informatie despre sisteme**. Masurat pe manuscrisul din §14:
    fiecare portativ are un rand de versuri dedesubt, si intre cele doua
    portative ale unui sistem, si intre doua sisteme. Distantele
    centru-la-centru ies uniforme (230-250 px pe pagina 1), iar spatiile dintre
    cutiile detectorului — umflate de text — ajung de 2 px si in interiorul
    sistemului, si intre sisteme. Niciun prag nu le poate separa, fiindca nu e
    o problema de calibrare: semnalul lipseste.

    Cand geometria nu poate raspunde, raspunsul se declara. E aceeasi hotarare
    ca la eticheta tiparit/scris de mana din §12.4: o declaratie gresita se
    vede si se corecteaza, pe cand un clasificator care ghiceste trimite
    jumatati de sisteme diferite in acelasi decupaj, tacut.

    Un rest care nu umple un sistem intreg **nu se arunca si nu se lipeste** de
    ultimul: devine sistemul lui, mai scurt. Asa neconcordanta ajunge pana la
    `parse_open_system`, care o raporteaza, in loc sa dispara aici.

    `per_system` poate fi si o **lista**, cate un numar pentru fiecare sistem:
    `[2, 2, 2, 3, 3]`. Un singur numar ramane scurtatura pentru „la fel peste
    tot". Lista exista fiindca numarul de portative pe sistem chiar variaza in
    manuscrise: in „Doamne bun asculta" vocea solo primeste un portativ propriu
    abia din sistemul al patrulea, deci pagina are 2, 2, 2, 3, 3. Declarata ca
    `2`, aceeasi pagina s-ar taia in sase sisteme de cate doua — jumatati de
    sisteme diferite in acelasi decupaj, exact ce evita §14.6.

    Ce ramane dupa ce lista se termina urmeaza aceeasi regula ca restul: devine
    un sistem in plus, nu se arunca si nu se lipeste de ultimul. Asa o
    declaratie prea scurta iese ca numar de sisteme care nu se potriveste cu
    referinta, si `dataset.py` refuza, in loc sa imperecheze tacut alte masuri
    cu alta muzica.
    """
    if isinstance(per_system, int):
        if per_system < 1:
            raise ValueError(f"un sistem are cel putin un portativ, nu {per_system}")
        return [boxes[i:i + per_system] for i in range(0, len(boxes), per_system)]

    counts = list(per_system)
    if not counts:
        raise ValueError("declaratia pe sisteme nu poate fi goala")
    if any(count < 1 for count in counts):
        raise ValueError(f"un sistem are cel putin un portativ, nu {min(counts)}")

    systems: list[list[Box]] = []
    start = 0
    for count in counts:
        if start >= len(boxes):
            break
        systems.append(boxes[start:start + count])
        start += count
    if start < len(boxes):
        systems.append(boxes[start:])
    return systems


class BadDeclaration(ValueError, ArgumentTypeError):
    """Declaratie despre structura paginii care nu se poate citi.

    Mosteneste din amandoua **ca sa se vada motivul**. `argparse` isi ia textul
    din `ArgumentTypeError`, dar pe orice `ValueError` scrie doar „invalid
    parse_grouping value", adica exact explicatia se pierde — iar aici greseala
    e usor de facut si greu de ghicit. Ramane si `ValueError`, pentru cine
    apeleaza functia din cod, nu de la linia de comanda.
    """


def parse_grouping(text: str) -> int | list[int]:
    """Valoarea lui `--portative-pe-sistem`: `2` sau `2,2,2,3,3`.

    Sta aici, nu in fiecare script, ca toate cele cinci care au optiunea sa
    citeasca aceeasi forma.
    """
    pieces = [piece.strip() for piece in str(text).split(",") if piece.strip()]
    if not pieces:
        raise BadDeclaration("asteptam un numar sau o lista, de pilda `2` sau `2,2,3`")
    try:
        counts = [int(piece) for piece in pieces]
    except ValueError:
        raise BadDeclaration(
            f"{text!r} nu e un numar si nici o lista de numere despartite prin "
            "virgula, de pilda `2` sau `2,2,2,3,3`"
        ) from None
    if any(count < 1 for count in counts):
        raise BadDeclaration(f"un sistem are cel putin un portativ, iar {text!r} cere mai putin")
    return counts[0] if len(counts) == 1 else counts


def parse_layout(text: str) -> list[list[list[str]]]:
    """Valoarea lui `--asezare`: ce voci stau pe fiecare portativ, de sus in jos.

    Portativele se despart cu `/`, vocile cu `,`. Partitura stransa corala e
    `S,A/T,B`; o cantare cu solo pe portativ separat, deasupra, e `X1/S,A/T,B`.

    Cu `;` se declara **cate o asezare pe sistem**, fiindca asezarea variaza in
    interiorul aceleiasi pagini: in „Doamne bun asculta", vocea de solo intra
    abia din sistemul al patrulea, deci pagina e
    `S,A/T,B;S,A/T,B;S,A/T,B;X1/S,A/T,B;X1/S,A/T,B`. Fara `;`, aceeasi asezare
    se foloseste pentru toate sistemele.

    Intoarce intotdeauna o **lista de asezari**, ca apelantul sa nu deosebeasca
    doua forme; una singura inseamna „la fel peste tot".

    Sta langa `parse_grouping` fiindca amandoua declara acelasi fel de lucru:
    structura paginii, acolo unde geometria si cheia nu pot raspunde (§14.5,
    §15.10). Numele vocilor sunt siruri, deci modulul ramane fara stiinta de
    modelul muzical.
    """
    systems = [part.strip() for part in str(text).split(";") if part.strip()]
    if not systems:
        raise BadDeclaration(
            "asteptam portative despartite prin `/`, de pilda `S,A/T,B`"
        )

    out: list[list[list[str]]] = []
    for system in systems:
        staves = [group.strip() for group in system.split("/") if group.strip()]
        if not staves:
            raise BadDeclaration(
                "asteptam portative despartite prin `/`, de pilda `S,A/T,B`"
            )
        layout = []
        for group in staves:
            voices = [voice.strip() for voice in group.split(",") if voice.strip()]
            if not voices:
                raise BadDeclaration(f"un portativ fara nicio voce, in {system!r}")
            layout.append(voices)
        flat = [voice for group in layout for voice in group]
        if len(set(flat)) != len(flat):
            raise BadDeclaration(
                f"acelasi nume de voce apare de doua ori in {system!r} — id-urile "
                "trebuie sa fie unice pe tot sistemul"
            )
        out.append(layout)
    return out


def group(
    boxes: list[Box],
    per_system: int | Sequence[int] | None = None,
    gap: int = SYSTEM_GAP,
) -> list[list[Box]]:
    """Gruparea in sisteme: declarata daca se stie, dedusa altfel.

    `per_system=None` pastreaza comportamentul de pana acum — potrivit pentru
    partituri tiparite, unde spatiul dintre sisteme chiar e mai mare decat cel
    dintre portativele aceluiasi sistem. Declaratia poate fi un numar sau o
    lista, cate un numar pe sistem (vezi `group_fixed`).
    """
    if per_system is None:
        return group_by_gap(boxes, gap)
    return group_fixed(boxes, per_system)


def bounding_box(boxes: list[Box], padding: int, size: tuple[int, int]) -> Box:
    """Dreptunghiul care cuprinde toate cutiile, marginit de pagina.

    `size` e (latime, inaltime) a imaginii, ca marginea sa nu iasa din ea.
    """
    width, height = size
    return (
        max(min(b[0] for b in boxes) - padding, 0),
        max(min(b[1] for b in boxes) - padding, 0),
        min(max(b[2] for b in boxes) + padding, width),
        min(max(b[3] for b in boxes) + padding, height),
    )
