"""Puntea dintre ce scoate `smt-grandstaff` si ce citeste `kern.py`.

Modelul nu scoate un document Humdrum, ci un **sir plat de tokeni**, in care
structura de tabel e codificata in doi tokeni speciali:

    <t>   separator de coloana (tab)
    <b>   sfarsit de linie

iar exponentul coloanei e `**ekern_1.0`, nu `**kern`. Diferenta pare cosmetica,
dar `split_voices()` cauta exact `**kern`: pe un document netradus gaseste zero
voci si pierde toata partitura, in tacere.

Despre `@`
----------
eKern imparte un token in subtokeni despartiti de `@`: `2.@ee@[` in loc de
`2.ee[`. Reconversia e stergerea separatorilor, nu pastrarea primului camp -
altfel `4@g@#` ar deveni durata `4` fara inaltime. Greutatile `smt-grandstaff`
scot deja tokeni fara `@` (verificat pe iesirea reala: zero aparitii), deci
codul de aici e aparare impotriva altui set de greutati, nu functionalitate
folosita azi.

Modulul e cod pur: fara torch, fara imagini, testabil in milisecunde.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

COLUMN = "<t>"
LINE = "<b>"
# `<s>` e un spatiu din documentul original. In Humdrum, un spatiu **in
# interiorul unei coloane** desparte notele unui acord: `4g 4b` e o coloana, doua
# note. Deci se traduce inapoi in spatiu, si problema trece mai departe la
# `kern.py`, care stie ca acordurile sunt in afara subsetului v1.
SPACE = "<s>"

# `**ekern_1.0`, `**bekern`, `**ekern` - toate sunt aceeasi coloana de note
# pentru noi. Versiunea din nume nu schimba ce contine.
_SPINE = re.compile(r"^\*\*b?ekern(_[\d.]+)?$")

# Orice alt marcaj `<...>` e ceva ce nu stim sa traducem. `<eos>` si `<bos>`
# sunt scoase de decodor, dar un set de greutati diferit poate emite altele.
_MARKER = re.compile(r"^<[^>]*>$")


@dataclass
class Conversion:
    """Documentul tradus, plus ce n-am stiut sa traducem.

    Tokenii necunoscuti nu se arunca in tacere: ajung in `problems`-ul paginii,
    ca sa se vada in raport ca traducerea a fost incompleta.
    """

    kern: str
    unknown: list[str] = field(default_factory=list)


def to_kern(tokens: list[str] | str) -> Conversion:
    """Sir de tokeni bekern -> document Humdrum cu tab-uri si linii.

    Accepta si textul salvat de scripturi (un token pe linie), fiindca asa
    ajunge iesirea pe disc si vrem sa o putem reciti fara pasi in plus.
    """
    if isinstance(tokens, str):
        tokens = tokens.split()

    unknown: list[str] = []
    lines: list[str] = []
    fields: list[str] = [""]

    for raw in tokens:
        token = raw.strip()
        if not token:
            continue

        if token == COLUMN:
            fields.append("")
            continue
        if token == LINE:
            lines.append("\t".join(fields))
            fields = [""]
            continue
        if token == SPACE:
            fields[-1] += " "
            continue
        if _MARKER.match(token):
            unknown.append(token)
            continue

        if _SPINE.match(token):
            token = "**kern"
        elif "@" in token:
            token = token.replace("@", "")

        # Doi tokeni fara `<t>` intre ei apartin aceleiasi coloane. Nu s-a
        # intamplat pe iesirea masurata, dar lipirea lor e mai putin
        # distructiva decat pierderea unuia dintre ei.
        fields[-1] += token

    # Iesirea taiata la `--max-tokens` se termina la mijlocul unei linii.
    # Pastram si linia aceea: e ultima muzica citita, si e exact ce vrem sa
    # vedem cand ne intrebam unde s-a oprit modelul.
    if any(fields):
        lines.append("\t".join(fields))

    return Conversion(kern="\n".join(lines), unknown=unknown)


# --------------------------------------------------------------------------
# Structura documentului, urmarita token cu token
# --------------------------------------------------------------------------

SPLIT = "*^"
MERGE = "*v"


@dataclass
class Grammar:
    """Cate coloane are documentul acum, si al catelea camp se scrie.

    Exista ca sa poata fi **interzis** un token inainte de a fi ales. Masurat pe
    pagina 3 (§15): din trei sisteme, toate trei au iesit cu randuri care au alt
    numar de campuri decat structura declarata — 47, 3 si 9. `split_voices` din
    `kern.py` imparte apoi campurile cu `zip`, care **taie tacut** ce nu se
    potriveste, si de acolo vine cea mai mare parte a notelor „lipsa".

    Nimic din asta nu e o problema de citire: modelul stie ce vede, dar scrie un
    tabel care nu se inchide. Iar un tabel care nu se inchide se poate face
    imposibil, nu doar semnalat, fiindca numarul de coloane e cunoscut in
    fiecare moment — antetul il declara, `*^` si `*v` il schimba.

    Ce **nu** face: nu imbunatateste nicio nota. Daca modelul ar fi citit
    gresit, va citi la fel de gresit, doar intr-un document lizibil. Cifra
    spune care din doua.
    """

    # Cate coloane declara antetul. Cunoscut din afara — un sistem de partitura
    # stransa are un portativ per coloana — sau `None`, si atunci prima linie
    # hotaraste singura, ca pana acum.
    spines: int | None = None
    # Plafon de coloane, adica de voci. Se **declara**, ca `--portative-pe-sistem`
    # (§14.6): pe „Iata-L pe Hristos" numarul de voci chiar variaza, deci nu se
    # presupune. Cand e dat, `*^` peste plafon devine imposibil.
    max_columns: int | None = None
    columns: int | None = None
    _row: list[str] = field(default_factory=lambda: [""])

    @property
    def field_index(self) -> int:
        """Al catelea camp al randului curent se scrie, de la zero."""
        return len(self._row) - 1

    def accept(self, word: str) -> None:
        """Inregistreaza un token deja ales."""
        token = word.strip()
        if token == COLUMN:
            self._row.append("")
        elif token == LINE:
            self.columns = columns_after(self._row, self.columns)
            self._row = [""]
        elif token and not _MARKER.match(token):
            self._row[-1] += token

    def forbidden(self) -> tuple[str, ...]:
        """Tokenii care ar strica structura daca ar fi alesi acum."""
        banned: list[str] = []
        expected = self.columns if self.columns is not None else self.spines
        if expected is not None:
            # Ultimul camp al randului nu mai poate fi urmat de un tab, iar un
            # rand neterminat nu se poate inchide.
            banned.append(COLUMN if self.field_index >= expected - 1 else LINE)
        if self.max_columns is not None and self.columns is not None:
            pending = sum(1 for value in self._row if value.strip() == SPLIT)
            if self.columns + pending >= self.max_columns:
                banned.append(SPLIT)
        return tuple(banned)


def columns_after(row: list[str], columns: int | None) -> int:
    """Cate coloane are documentul dupa randul asta.

    Aceeasi regula ca `_restructure` din `kern.py`, si tocmai de aceea scrisa
    separat: acolo se muta vocile, aici se numara doar coloanele, iar numaratul
    trebuie sa se poata face **inainte** ca randul sa existe ca text.

    Un camp obisnuit da o coloana; `*^` da doua; un sir de `*v` alaturate da una
    singura, fiindca ele lipesc la loc coloanele pe care `*^` le desfacuse.
    """
    if columns is None:
        return len(row)

    count = 0
    index = 0
    while index < len(row):
        if row[index].strip() == SPLIT:
            count += 2
            index += 1
        elif row[index].strip() == MERGE:
            while index < len(row) and row[index].strip() == MERGE:
                index += 1
            count += 1
        else:
            count += 1
            index += 1
    return count


def from_kern(text: str) -> list[str]:
    """Document Humdrum -> sir de tokeni bekern. Inversul lui `to_kern`.

    Exista pentru antrenare: tinta pe care o invata modelul e un sir de tokeni,
    nu un tabel. Traducerea e mecanica — tab devine `<t>`, sfarsit de linie
    `<b>`, spatiu `<s>` — cu o singura hotarare de continut: exponentul coloanei
    se scrie `**ekern_1.0`, cum il scrie modelul, nu `**kern`. `to_kern` face
    exact drumul invers, deci perechea lor se poate verifica pe orice document.
    """
    tokens: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        for index, field in enumerate(line.split("\t")):
            if index:
                tokens.append(COLUMN)
            if field == "**kern":
                tokens.append("**ekern_1.0")
                continue
            parts = field.split(" ")
            for position, piece in enumerate(parts):
                if position:
                    tokens.append(SPACE)
                if piece:
                    tokens.append(piece)
        tokens.append(LINE)
    return tokens
