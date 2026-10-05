"""Model intern -> document Humdrum `**kern`. Inversul lui `kern.py`.

De ce exista: ca sa antrenam modelul pe paginile scrise de tine, **tinta trebuie
scrisa in limba modelului**. Lantul exista azi intr-un singur sens — tokeni ->
`bekern.to_kern` -> text Humdrum -> `kern.parse_open_system` -> `Score` — iar
fara inversul lui, o referinta din MuseScore ramane doar material de masurat,
niciodata material de antrenare (PROJECT.md §15.7).

Partea grea nu e vocabularul, ci ca **randurile Humdrum sunt felii de timp**.
Doua voci cu ritmuri diferite dau randuri in care una are token si cealalta are
`.`, tokenul null. Deci nu se scrie voce dupa voce, ci se construieste o grila de
timp peste toate vocile sistemului, si abia apoi se scriu randurile.

Doua conventii pe care le respecta, amandoua citite din ce produce modelul real
si verificate de `kern.py` la citire:

* **coloana din stanga e portativul de jos.** Humdrum pune vocea cea mai joasa
  intai; iesirea lui `smt-grandstaff` incepe cu `*clefF4` si abia apoi `*clefG2`.
* **intr-un portativ despartit, ramura din stanga e vocea de sus** — S la
  portativul de sus, T la cel de jos.

Un portativ ale carui doua voci canta acelasi lucru intr-o masura se scrie pe **o
singura coloana**, cu `*^` si `*v` la trecerile dintre stari. Asta nu e o
economie de spatiu: e felul in care arata partiturile pe care a fost antrenat
modelul, si felul in care `split_voices` citeste unisonul.
"""

from __future__ import annotations

from fractions import Fraction

from app.music_model.model import (
    Clef,
    Duration,
    Event,
    KeySignature,
    Measure,
    Note,
    Part,
    Pitch,
    Rest,
    Score,
    TimeSignature,
)

NULL = "."
SPLIT = "*^"
MERGE = "*v"
HOLD = "*"

# Reciproca duratei: o patrime se scrie `4`. Inversul lui `_DURATIONS` din
# `kern.py`, tinut aici ca sa nu depinda un modul de intimitatile celuilalt.
_RECIPROCAL = {
    "whole": 1,
    "half": 2,
    "quarter": 4,
    "eighth": 8,
    "16th": 16,
    "32nd": 32,
}

_ACCIDENTALS = {"sharp": "#", "flat": "-", "natural": "n"}


class UnwritableScore(ValueError):
    """Partitura contine ceva ce nu se poate scrie exact in `**kern`.

    Se ridica in loc sa se aproximeze, din acelasi motiv ca `UnsupportedNotation`
    la citire: o tinta de antrenare gresita e mai rea decat una lipsa, fiindca
    modelul ar invata din ea fara ca cineva sa observe.
    """


def write_duration(duration: Duration) -> str:
    """Durata, ca reciproca ei. Trioletul de optimi se scrie `12`.

    Regula generala, aceeasi pe care o citeste `kern.parse_duration`: valoarea
    scrisa e reciproca inmultita cu raportul grupului neregulat. O optime (8)
    intr-un grup 3:2 da `8 * 3 / 2 = 12`, iar durata sunata iese `4/12`, adica
    exact o treime de patrime.
    """
    base = _RECIPROCAL[duration.note_type.value]
    if duration.tuplet is not None:
        value = Fraction(base * duration.tuplet.actual, duration.tuplet.normal)
        if value.denominator != 1:
            raise UnwritableScore(
                f"grupul {duration.tuplet} peste {duration.note_type.value} nu se "
                "poate scrie ca o singura durata Humdrum"
            )
        base = int(value)
    return f"{base}{'.' * duration.dots}"


def write_pitch(pitch: Pitch) -> str:
    """`c` = Do4, `cc` = Do5, `C` = Do3, `CC` = Do2 — conventia Humdrum.

    Literele mici urca de la octava 4, cele mari coboara de la octava 3.
    """
    if pitch.octave >= 4:
        letters = pitch.step.lower() * (pitch.octave - 3)
    else:
        letters = pitch.step.upper() * (4 - pitch.octave)
    if not letters:
        raise UnwritableScore(f"octava {pitch.octave} nu se poate scrie in kern")
    mark = _ACCIDENTALS.get(pitch.accidental.value, "") if pitch.accidental else ""
    return letters + mark


def write_event(event: Event, ties: bool = True) -> str:
    """Un eveniment, ca token Humdrum. Legaturile de durata intra in token.

    `ties=False` scrie nota **fara marcajul de legato**. Nu e o simplificare de
    comoditate: vocabularul lui `smt-grandstaff` n-are niciun token de legato —
    are 2552 de tokeni cu `[`, dar toti sunt armuri (`*k[f#]`), iar `2f#` fara
    legato exista. Masurat pe 2026-09-03.

    Deci o tinta care poarta `[2f#` nu poate fi invatata, si intregul sistem
    cade la verificarea de vocabular: **6 din 10 sisteme** din „Doamne bun
    asculta", 0 din 3 din pagina 3. Renuntand la marcaj, muzica ramane aceeasi —
    doua note legate sunt tot doua note, cu aceleasi inaltimi si durate — si se
    pierde doar semnul ca sunt tinute. Alternativa era sa pierdem sistemele.
    """
    if isinstance(event, Rest):
        return write_duration(event.duration) + "r"
    token = write_duration(event.duration) + write_pitch(event.pitch)
    if not ties:
        return token
    if event.tie_start and event.tie_stop:
        return "_" + token
    if event.tie_start:
        return "[" + token
    if event.tie_stop:
        return token + "]"
    return token


def write_clef(clef: Clef) -> str:
    return f"*clef{clef.sign}{'v' if clef.octave_change else ''}{clef.line}"


def write_key(key: KeySignature) -> str:
    order = ("f", "c", "g", "d", "a", "e", "b")
    if key.fifths > 0:
        return "*k[" + "".join(f"{step}#" for step in order[: key.fifths]) + "]"
    if key.fifths < 0:
        flats = list(reversed(order))[: -key.fifths]
        return "*k[" + "".join(f"{step}-" for step in flats) + "]"
    return "*k[]"


def write_time(time: TimeSignature) -> str:
    return f"*M{time.beats}/{time.beat_type}"


def onsets(measure: Measure) -> list[tuple[Fraction, Event]]:
    """Evenimentele masurii, cu momentul in care incepe fiecare."""
    position = Fraction(0)
    result: list[tuple[Fraction, Event]] = []
    for event in measure.events:
        result.append((position, event))
        position += event.quarter_length
    return result


def _same_music(first: Measure, second: Measure, ties: bool = True) -> bool:
    """Cele doua voci canta acelasi lucru in masura asta?

    Atunci se scriu pe o singura coloana — asta *este* unisonul in Humdrum, si e
    regula pe care `split_voices` o citeste la `*v`.
    """
    return [write_event(e, ties) for e in first.events] == [
        write_event(e, ties) for e in second.events
    ]


def _staff_columns(voices: list[Part], number: int, ties: bool = True) -> int:
    """Cate coloane cere portativul la masura `number`: una la unison, doua altfel."""
    if len(voices) < 2:
        return 1
    measures = [voice.measure(number) for voice in voices]
    return 1 if _same_music(measures[0], measures[1], ties) else 2


def _fields_at(
    voices: list[Part], number: int, columns: int, moment: Fraction, ties: bool = True
) -> list[str]:
    """Ce scrie fiecare coloana a portativului la momentul dat."""
    used = voices[:1] if columns == 1 else voices[:2]
    fields = []
    for voice in used:
        starting = [event for start, event in onsets(voice.measure(number)) if start == moment]
        fields.append(write_event(starting[0], ties) if starting else NULL)
    return fields


def _moments(staves: list[list[Part]], number: int) -> list[Fraction]:
    """Momentele in care incepe ceva, undeva in sistem, in masura asta.

    Grila e **comuna intregului sistem**, nu a fiecarui portativ: un rand
    Humdrum e o felie de timp peste toate coloanele deodata.
    """
    found: set[Fraction] = set()
    for voices in staves:
        for voice in voices:
            found.update(start for start, _ in onsets(voice.measure(number)))
    return sorted(found)


def _barline_before(voice: Part, index: int) -> str:
    """Tokenul barei dinaintea masurii a `index`-a, cu repetitiile ei.

    Barele sunt ale sistemului, nu ale unei voci, deci se iau dintr-una singura.
    Humdrum pune intr-un singur token ce MusicXML tine pe doua masuri: `:|` din
    dreapta celei dinainte si `|:` din stanga celei de acum — `=:|!|:` cand sunt
    amandoua. Prima bara a documentului poate doar deschide: `=!|:` dupa cheie
    si armura, unde il deseneaza MuseScore (si `pagina3`).
    """
    deschide = voice.measures[index].repeat_start
    inainte = voice.measures[index - 1].barline if index else None
    if inainte is not None and inainte.repeat:
        return "=:|!|:" if deschide else "=:|!"
    return "=!|:" if deschide else "="


def _transition(before: int, after: int) -> list[str] | None:
    """Randul care trece un portativ de la `before` coloane la `after`.

    **Cate un portativ pe rand, niciodata doua deodata.** In Humdrum, un sir de
    `*v` alaturate formeaza *o singura* lipire: scrise pe acelasi rand, cele doua
    perechi ale unui sistem — jos si sus — ar deveni o singura coloana. Iesirea
    reala a modelului face la fel, `*v *v *` pe un rand si restul pe urmatorul.
    """
    if before == after:
        return None
    return [SPLIT] if after > before else [MERGE, MERGE]


def to_kern_text(
    score: Score,
    layout: tuple[tuple[str, ...], ...] | None = None,
    ties: bool = True,
) -> str:
    """Un `Score` -> document Humdrum, cu portativul de jos in stanga.

    `layout` spune ce voci stau pe fiecare portativ, **de jos in sus** — ordinea
    coloanelor Humdrum, care e inversa fata de pagina.

    Nedat, se ia din `score.staves`, care merge de sus in jos si se intoarce.
    Asa merg si partiturile care nu sunt SATB: o cantare cu voce de solo are un
    portativ in plus, cu una sau doua voci pe el, si se scrie fara nimic special
    — `layout` a fost dintotdeauna general, doar valoarea implicita nu era.

    Fara nici una, nici alta, se presupune partitura stransa corala: (T, B) jos,
    (S, A) sus. E forma pe care o produce conducta noastra, deci partiturile
    construite in teste raman valide.

    `ties=False` scoate marcajele de legato — vezi `write_event`. Se da la
    scrierea tintelor de antrenare, fiindca modelul nu le poate reprezenta, si
    **nu** la scrierea unei partituri pentru om.
    """
    layout = layout or tuple(tuple(group) for group in reversed(score.staves)) or (
        ("T", "B"),
        ("S", "A"),
    )
    staves = [[score.part(part_id) for part_id in group] for group in layout]

    numbers = [measure.number for measure in staves[0][0].measures]
    if any(
        [m.number for m in voice.measures] != numbers
        for voices in staves
        for voice in voices
    ):
        raise UnwritableScore(
            "vocile nu au aceleasi masuri — un document Humdrum are un singur "
            "sir de bare pentru tot sistemul"
        )

    # Un portativ are **o singura cheie**, fiindca e o singura coloana pana la
    # `*^`. Doua voci care declara chei diferite nu stau pe acelasi portativ:
    # partitura aceea e scrisa pe patru portative, nu in partitura stransa, si
    # nu se poate scrie asa fara sa pierdem una dintre chei.
    for group, voices in zip(layout, staves):
        declared = {voice.context_at(voice.measures[0].number)[0] for voice in voices}
        if len(declared) > 1:
            raise UnwritableScore(
                f"vocile {'/'.join(group)} declara chei diferite ({declared}) — "
                "nu sunt pe acelasi portativ, deci nu e partitura stransa"
            )

    lines: list[str] = []
    columns = [1 for _ in staves]

    def emit(per_staff: list[list[str]]) -> None:
        lines.append("\t".join(field for staff in per_staff for field in staff))

    emit([["**kern"] * count for count in columns])
    for index, voices in enumerate(staves):
        clef, key, time = voices[0].context_at(numbers[0])
        for value in (write_clef(clef), write_key(key), write_time(time)):
            emit([[value] * count if i == index else [HOLD] * columns[i]
                  for i, count in enumerate(columns)])

    for position, number in enumerate(numbers):
        wanted = [_staff_columns(voices, number, ties) for voices in staves]
        for index in range(len(staves)):
            change = _transition(columns[index], wanted[index])
            if change is None:
                continue
            emit([
                change if i == index else [HOLD] * columns[i]
                for i in range(len(staves))
            ])
            columns[index] = wanted[index]

        bara = _barline_before(staves[0][0], position)
        emit([[bara] * count for count in columns])
        for moment in _moments(staves, number):
            emit([
                _fields_at(staves[i], number, columns[i], moment, ties)
                for i in range(len(staves))
            ])

    # Documentul se incheie cu `==`, fiindca e sfarsitul decupajului (vezi
    # `page._joinable`); cu puncte cand ultima masura inchide o repetitie.
    ultima = staves[0][0].measures[-1].barline
    emit([["==:|!" if ultima is not None and ultima.repeat else "=="] * count
          for count in columns])
    emit([["*-"] * count for count in columns])
    return "\n".join(lines) + "\n"
