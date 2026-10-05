"""Conversie `**kern` -> model muzical intern.

Modelele SMT nu scot MusicXML, ci Humdrum `**kern`. Modulul asta e puntea.
E cod determinist si testabil: nu importa torch si nu stie ce e o imagine.

Ce intelege (subsetul v1, plus ce cere muzica de cor)
-----------------------------------------------------
  `**kern` / `*-`   inceput si sfarsit de voce
  `*clefG2`         cheie; `*clefGv2` = Sol cu 8 dedesubt, cheia de tenor
  `*k[f#c#]`        armura
  `*M3/4`           metrica
  `=` `==`          bare de masura
  `4g` `8b-` `2.cc` note: durata, inaltime, alteratie, punct
  `4r`              pauze
  `[` `_` `]`       legaturi de durata

Ce ignora deliberat
-------------------
  `**mxhm` si tot ce e pe acea coloana — simboluri de acord de jazz;
  `*met(c)`, `*MM120`, `*^`, `*v` — prezentare, tempo, impartire de voci;
  `;` `J` `L` `/` `\\` — fermate, legaturi de codite, directii de coada.

Ce nu intelege e raportat, nu inghitit: `parse_spine` intoarce si o lista de
tokeni nerecunoscuti, ca sa nu descoperim tacut ca am pierdut jumatate din
partitura.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.music_model.model import (
    Accidental,
    Barline,
    BarlineStyle,
    Clef,
    Duration,
    KeySignature,
    Measure,
    Note,
    NoteType,
    Part,
    Pitch,
    Rest,
    Score,
    TimeSignature,
    Tuplet,
)

# Un token de nota: durata (cifre), eventual puncte, apoi literele inaltimii,
# apoi alteratiile. Restul caracterelor Humdrum sunt ornamente de notatie.
_NOTE = re.compile(r"^(?P<dur>\d+)(?P<dots>\.*)(?P<pitch>[a-gA-G]+)(?P<alter>[#\-n]*)")
_REST = re.compile(r"^(?P<dur>\d+)(?P<dots>\.*)r")
_KEY = re.compile(r"^\*k\[(?P<content>[a-g#\-]*)\]$")
_METER = re.compile(r"^\*M(?P<beats>\d+)/(?P<beat_type>\d+)$")
_CLEF = re.compile(r"^\*clef(?P<sign>[GFC])(?P<octave>[vV^]?)(?P<line>\d)$")

_DURATIONS = {
    1: NoteType.WHOLE,
    2: NoteType.HALF,
    4: NoteType.QUARTER,
    8: NoteType.EIGHTH,
    16: NoteType.SIXTEENTH,
    32: NoteType.THIRTY_SECOND,
}


class KernError(ValueError):
    """Tokenul e sintactic kern, dar in afara a ce sustine v1."""


@dataclass
class ParseResult:
    part: Part
    unrecognised: list[str] = field(default_factory=list)


def _split_power_of_two(value: int) -> tuple[int, int]:
    """Desparte `value` in (putere a lui doi, rest impar)."""
    power = 1
    while value % 2 == 0:
        value //= 2
        power *= 2
    return power, value


def parse_duration(digits: str, dots: str) -> Duration:
    """Durata Humdrum, care e **reciproca**: 4 = patrime, 8 = optime.

    Un numar care nu e putere a lui doi inseamna un grup neregulat, si asta e
    regula generala din care ies trioletele: 12 = 3x4, deci trei note in timpul
    a doua, scrise ca optimi. Verificare: durata sunata iese `4/12 = 1/3` de
    patrime, adica exact o optime de triolet.

    Partea impara spune cate note se scriu; cea normala e cea mai mare putere a
    lui doi care incape in ea — 3:2 pentru triolet, 5:4 pentru cvintolet.
    """
    value = int(digits)
    if value < 1:
        raise KernError(f"durata {value} nu e un numar valid")

    power, odd = _split_power_of_two(value)
    if odd == 1:
        if value not in _DURATIONS:
            raise KernError(f"durata {value} nu e sustinuta in v1")
        return Duration(_DURATIONS[value], len(dots))

    normal = 1
    while normal * 2 <= odd:
        normal *= 2
    written = power * normal
    if written not in _DURATIONS:
        raise KernError(f"durata {value} nu e sustinuta in v1")
    return Duration(_DURATIONS[written], len(dots), Tuplet(odd, normal))


def parse_pitch(letters: str, alterations: str) -> Pitch:
    """`c` = Do4, `cc` = Do5, `C` = Do3, `CC` = Do2 — conventia Humdrum.

    Literele mici urca de la octava 4, cele mari coboara de la octava 3.
    """
    step = letters[0].upper()
    if any(letter.upper() != step for letter in letters):
        raise KernError(f"inaltime inconsistenta: {letters!r}")

    count = len(letters)
    octave = 3 + count if letters[0].islower() else 4 - count

    alter = alterations.count("#") - alterations.count("-")
    accidental = None
    if "n" in alterations:
        accidental = Accidental.NATURAL
    elif "#" in alterations:
        accidental = Accidental.SHARP
    elif "-" in alterations:
        accidental = Accidental.FLAT

    return Pitch(step, octave, alter, accidental)


def parse_key(content: str) -> KeySignature:
    """`*k[f#c#]` -> 2 diezi; `*k[b-e-]` -> 2 bemoli; `*k[]` -> Do major."""
    if not content:
        return KeySignature(0)
    sharps = content.count("#")
    flats = content.count("-")
    if sharps and flats:
        raise KernError(f"armura amestecata: {content!r}")
    return KeySignature(sharps if sharps else -flats)


def parse_clef(sign: str, octave: str, line: str) -> Clef:
    return Clef(sign, int(line), octave_change=-1 if octave.lower() == "v" else 0)


def _strip_ornaments(token: str) -> str:
    """Scoate ce tine de desen, nu de continut muzical."""
    return re.sub(r"[;JLKk\\/()<>~^'\"`Tt]", "", token)


def parse_spine(text: str, part_id: str = "S", name: str = "") -> ParseResult:
    """Citeste o singura coloana `**kern` si construieste o `Part`.

    Coloana `**mxhm` (acorduri de jazz), daca exista, e sarita: din fiecare
    linie luam doar primul camp separat prin tab.

    Pentru portative cu doua voci (partitura stransa), foloseste
    `parse_document`, care descurca intai `*^` si `*v`.
    """
    return _parse_lines(
        [line.split("\t")[0] for line in text.splitlines()], part_id=part_id, name=name
    )


def _parse_lines(lines: list[str], part_id: str = "S", name: str = "") -> ParseResult:
    """Construieste o `Part` dintr-un flux de tokeni ai unei singure voci."""
    part = Part(id=part_id, name=name)
    unrecognised: list[str] = []

    measure = Measure(number=1)
    pending_attributes: dict[str, object] = {}

    def flush() -> None:
        if measure.events or measure.clef or measure.key or measure.time:
            part.measures.append(measure)

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue

        if line in ("**kern", "*-", "**mxhm") or line.startswith("*MM"):
            continue
        if line.startswith("*met(") or line in ("*", "*^", "*v", "*+"):
            continue

        # `.` e tokenul null Humdrum: „aici nu incepe nimic", fiindca
        # evenimentul e pe alta coloana. Se sare in liniste. Pana la
        # partitura stransa citita ca un singur document cu mai multe
        # coloane, cazul nu aparea: coloana `**mxhm` era aruncata intreaga.
        if line == ".":
            continue

        if line.startswith("="):
            # In Humdrum, repetitia se scrie cu doua puncte in tokenul barei:
            # `=:|!` inchide, `=!|:` deschide, `=:|!|:` amandoua. Peste o bara de
            # repetitie nu se lipesc masuri (`app/recognition/page.py`): masura
            # de dinaintea ei e scurta dinadins, si ar parea taiata de hotarul
            # dintre sisteme. Un `|:` chiar la capatul documentului n-are masura
            # pe care s-o deschida si se pierde; MuseScore si imnarele il
            # deseneaza la inceputul sistemului urmator, unde se citeste.
            inchide, deschide = ":|" in line, "|:" in line
            # Bara dinaintea primei note **deschide** masura 1, nu o inchide pe
            # una anterioara. Fara verificarea asta, fiecare voce ar incepe cu o
            # masura goala si toata numerotarea ar fi decalata cu unu.
            if not measure.events:
                measure.repeat_start = measure.repeat_start or deschide
                continue
            flush()
            style = BarlineStyle.FINAL if line.startswith("==") else BarlineStyle.REGULAR
            part.measures[-1].barline = Barline(style, repeat=inchide)
            measure = Measure(number=len(part.measures) + 1, repeat_start=deschide)
            for key, value in pending_attributes.items():
                setattr(measure, key, value)
            pending_attributes.clear()
            continue

        if match := _CLEF.match(line):
            clef = parse_clef(match["sign"], match["octave"], match["line"])
            _set_attribute(measure, pending_attributes, "clef", clef)
            continue
        if match := _KEY.match(line):
            _set_attribute(measure, pending_attributes, "key", parse_key(match["content"]))
            continue
        if match := _METER.match(line):
            time = TimeSignature(int(match["beats"]), int(match["beat_type"]))
            _set_attribute(measure, pending_attributes, "time", time)
            continue
        if line.startswith("*"):
            unrecognised.append(line)
            continue

        # Un spatiu in interiorul coloanei desparte notele unui acord. Acordurile
        # sunt in afara subsetului v1 (§5.2), iar `_NOTE` ar potrivi tacut doar
        # prima nota si le-ar pierde pe celelalte. Pastram prima — masura ramane
        # cu durata corecta — dar spunem ca am pierdut ceva.
        if " " in line:
            unrecognised.append(f"{line} (acord: s-a pastrat doar prima nota)")
            line = line.split(" ")[0]

        token = _strip_ornaments(line)
        # `[` deschide o legatura, `]` o inchide, `_` face amandoua: nota din
        # mijlocul unei legaturi peste mai multe masuri.
        tie_start = "[" in token or "_" in token
        tie_stop = "]" in token or "_" in token
        token = token.replace("[", "").replace("]", "").replace("_", "")

        # Un token in afara subsetului v1 — un triolet, de pilda — se
        # raporteaza si se sare. Sa aruncam din cauza lui ar insemna sa pierdem
        # tot portativul, si odata cu el toata pagina.
        try:
            if match := _REST.match(token):
                measure.events.append(Rest(parse_duration(match["dur"], match["dots"])))
                continue
            if match := _NOTE.match(token):
                measure.events.append(
                    Note(
                        parse_pitch(match["pitch"], match["alter"]),
                        parse_duration(match["dur"], match["dots"]),
                        tie_start=tie_start,
                        tie_stop=tie_stop,
                    )
                )
                continue
        except (KernError, ValueError) as error:
            unrecognised.append(f"{line} ({error})")
            continue

        unrecognised.append(line)

    flush()
    return ParseResult(part=part, unrecognised=unrecognised)


def _set_attribute(measure: Measure, pending: dict, name: str, value) -> None:
    """Atributele dinaintea primei bare apartin masurii curente.

    Daca masura are deja evenimente, schimbarea e pentru masura urmatoare —
    kern pune schimbarile de cheie sau metrica *inaintea* barei.
    """
    if measure.events:
        pending[name] = value
    else:
        setattr(measure, name, value)


def split_voices(
    text: str,
    problems: list[str] | None = None,
    origins: list[int] | None = None,
) -> list[list[str]]:
    """Descurca un document Humdrum in cate un flux de tokeni per voce.

    Partitura stransa — doua voci pe acelasi portativ, cu codite opuse — se
    scrie in Humdrum prin **impartirea coloanei**: `*^` desface o coloana in
    doua, `*v *v` le lipeste la loc. Intre ele, fiecare linie are un camp in
    plus.

    Doua decizii care se vad in rezultat:

    * **La reunire, tokenii se scriu in amandoua vocile.** Cand corul canta la
      unison, ambele voci impart aceeasi nota. Daca am pune-o intr-una singura,
      cealalta ar ramane cu masuri incomplete si validatorul ar semnala erori
      care nu exista.
    * **Coloana `**mxhm` e aruncata de tot** — sunt acorduri de jazz. Numai
      coloanele `**kern` produc voci.

    Intoarce cate o lista de linii per voce. **Ordinea e cea a crearii, nu a
    paginii**: vocile nascute dintr-un `*^` se adauga la coada, nu langa sora
    lor. Cine are nevoie sa stie pe ce portativ sta fiecare voce trebuie deci sa
    ceara `origins` — altfel ar grupa dupa pozitie si ar pune pe acelasi
    portativ coloane venite din portative diferite.

    `origins`, daca se da, primeste pentru fiecare voce **din a cata coloana
    `**kern` initiala se trage** — 0 fiind cea din stanga, adica portativul de
    jos. O voce nascuta dintr-un `*^` mosteneste coloana parintelui.
    """
    voices: list[list[str]] = []
    # `tracks[i]` = indicii vocilor pe care le alimenteaza coloana i
    tracks: list[list[int]] = []
    origin: list[int] = []

    for number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.rstrip("\n")
        if not line.strip() or line.startswith("!"):
            continue
        fields = line.split("\t")

        if not tracks:
            for field in fields:
                if field.strip() == "**kern":
                    voices.append([])
                    origin.append(len(origin))
                    tracks.append([len(voices) - 1])
                else:
                    tracks.append([])
            continue

        # `*^` si `*v` schimba numarul de coloane, deci se trateaza inaintea
        # distribuirii tokenilor.
        if any(field.strip() in ("*^", "*v") for field in fields):
            tracks = _restructure(fields, tracks, voices, origin)
            continue

        # `zip` ar taia tacut campurile in plus. Un model care scoate un rand cu
        # patru campuri acolo unde structura are trei coloane nu a scris un
        # document Humdrum valid, iar noi am pierde muzica fara sa spunem nimic.
        if problems is not None and len(fields) != len(tracks):
            problems.append(
                f"linia {number}: {len(fields)} campuri, structura are {len(tracks)} coloane"
            )

        for field, targets in zip(fields, tracks):
            for voice in targets:
                voices[voice].append(field)

    if origins is not None:
        origins.extend(origin)
    return voices


def _restructure(
    fields: list[str],
    tracks: list[list[int]],
    voices: list[list[str]],
    origin: list[int] | None = None,
) -> list[list[int]]:
    """Aplica o linie de `*^` / `*v` peste structura de coloane."""
    updated: list[list[int]] = []
    index = 0
    while index < len(fields):
        marker = fields[index].strip()
        targets = tracks[index] if index < len(tracks) else []

        if marker == "*^":
            # Ramura stanga pastreaza vocea existenta, cea dreapta primeste una
            # noua, care **mosteneste tot ce s-a scris pana aici**: cheia,
            # armura, metrica, dar si barele si notele.
            #
            # Doua motive, amandoua verificate pe iesirea reala a modelului:
            #
            # * fara cheie mostenita, vocea de jos a unui portativ in cheia Fa
            #   ar fi citita in cheia Sol, fiindca declaratia a aparut inaintea
            #   impartirii;
            # * fara barele mostenite, o voce care apare **la mijlocul piesei**
            #   isi numara masurile de la 1 si se desincronizeaza de restul.
            #   Cazul apare de fiecare data cand T&B intra mai tarziu sau cand
            #   un portativ se desparte in divisi dupa un pasaj la unison.
            #
            # Ca mostenim si muzica, nu doar declaratiile, e aceeasi hotarare ca
            # la `*v`: pana la impartire, o singura coloana inseamna ca vocile
            # canta la unison, deci notele sunt ale amandurora.
            #
            # **Dar numai cand coloana hranea o singura voce.** Dupa un `*v`,
            # coloana hraneste doua voci unite la unison, iar despartirea le da
            # inapoi — cate una pe ramura — in loc sa inventeze a treia. Fara
            # regula asta, un portativ care se desparte, se uneste si se
            # desparte iar iesea cu trei voci in loc de doua, **iar vocea a doua
            # primea linia primei**: ramura stanga mergea la amandoua vocile
            # unite, iar linia ei ajungea la vocea inventata. De acolo veneau
            # cele 8-11 voci de pe pagina 3 (§15), nu din model.
            if len(targets) > 1:
                updated.append(targets[:1])
                updated.append(targets[1:])
                index += 1
                continue

            source = targets[0] if targets else None
            inherited = [
                line for line in (voices[source] if source is not None else [])
                if line.strip() != "*-"
            ]
            voices.append(inherited)
            if origin is not None:
                # Vocea noua sta pe portativul parintelui, nu pe altul.
                origin.append(origin[source] if source is not None else 0)
            updated.append(targets)
            updated.append([len(voices) - 1])
            index += 1
            continue

        if marker == "*v":
            merged: list[int] = []
            while index < len(fields) and fields[index].strip() == "*v":
                merged.extend(tracks[index] if index < len(tracks) else [])
                index += 1
            updated.append(merged)
            continue

        updated.append(targets)
        index += 1
    return updated


def parse_document(
    text: str, names: list[tuple[str, str]] | None = None
) -> list[ParseResult]:
    """Citeste un document Humdrum complet, cu oricate voci pe portativ."""
    streams = split_voices(text)
    names = names or [(f"V{i}", "") for i in range(1, len(streams) + 1)]
    if len(streams) != len(names):
        raise ValueError(f"{len(streams)} voci gasite, {len(names)} nume date")
    return [
        _parse_lines(stream, part_id=part_id, name=name)
        for stream, (part_id, name) in zip(streams, names)
    ]


CLOSED_SCORE_SATB = (
    (("S", "Soprano"), ("A", "Alto")),  # portativul de sus, cheia Sol
    (("T", "Tenor"), ("B", "Bass")),  # portativul de jos, cheia Fa
)


def parse_closed_score(staff_texts: list[str]) -> Score:
    """Partitura stransa: doua portative, cate doua voci pe fiecare.

    Asta e forma in care scrie utilizatorul. Iesirea are tot patru `Part`-uri
    separate, nu doua cu cate doua voci: SATB pe patru portative e reprezentarea
    corecta muzical, e mai usor de corectat nota cu nota, iar in MuseScore se
    strange inapoi in partitura stransa cu o singura comanda (Implode).

    Vocea de sus a fiecarui portativ e cea cu codite in sus. Humdrum pastreaza
    ordinea coloanelor, iar `*^` pune ramura noua la dreapta — deci ordinea din
    `CLOSED_SCORE_SATB` corespunde ordinii de pe pagina.
    """
    if len(staff_texts) != 2:
        raise ValueError(f"partitura stransa cere 2 portative, primite {len(staff_texts)}")

    score = Score(staves=[[part_id for part_id, _ in names] for names in CLOSED_SCORE_SATB])
    for text, names in zip(staff_texts, CLOSED_SCORE_SATB):
        # Numaram vocile inainte de a le da nume: daca modelul a ratat `*^`,
        # vrem sa spunem *care* portativ si de ce, nu doar ca numerele nu se
        # potrivesc.
        found = len(split_voices(text))
        if found != 2:
            raise ValueError(
                f"portativul cu vocile {names[0][0]}/{names[1][0]} a dat "
                f"{found} voci in loc de 2 — probabil lipseste `*^`"
            )
        score.parts.extend(result.part for result in parse_document(text, list(names)))
    return score


def merge_voices(results: list[ParseResult], voices: list[tuple[str, str]]) -> Score:
    """Imbina portativele unui sistem intr-o partitura cu mai multe parti.

    Pasul asta e determinist si separat de model intentionat: daca modelul
    citeste gresit un portativ, vrem ca greseala sa apara ca o voce
    desincronizata, pe care validatorul o semnaleaza, nu topita in restul.
    """
    if len(results) != len(voices):
        raise ValueError(f"{len(results)} portative pentru {len(voices)} voci")

    score = Score()
    for result, (part_id, name) in zip(results, voices):
        result.part.id = part_id
        result.part.name = name
        score.parts.append(result.part)
    return score


# --------------------------------------------------------------------------
# Un sistem intreg, citit dintr-un singur document
# --------------------------------------------------------------------------
#
# `smt-grandstaff` nu citeste un portativ, ci un **sistem**: doua portative
# intr-o singura imagine, deci toate vocile intr-un singur document Humdrum.
# `parse_closed_score` nu se aplica — el cere cate un text per portativ.

UPPER_STAFF = (("S", "Soprano"), ("A", "Alto"))
LOWER_STAFF = (("T", "Tenor"), ("B", "Bass"))


@dataclass
class SystemParse:
    """Un sistem citit: partitura, plus tot ce a fost nesigur pe drum."""

    score: Score
    problems: list[str] = field(default_factory=list)
    unrecognised: list[str] = field(default_factory=list)


def _first_clef(lines: list[str]) -> Clef | None:
    for line in lines:
        if match := _CLEF.match(line.strip()):
            return parse_clef(match["sign"], match["octave"], match["line"])
    return None


def _is_lower_staff(clef: Clef | None) -> bool:
    """Cheia spune pe ce portativ sta vocea, nu pozitia coloanei.

    Fa4 si Sol-cu-8-dedesubt sunt chei de portativ de jos (T&B); Sol2 curat e
    portativul de sus (S&A). Ne sprijinim pe cheie fiindca e singura informatie
    din document care nu depinde de o conventie de ordonare neverificata.
    """
    if clef is None:
        return False
    return clef.sign in ("F", "C") or clef.octave_change < 0


def _mean_height(part: Part) -> float | None:
    """Cat de sus sta vocea pe portativ, ca medie. `None` daca n-are note."""
    heights = [
        event.pitch.diatonic_index
        for measure in part.measures
        for event in measure.events
        if isinstance(event, Note)
    ]
    return sum(heights) / len(heights) if heights else None


def _parse_declared(
    streams: list[list[str]],
    origins: list[int],
    layout: Sequence[Sequence[str]],
    problems: list[str],
    unrecognised: list[str],
) -> SystemParse:
    """Citirea cand asezarea pe portative e declarata."""
    score = Score(staves=[list(group) for group in layout])
    # Coloanele vin de jos in sus; partile se asaza de sus in jos, ca pe pagina
    # si ca in `Score.staves` — altfel exportul MusicXML ar scrie basul primul.
    per_staff: list[list[Part]] = []
    assigned = _by_declared_layout(streams, origins, layout, problems)

    # A doua parere, si singura pe care cheia o mai poate da aici. **Gruparea**
    # nu se poate verifica prin cheie: se face pe coloana de origine, iar o
    # coloana are o singura cheie, deci verificarea ar fi mereu multumita. Ce se
    # poate verifica e **ordinea coloanelor din document**: ele merg de jos in
    # sus, deci o cheie de portativ jos (Fa, Do, Sol-8) nu are ce cauta deasupra
    # uneia inalte. Cand se intampla, modelul a scris portativele in alta ordine
    # decat cea de pe pagina, iar numele declarate ajung pe muzica gresita.
    #
    # Se raporteaza, nu se corecteaza: nu stim care dintre declaratie si model
    # greseste. Acelasi tipar ca `_mean_height`.
    seen_upper = False
    for index, (staff, names) in enumerate(assigned[: len(layout)]):
        lower = _is_lower_staff(_first_clef(staff[0])) if staff else False
        if lower and seen_upper:
            problems.append(
                f"portativul {index + 1} de jos ({'/'.join(n for n, _ in names)}) "
                "are cheie de portativ jos, dar modelul l-a scris deasupra unuia "
                "cu cheie inalta — coloanele nu sunt in ordinea de pe pagina, "
                "deci numele declarate cad pe alta muzica"
            )
        seen_upper = seen_upper or not lower

    for index, (staff, names) in enumerate(assigned):
        parts: list[Part] = []
        for stream, (part_id, name) in zip(staff, names):
            result = _parse_lines(stream, part_id=part_id, name=name)
            unrecognised.extend(result.unrecognised)
            parts.append(result.part)
        per_staff.append(parts)

    # Coloanele peste asezare raman la coada: nu stim pe ce portativ stau.
    placed = per_staff[: len(layout)]
    for parts in reversed(placed):
        score.parts.extend(parts)
    for parts in per_staff[len(layout):]:
        score.parts.extend(parts)
    return SystemParse(score=score, problems=problems, unrecognised=unrecognised)


SATB_NAMES = dict(UPPER_STAFF + LOWER_STAFF)


def _named(part_id: str) -> tuple[str, str]:
    """Id-ul si numele lizibil al unei voci declarate."""
    return part_id, SATB_NAMES.get(part_id, "")


def _unique(base: str, used: set[str]) -> str:
    """Un id generic care nu se ciocneste de nimic folosit deja.

    Numele inventate pentru coloanele peste declaratie ar putea nimeri exact
    peste un id declarat de utilizator — `X1` plus un `3` da `X13`, care poate
    fi si numele unei voci declarate. Doua parti cu acelasi id fac
    `Score.part()` sa intoarca mereu prima, iar exportul scria doua `score-part`
    cu acelasi id: **MusicXML invalid**, produs de codul nostru. E defectul din
    §15.6, si merita aparat acolo unde poate reaparea.
    """
    candidate = base
    suffix = 1
    while candidate in used:
        suffix += 1
        candidate = f"{base}_{suffix}"
    used.add(candidate)
    return candidate


def _by_declared_layout(
    streams: list[list[str]],
    origins: list[int],
    layout: Sequence[Sequence[str]],
    problems: list[str],
) -> list[tuple[list[list[str]], list[tuple[str, str]]]]:
    """Imparte coloanele pe portative dupa asezarea **declarata**.

    Gruparea se face pe `origins` — din a cata coloana initiala se trage fiecare
    voce — nu pe pozitia din lista si nu pe cheie. Pozitia n-ar merge fiindca
    `split_voices` adauga vocile nascute din `*^` la coada, deci vocile unui
    portativ nu sunt alaturate. Cheia n-ar merge fiindca pe o cantare cu solo nu
    mai raspunde: un portativ de solo in cheia Sol e identic la cheie cu
    portativul de sopran.

    Cheia nu se arunca, devine **a doua parere**, verificata de apelant: acelasi
    tipar ca inaltimea medie a vocilor. Cand cele doua se contrazic, o spunem,
    fiindca amandoua pot gresi.

    Coloanele merg de jos in sus, `layout` merge de sus in jos ca pe pagina,
    deci se intoarce: portativul `k` de jos primeste vocile din `origins == k`.
    """
    groups = list(reversed([list(group) for group in layout]))
    used = {voice for group in groups for voice in group}
    per_origin: dict[int, list[list[str]]] = {}
    for stream, origin in zip(streams, origins):
        per_origin.setdefault(origin, []).append(stream)

    out: list[tuple[list[list[str]], list[tuple[str, str]]]] = []
    for index, group in enumerate(groups):
        staff = per_origin.pop(index, [])
        if not staff:
            problems.append(
                f"portativul {index + 1} de jos ({'/'.join(group)}) n-a primit "
                "nicio coloana — documentul are mai putine decat asezarea declarata"
            )
            continue
        if len(staff) == 1 and len(group) > 1:
            # Unison scris pe o singura coloana — aceeasi hotarare ca la `*v`.
            problems.append(
                f"portativul {index + 1} de jos a dat o singura voce — a fost "
                f"scrisa in toate ({'/'.join(group)}), ca la unison"
            )
            staff = [staff[0]] + [list(staff[0]) for _ in group[1:]]
        names = [_named(part_id) for part_id in group]
        if len(staff) > len(group):
            problems.append(
                f"portativul {index + 1} de jos a dat {len(staff)} voci in loc "
                f"de {len(group)} — cele in plus au fost pastrate cu nume generice"
            )
            names += [
                (_unique(f"X{index + 1}_{i}", used), "")
                for i in range(len(group) + 1, len(staff) + 1)
            ]
        out.append((staff, names))

    for origin in sorted(per_origin):
        extra = per_origin[origin]
        problems.append(
            f"modelul a scris un portativ peste asezarea declarata (coloana "
            f"{origin + 1} de jos, {len(extra)} voci) — pastrat cu nume generice, "
            "ca sa nu dispara muzica pe care chiar a scris-o"
        )
        out.append(
            (extra, [(_unique(f"Y{origin + 1}_{i}", used), "") for i in range(1, len(extra) + 1)])
        )
    return out


def parse_open_system(
    text: str,
    staves: int | None = None,
    layout: Sequence[Sequence[str]] | None = None,
) -> SystemParse:
    """Un document cu toate vocile unui sistem -> partitura.

    `layout` declara ce voci stau pe fiecare portativ, **de sus in jos**, ca
    `Score.staves`. Dat, repartizarea se face pe ordinea coloanelor si numele
    vin de acolo — asa se citesc si sistemele care nu sunt SATB, de pilda o
    cantare cu voce de solo pe portativ separat. Nedat, se pastreaza
    repartizarea dupa cheie de mai jos, care merge doar pe SATB.

    Trei lucruri pe care le rezolva, si care nu apar cand fiecare portativ e
    transcris separat:

    * **ordinea coloanelor e inversa ordinii de pe pagina.** Humdrum pune vocea
      cea mai joasa in stanga, deci prima coloana e portativul de jos. De asta
      repartizarea se face dupa cheie, nu dupa pozitie.
    * **un sistem poate avea intre una si patru voci.** Primul sistem al unei
      piese poate avea doar S&A, daca T&B intra mai tarziu.
    * **un portativ cu o singura coloana inseamna unison**, nu o voce lipsa:
      tokenii se scriu in amandoua vocile. Aceeasi hotarare ca la `*v` in
      `split_voices` — altfel vocea cealalta ar ramane cu masuri goale si
      validatorul ar raporta erori inexistente.

    `staves` e numarul de portative pe care **detectorul** le-a gasit in
    imagine. Cand modelul declara mai multe decat atat, a inventat un portativ:
    `smt-grandstaff` a fost antrenat numai pe sisteme de doua portative, si pe
    unul singur incearca sa produca tot doua. Nu ghicim care e cel inventat — il
    raportam, fiindca diferenta se vede imediat in MusicXML.
    """
    # Randurile stricate se numara, nu se insira: pe o transcriere iesita prost
    # sunt zeci, si ar ineca restul raportului. Primul e pastrat ca exemplu.
    mismatched: list[str] = []
    origins: list[int] = []
    streams = split_voices(text, mismatched, origins)
    problems: list[str] = []
    if mismatched:
        problems.append(
            f"{len(mismatched)} rand{'uri' if len(mismatched) > 1 else ''} cu alt "
            f"numar de campuri decat structura documentului "
            f"(primul — {mismatched[0]}); modelul n-a scris un document Humdrum valid"
        )

    if not streams:
        raise ValueError(
            "documentul nu contine nicio coloana `**kern` — daca vine de la "
            "`smt-grandstaff`, trebuie trecut intai prin `bekern.to_kern`"
        )

    unrecognised: list[str] = []

    # `if layout:`, nu `is not None`: o asezare **goala** inseamna „nu se stie",
    # exact cum o documenteaza `Score.staves`. Tratata ca declaratie, ar spune
    # „partitura are zero portative" si n-ar mai potrivi nicio voce.
    if layout:
        return _parse_declared(streams, origins, layout, problems, unrecognised)

    upper: list[list[str]] = []
    lower: list[list[str]] = []
    for stream in streams:
        clef = _first_clef(stream)
        if clef is None:
            problems.append(
                "o voce nu declara nicio cheie — a fost pusa pe portativul de sus"
            )
        (lower if _is_lower_staff(clef) else upper).append(stream)

    declared = sum(1 for group in (upper, lower) if group)
    if staves is not None and declared > staves:
        problems.append(
            f"modelul a declarat {declared} portative, iar in imagine sunt "
            f"{staves} — a inventat unul, deci una dintre chei nu e reala"
        )

    score = Score()
    for label_prefix, (staff, names, label) in enumerate(
        ((upper, UPPER_STAFF, "de sus"), (lower, LOWER_STAFF, "de jos")), 1
    ):
        if not staff:
            continue
        if len(staff) == 1:
            # Unison scris pe o singura coloana. Vezi docstring-ul.
            staff = [staff[0], list(staff[0])]
            problems.append(
                f"portativul {label} a dat o singura voce — a fost scrisa in "
                f"amandoua ({names[0][0]} si {names[1][0]}), ca la unison"
            )
        if len(staff) > len(names):
            problems.append(
                f"portativul {label} a dat {len(staff)} voci in loc de "
                f"{len(names)} — cele in plus au fost pastrate cu nume generice"
            )

        parts: list[Part] = []
        for index, stream in enumerate(staff):
            # Numele generice trebuie sa fie unice pe **tot sistemul**, nu pe
            # portativ: numarate per portativ, a treia voce de sus si a treia de
            # jos iesesc amandoua „X3". Doua parti cu acelasi id fac
            # `Score.part()` sa intoarca mereu prima, iar exportul scria doua
            # `score-part` cu acelasi id — MusicXML invalid. Vazut pe pagina 3.
            part_id, name = (
                names[index]
                if index < len(names)
                else (f"X{label_prefix}{index + 1}", "")
            )
            result = _parse_lines(stream, part_id=part_id, name=name)
            unrecognised.extend(result.unrecognised)
            parts.append(result.part)

        # Ordinea coloanelor spune care voce e sus; inaltimea medie e a doua
        # parere. Cand se contrazic, o spunem — nu alegem in tacere, fiindca
        # amandoua pot gresi: modelul poate incurca vocile, iar conventia de
        # ordonare dupa `*^` nu e verificata pentru aceste greutati.
        heights = [_mean_height(part) for part in parts[: len(names)]]
        if len(heights) == 2 and None not in heights and heights[1] > heights[0]:
            problems.append(
                f"pe portativul {label}, vocea {parts[1].id} sta mai sus decat "
                f"{parts[0].id} — verifica daca nu s-au inversat"
            )
        score.parts.extend(parts)

    return SystemParse(score=score, problems=problems, unrecognised=unrecognised)
