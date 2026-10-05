"""Serializare model intern -> MusicXML 4.0 partwise.

Modulul nu importa nimic din `recognition/` si nu stie ce e un model AI.
Primeste un `Score` si scoate un fisier care trebuie sa se deschida in
MuseScore fara erori — acesta e criteriul de succes al etapei 4.

MusicXML masoara duratele in `divisions` pe patrime, ca numere *intregi*.
Modelul intern le tine ca `Fraction`, deci calculam divisions ca cel mai mic
numitor comun al tuturor duratelor din partitura. Asa nu pierdem nimic la
rotunjire, indiferent ce valori de nota apar.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from math import gcd
from pathlib import Path

from app.music_model.model import (
    Barline,
    Clef,
    KeySignature,
    Measure,
    Note,
    Part,
    Rest,
    Score,
    TimeSignature,
)

DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def divisions_for(score: Score) -> int:
    """Cate subdiviziuni pe patrime sunt necesare ca toate duratele sa fie intregi."""
    denominator = 1
    for part in score.parts:
        for measure in part.measures:
            for event in measure.events:
                denominator = _lcm(denominator, event.quarter_length.denominator)
    return denominator


def _duration_ticks(quarter_length: Fraction, divisions: int) -> int:
    ticks = quarter_length * divisions
    if ticks.denominator != 1:
        raise ValueError(
            f"durata {quarter_length} nu e reprezentabila cu divisions={divisions}"
        )
    return int(ticks)


def _sub(parent: ET.Element, tag: str, text: str | int | None = None) -> ET.Element:
    element = ET.SubElement(parent, tag)
    if text is not None:
        element.text = str(text)
    return element


def _write_attributes(
    parent: ET.Element,
    divisions: int | None,
    key: KeySignature | None,
    time: TimeSignature | None,
    clef: Clef | None,
) -> None:
    if divisions is None and key is None and time is None and clef is None:
        return
    attributes = _sub(parent, "attributes")
    # Ordinea elementelor e impusa de DTD: divisions, key, time, ..., clef.
    if divisions is not None:
        _sub(attributes, "divisions", divisions)
    if key is not None:
        node = _sub(attributes, "key")
        _sub(node, "fifths", key.fifths)
        _sub(node, "mode", key.mode)
    if time is not None:
        node = _sub(attributes, "time")
        _sub(node, "beats", time.beats)
        _sub(node, "beat-type", time.beat_type)
    if clef is not None:
        node = _sub(attributes, "clef")
        _sub(node, "sign", clef.sign)
        _sub(node, "line", clef.line)
        if clef.octave_change:
            _sub(node, "clef-octave-change", clef.octave_change)


def tuplet_marks(events: list) -> list[tuple[bool, bool]]:
    """Pentru fiecare eveniment: (incepe un grup, inchide un grup).

    MusicXML cere doua lucruri diferite pentru un triolet: `time-modification`
    pe **fiecare** nota, care spune cat dureaza, si `notations/tuplet` doar pe
    prima si pe ultima, care deseneaza paranteza cu „3". Fara a doua, MuseScore
    ar reda duratele corect dar n-ar arata grupul.

    Grupurile se deduc din siruri de evenimente vecine cu acelasi raport. E o
    aproximatie constienta: doua triolete alaturate in aceeasi masura ies ca un
    grup lung. Ritmul iese corect oricum, si nu avem de unde sti granita —
    `**kern` nu o scrie.
    """
    marks: list[tuple[bool, bool]] = []
    for index, event in enumerate(events):
        current = event.duration.tuplet
        if current is None:
            marks.append((False, False))
            continue
        previous = events[index - 1].duration.tuplet if index > 0 else None
        following = events[index + 1].duration.tuplet if index + 1 < len(events) else None
        marks.append((previous != current, following != current))
    return marks


def _write_time_modification(node: ET.Element, duration) -> None:
    if duration.tuplet is None:
        return
    modification = _sub(node, "time-modification")
    _sub(modification, "actual-notes", duration.tuplet.actual)
    _sub(modification, "normal-notes", duration.tuplet.normal)


def _write_notations(
    node: ET.Element, tie_start: bool, tie_stop: bool, tuplet: tuple[bool, bool]
) -> None:
    starts, stops = tuplet
    if not (tie_start or tie_stop or starts or stops):
        return
    notations = _sub(node, "notations")
    if tie_stop:
        ET.SubElement(notations, "tied", {"type": "stop"})
    if tie_start:
        ET.SubElement(notations, "tied", {"type": "start"})
    if starts:
        ET.SubElement(notations, "tuplet", {"type": "start"})
    if stops:
        ET.SubElement(notations, "tuplet", {"type": "stop"})


def _write_note(
    parent: ET.Element, note: Note, divisions: int, tuplet: tuple[bool, bool] = (False, False)
) -> None:
    node = _sub(parent, "note")
    pitch = _sub(node, "pitch")
    # DTD: step, alter, octave — in aceasta ordine.
    _sub(pitch, "step", note.pitch.step)
    if note.pitch.alter:
        _sub(pitch, "alter", note.pitch.alter)
    _sub(pitch, "octave", note.pitch.octave)
    _sub(node, "duration", _duration_ticks(note.quarter_length, divisions))
    # <tie> descrie sunetul, <tied> descrie desenul. MusicXML le cere pe ambele.
    if note.tie_stop:
        ET.SubElement(node, "tie", {"type": "stop"})
    if note.tie_start:
        ET.SubElement(node, "tie", {"type": "start"})
    _sub(node, "voice", 1)
    _sub(node, "type", note.duration.note_type.value)
    for _ in range(note.duration.dots):
        _sub(node, "dot")
    if note.pitch.accidental is not None:
        _sub(node, "accidental", note.pitch.accidental.value)
    # Ordinea elementelor e impusa de DTD: accidental, apoi time-modification,
    # si abia la sfarsit notations.
    _write_time_modification(node, note.duration)
    _write_notations(node, note.tie_start, note.tie_stop, tuplet)


def _write_rest(
    parent: ET.Element, rest: Rest, divisions: int, tuplet: tuple[bool, bool] = (False, False)
) -> None:
    node = _sub(parent, "note")
    _sub(node, "rest")
    _sub(node, "duration", _duration_ticks(rest.quarter_length, divisions))
    _sub(node, "voice", 1)
    _sub(node, "type", rest.duration.note_type.value)
    for _ in range(rest.duration.dots):
        _sub(node, "dot")
    _write_time_modification(node, rest.duration)
    _write_notations(node, False, False, tuplet)


def _write_barline(parent: ET.Element, barline: Barline) -> None:
    node = ET.SubElement(parent, "barline", {"location": "right"})
    _sub(node, "bar-style", barline.style.value)


def _write_measure(
    parent: ET.Element, measure: Measure, divisions: int, first: bool
) -> None:
    attributes = {"number": str(measure.number)}
    if measure.partial:
        # `implicit="yes"` e felul in care MusicXML spune „masura asta e mai
        # scurta dinadins". Fara el, dus-intorsul prin MuseScore ar transforma
        # anacruza intr-o masura pe care orice cititor o socoteste stricata.
        attributes["implicit"] = "yes"
    node = ET.SubElement(parent, "measure", attributes)
    _write_attributes(
        node,
        divisions if first else None,
        measure.key,
        measure.time,
        measure.clef,
    )
    marks = tuplet_marks(measure.events)
    for event, tuplet in zip(measure.events, marks):
        if isinstance(event, Note):
            _write_note(node, event, divisions, tuplet)
        else:
            _write_rest(node, event, divisions, tuplet)
    if measure.barline is not None:
        _write_barline(node, measure.barline)


def build_tree(score: Score) -> ET.ElementTree:
    """Construieste arborele XML fara a-l scrie pe disc."""
    divisions = divisions_for(score)
    root = ET.Element("score-partwise", {"version": "4.0"})

    if score.title:
        _sub(_sub(root, "work"), "work-title", score.title)
    if score.composer:
        identification = _sub(root, "identification")
        creator = _sub(identification, "creator", score.composer)
        creator.set("type", "composer")

    # MusicXML cere id-uri proprii de parte; ale noastre ("S", "A", ...) devin
    # doar nume afisate, ca sa nu depindem de conventia lui.
    # Numerotare **pozitionala**, nu dupa `part.id`. Un dictionar cheiat pe id
    # pierde tacut o parte cand doua au acelasi id — si se intampla: pe pagina 3,
    # recunoasterea a dat de doua ori „X3", iar fisierul a iesit cu `P6` scris de
    # doua ori si fara `P3`, adica MusicXML invalid. Pozitia nu poate coliziona.
    part_ids = [f"P{index}" for index, _ in enumerate(score.parts, 1)]

    part_list = _sub(root, "part-list")
    for position, part in enumerate(score.parts):
        score_part = ET.SubElement(part_list, "score-part", {"id": part_ids[position]})
        _sub(score_part, "part-name", part.name or part.id)

    for position, part in enumerate(score.parts):
        node = ET.SubElement(root, "part", {"id": part_ids[position]})
        for index, measure in enumerate(part.measures):
            _write_measure(node, measure, divisions, first=index == 0)

    return ET.ElementTree(root)


def to_string(score: Score) -> str:
    """MusicXML complet, cu declaratie XML si DOCTYPE, indentat lizibil."""
    tree = build_tree(score)
    ET.indent(tree, space="  ")
    body = ET.tostring(tree.getroot(), encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{DOCTYPE}\n{body}\n'


def write(score: Score, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_string(score), encoding="utf-8")
    return path
