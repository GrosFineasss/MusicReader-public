"""Construieste manual o partitura SATB de referinta si o exporta ca MusicXML.

Etapa 4 din roadmap: dovedim ca partea determinista produce MusicXML valid
*inainte* sa depindem de AI. Daca fisierul asta nu se deschide in MuseScore,
nimic din ce vine dupa nu conteaza.

Fragmentul e o cadenta simpla in Do major, 4/4, doua masuri, patru voci.
Nu are pretentii muzicale — are pretentii de corectitudine.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.music_model.model import (  # noqa: E402
    SATB,
    Accidental,
    Barline,
    BarlineStyle,
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
)
from app.musicxml import export  # noqa: E402

Q = Duration(NoteType.QUARTER)
H = Duration(NoteType.HALF)
H_DOT = Duration(NoteType.HALF, 1)
E = Duration(NoteType.EIGHTH)
W = Duration(NoteType.WHOLE)


def n(step: str, octave: int, duration: Duration, **kw) -> Note:
    return Note(Pitch(step, octave), duration, **kw)


# Fiecare voce: (masura 1, masura 2). Sumele trebuie sa dea 4 patrimi.
VOICES = {
    "S": (
        [n("C", 5, Q), n("D", 5, Q), n("E", 5, H)],
        [n("D", 5, Q), n("D", 5, Q), n("C", 5, H)],
    ),
    "A": (
        [n("G", 4, H), n("G", 4, Q), n("G", 4, Q)],
        [n("F", 4, Q, tie_start=True), n("F", 4, Q, tie_stop=True), n("E", 4, H)],
    ),
    "T": (
        [n("E", 4, Q), n("E", 4, E), n("F", 4, E), n("G", 4, H)],
        [Rest(Q), n("A", 4, Q), n("G", 4, H)],
    ),
    "B": (
        [n("C", 3, W)],
        [n("G", 2, H_DOT), n("C", 3, Q)],
    ),
}


def build() -> Score:
    """Construieste o partitura noua la fiecare apel.

    `deepcopy` nu e un detaliu: `VOICES` tine obiecte `Note` create o singura
    data, la import. O copie superficiala ar da doua partituri care impart
    aceleasi note, iar un test care strica o legatura de durata ar strica-o
    pentru toate testele de dupa el.
    """
    score = Score(title="Cadenta de referinta", composer="test determinist")
    for part_id, name, clef in SATB:
        part = Part(id=part_id, name=name)
        for number, events in enumerate(VOICES[part_id], 1):
            last = number == len(VOICES[part_id])
            part.measures.append(
                Measure(
                    number=number,
                    events=deepcopy(list(events)),
                    clef=clef if number == 1 else None,
                    key=KeySignature(0) if number == 1 else None,
                    time=TimeSignature(4, 4) if number == 1 else None,
                    barline=Barline(BarlineStyle.FINAL) if last else None,
                )
            )
        score.parts.append(part)
    return score


def main() -> int:
    score = build()
    out = Path(__file__).resolve().parents[1] / "data" / "validation" / "reference_satb.musicxml"
    export.write(score, out)

    print(f"divisions = {export.divisions_for(score)}")
    for part in score.parts:
        lengths = [str(m.quarter_length) for m in part.measures]
        print(f"  {part.id} ({part.name}): masuri de {', '.join(lengths)} patrimi")
    print(f"scris: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
