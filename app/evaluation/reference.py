"""MusicXML -> model intern, pentru partitura de referinta.

Referinta o scrii tu in MuseScore si o exporti ca MusicXML. Ca s-o putem
compara, trebuie citita — iar `app/musicxml/` are doar scriere.

**Se citeste cu `music21`, deliberat, nu cu un parser propriu.** Daca as scrie
si scriitorul, si cititorul, o greseala facuta simetric in amandoua ar trece
neobservata: harnessul ar raporta 100% pe o partitura gresita, si tocmai cifra
de incredere ar fi cea in care nu se poate avea incredere. `music21` e deja
folosit exact asa in `tests/test_musicxml.py` — ca a doua implementare, care ne
contrazice cand gresim. Importul e lenes, ca `align` si `metrics` sa ramana
fara dependinte.

Doua normalizari, amandoua necesare fiindca MuseScore exporta **partitura
stransa** — 2 portative cu cate doua voci — iar noi comparam pe patru parti
separate (§11.5):

- vocile de pe un portativ se desfac in parti de sine statatoare;
- cand o masura are o singura voce, continutul ei se scrie in **amandoua**
  vocile portativului. E regula de la `*v` din `app/recognition/kern.py`: o
  singura coloana inseamna ca vocile canta la unison, si daca am pune nota
  intr-una singura, cealalta ar aparea cu masuri lipsa care nu lipsesc.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from app.music_model.model import (
    SATB,
    Accidental,
    Barline,
    Duration,
    Event,
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
    rests_for,
)
from app.music_model.model import Clef as InternalClef


class UnsupportedNotation(ValueError):
    """Partitura foloseste notatie pe care v1 nu o reprezinta (§5.2).

    Se ridica in loc sa se aproximeze: o referinta citita aproximativ ar
    produce erori raportate in dreptul modelului, care n-are nicio vina.
    """


# `NoteType` acopera exact valorile pe care le scrie MusicXML pentru duratele
# din scop. Restul — breve, longa, 64th — n-ar putea fi reprezentat exact.
_NOTE_TYPES = {item.value for item in NoteType}

_ACCIDENTALS = {
    "sharp": Accidental.SHARP,
    "flat": Accidental.FLAT,
    "natural": Accidental.NATURAL,
}


def _tuplet_of(duration: Any, where: str) -> Tuplet | None:
    """Grupul neregulat, daca exista.

    Grupurile imbricate — un triolet inauntrul altuia — sunt refuzate, nu
    aproximate: durata lor n-ar iesi corecta, iar o referinta gresita ar
    raporta erori in dreptul modelului, care n-are nicio vina.
    """
    if not duration.tuplets:
        return None
    if len(duration.tuplets) > 1:
        raise UnsupportedNotation(f"{where}: grupuri neregulate imbricate")
    group = duration.tuplets[0]
    return Tuplet(int(group.numberNotesActual), int(group.numberNotesNormal))


def _duration_of(element: Any, where: str) -> Duration:
    duration = element.duration
    if duration.type not in _NOTE_TYPES:
        raise UnsupportedNotation(
            f"{where}: durata {duration.type!r} nu e reprezentabila in modelul intern"
        )
    return Duration(NoteType(duration.type), duration.dots, _tuplet_of(duration, where))


def _pitch_of(m21_pitch: Any) -> Pitch:
    alter = int(m21_pitch.alter)
    if alter != m21_pitch.alter:
        raise UnsupportedNotation(
            f"alteratie fractionara ({m21_pitch.alter}) — microtonalitatea nu e in scop"
        )
    accidental = None
    if m21_pitch.accidental is not None:
        # Dublul diez exista in `alter`, dar nu si in `Accidental`; pastram
        # sunetul si renuntam la semnul scris, care oricum nu intra in
        # comparatie (vezi `metrics.pitch_key`).
        accidental = _ACCIDENTALS.get(m21_pitch.accidental.name)
    return Pitch(m21_pitch.step, m21_pitch.octave, alter, accidental)


def _note_of(element: Any, pitch: Any, where: str) -> Note:
    tie_start = tie_stop = False
    if element.tie is not None:
        tie_start = element.tie.type in ("start", "continue")
        tie_stop = element.tie.type in ("stop", "continue")
    return Note(_pitch_of(pitch), _duration_of(element, where), tie_start, tie_stop)


def _events_of(container: Any, voice_index: int, length: Fraction, where: str) -> list[Event]:
    """Evenimentele unei voci, in ordinea in care se canta.

    Golurile se umplu cu pauze. MuseScore nu scrie pauze pentru timpii in care
    o voce tace: pune `<forward>`, care sare peste ei. `music21` il respecta —
    notele urmatoare primesc offsetul corect — dar `notesAndRests` nu intoarce
    nimic in dreptul lui, deci o citire naiva scurteaza masura. Masurat pe
    pagina 3: alto iesea cu 2 patrimi in m5, iar basul cu 7/2 in m4, din cauza
    unui gol de o optime la mijloc. Comparate asa, vocile acelea ar fi aratat
    masuri lipsa care nu lipsesc.

    `length` e lungimea *masurii* — 4 patrimi de obicei, 1 la anacruza — nu
    lungimea metricii. O anacruza nu se umple pana la masura intreaga.
    """
    events: list[Event] = []
    cursor = Fraction(0)

    def fill_to(position: Fraction) -> None:
        nonlocal cursor
        if position < cursor:
            raise UnsupportedNotation(
                f"{where}: doua evenimente ale aceleiasi voci se suprapun la {position}"
            )
        if position > cursor:
            try:
                events.extend(rests_for(position - cursor))
            except ValueError as error:
                raise UnsupportedNotation(f"{where}: {error}") from error
            cursor = position

    for element in container.notesAndRests:
        fill_to(Fraction(element.getOffsetBySite(container)))
        cursor += Fraction(element.duration.quarterLength)
        name = type(element).__name__
        if name == "Rest":
            events.append(Rest(_duration_of(element, where)))
        elif name == "Chord":
            # Intr-o partitura stransa scrisa corect, cele doua voci au codite
            # opuse si nu formeaza acorduri. Cand totusi apare unul, il impartim
            # de sus in jos: vocea de sus ia nota de sus. E o conventie, si e
            # spusa aici tocmai ca sa se stie ca a fost o alegere.
            pitches = sorted(element.pitches, key=lambda p: p.ps, reverse=True)
            index = min(voice_index, len(pitches) - 1)
            events.append(_note_of(element, pitches[index], where))
        else:
            events.append(_note_of(element, element.pitch, where))

    fill_to(length)
    return events


def _clef_of(m21_clef: Any) -> InternalClef | None:
    if m21_clef is None or m21_clef.sign is None:
        return None
    return InternalClef(m21_clef.sign, int(m21_clef.line or 0), int(m21_clef.octaveChange))


def _repeat(m21_barline: Any, direction: str) -> bool:
    """Bara poarta semnul de repetitie in directia data (`start` = `|:`, `end` = `:|`).

    Voltele (`RepeatBracket`) nu se citesc: paginile generate din aceste
    referinte nu le deseneaza, iar tinta trebuie sa descrie exact imaginea.
    """
    return getattr(m21_barline, "direction", None) == direction


def _voice_container(m21_measure: Any, voice_index: int) -> Any:
    """Din ce se citeste vocea `voice_index` a acestei masuri.

    O masura fara voci explicite se citeste de toate vocile portativului —
    regula unisonului descrisa in antetul modulului.
    """
    voices = list(m21_measure.voices)
    if not voices:
        return m21_measure
    by_id = {str(voice.id): voice for voice in voices}
    wanted = by_id.get(str(voice_index + 1))
    if wanted is not None:
        return wanted
    if len(voices) == 1:
        return voices[0]
    return voices[min(voice_index, len(voices) - 1)]


def _voice_count(m21_part: Any) -> int:
    """Cate voci poarta portativul — cel mai incarcat moment al lui decide."""
    return max(
        (len(m.voices) for m in m21_part.getElementsByClass("Measure")),
        default=1,
    ) or 1


def _part_ids(count: int) -> list[tuple[str, str]]:
    """Id si nume pentru vocile rezultate, de sus in jos.

    Patru voci inseamna SATB — e forma pe care o produce si conducta noastra
    (§11.5), deci comparatia se face pe id-uri identice, fara imperechere
    pozitionala. Orice alt numar primeste nume neutre: a boteza trei voci
    „S, A, T" ar fi o presupunere despre muzica, nu o constatare.
    """
    if count == 4:
        return [(part_id, name) for part_id, name, _ in SATB]
    return [(f"V{index}", f"Vocea {index}") for index in range(1, count + 1)]


def from_music21(
    m21_score: Any, trust_short_measures: bool = True, min_voices: int | Sequence[int] = 1
) -> Score:
    """Converteste o partitura `music21` deja parsata in modelul intern.

    `trust_short_measures` spune daca o masura mai scurta decat metrica a fost
    scrisa **dinadins**. Pe o referinta scrisa de om, da: anacruza si perechea
    ei de la bara de repetitie sunt notatie corecta, si ar fi nedrept sa le
    numaram ca masuri stricate. Pe o **transcriere**, nu: acolo o masura scurta
    e o greseala de citire pana la proba contrarie, iar MusicXML nu pastreaza
    diferenta dintre cele doua. Fara comutatorul asta, aceeasi transcriere
    raporta 79 de masuri neinchise citita din memorie si 9 citita de pe disc —
    adica o cifra care depinde de drumul pe care a venit, nu de muzica.

    `min_voices` spune cate voci are **cel putin** fiecare portativ, chiar daca
    fisierul nu le declara. Imnurile scrise pe pian pun adesea sopranul si
    altistul ca acorduri de doua note, fara voci: numarate dupa `<voice>`, ar
    iesi doua voci in loc de patru. Cu 2, acordul se imparte de sus in jos, ca
    mai jos. Implicit 1 — referintele se citesc ca pana acum. Poate fi si cate
    un numar pe portativ, de sus in jos: o pagina cu solo deasupra corului
    (`[1, 2, 2]`) are o singura voce pe portativul de solo, iar 2 acolo ar
    adauga o voce de pauze pe care imaginea n-o are.
    """
    m21_parts = list(m21_score.parts)
    if not m21_parts:
        raise UnsupportedNotation("partitura nu are niciun portativ")
    if isinstance(min_voices, int):
        minimums = [min_voices] * len(m21_parts)
    else:
        minimums = list(min_voices)
        if len(minimums) != len(m21_parts):
            raise ValueError(
                f"min_voices are {len(minimums)} valori, partitura are {len(m21_parts)} portative"
            )

    # Intai desfacem fiecare portativ in vocile lui, ca sa stim cate ies in
    # total; abia dupa aceea le putem boteza (patru = SATB).
    staves = [
        (m21_part, max(_voice_count(m21_part), minimum))
        for m21_part, minimum in zip(m21_parts, minimums)
    ]
    names = _part_ids(sum(count for _, count in staves))

    score = Score()
    position = 0
    for m21_part, count in staves:
        # Vocile portativului curent, ca sa se stie mai tarziu pe ce coloane se
        # scriu. `staves` merge de sus in jos, ca portativele pe pagina.
        score.staves.append([names[position + i][0] for i in range(count)])
        measures = list(m21_part.getElementsByClass("Measure"))
        for voice_index in range(count):
            part_id, part_name = names[position]
            position += 1
            part = Part(id=part_id, name=part_name)
            for number, m21_measure in enumerate(measures, 1):
                where = f"{part_id} m{number}"
                container = _voice_container(m21_measure, voice_index)
                length = Fraction(m21_measure.duration.quarterLength)
                part.measures.append(
                    Measure(
                        number=number,
                        events=_events_of(container, voice_index, length, where),
                        partial=trust_short_measures
                        and length != Fraction(m21_measure.barDuration.quarterLength),
                        repeat_start=_repeat(m21_measure.leftBarline, "start"),
                        barline=(
                            Barline(repeat=True)
                            if _repeat(m21_measure.rightBarline, "end")
                            else None
                        ),
                        clef=_clef_of(m21_measure.clef),
                        key=(
                            KeySignature(int(m21_measure.keySignature.sharps))
                            if m21_measure.keySignature is not None
                            else None
                        ),
                        time=(
                            TimeSignature(
                                m21_measure.timeSignature.numerator,
                                m21_measure.timeSignature.denominator,
                            )
                            if m21_measure.timeSignature is not None
                            else None
                        ),
                    )
                )
            score.parts.append(part)

    metadata = m21_score.metadata
    if metadata is not None:
        score.title = metadata.title or ""
        score.composer = metadata.composer or ""
    return score


def system_starts(m21_score: Any) -> list[int]:
    """La ce masuri incepe un sistem nou pe pagina.

    MuseScore scrie rupturile de rand in MusicXML (`<print new-system="yes">`),
    iar `music21` le pastreaza ca `SystemLayout`. Pe pagina 3 ies masurile 6 si
    12 — exact rupturile manuscrisului, fiindca asa a fost asezata partitura.

    De asta atarna intreaga conducta de date de antrenare: fara ele, fiecare
    sistem decupat din poza ar trebui imperecheat manual cu masurile lui.

    **O ruptura de pagina porneste si un sistem nou**, dar MuseScore o scrie
    doar ca `<print new-page="yes">`, fara `new-system` alaturi. Numarand numai
    `SystemLayout`, referinta de doua pagini a piesei „Doamne bun asculta"
    dadea 9 sisteme in loc de 10, si primul sistem al paginii a doua se lipea
    de ultimul sistem al paginii intai — adica perechea (decupaj, tinta) ar fi
    invatat modelul alte note decat cele din imagine. Pe pagina 3, care e o
    singura pagina, defectul n-avea cum sa se vada.

    **Masurile se numara dupa pozitie (1..n), ca in `from_music21`**, nu dupa
    numarul scris in fisier: MuseScore numeste anacruza 0 si sare peste numar
    la masurile scurte, iar pe „Doamne bun asculta" numarul scris ramane cu doua
    in urma pozitiei — fiecare sistem si-ar fi luat masurile altui loc.
    """
    starts = [1]
    for m21_part in m21_score.parts:
        for position, measure in enumerate(m21_part.getElementsByClass("Measure"), 1):
            layouts = list(measure.getElementsByClass("SystemLayout")) + list(
                measure.getElementsByClass("PageLayout")
            )
            if any(layout.isNew for layout in layouts) and position not in starts:
                starts.append(position)
        break
    return sorted(starts)


def page_starts(m21_score: Any) -> list[int]:
    """La ce masuri incepe o pagina noua.

    Perechea lui `system_starts`, si necesara din acelasi motiv: o referinta
    poate acoperi mai multe pagini, pe cand poza data lui `dataset.py` e a
    **unei singure pagini**. Fara asta, o piesa de doua pagini nu poate produce
    perechi de antrenare deloc — numarul de sisteme din poza nu va fi niciodata
    cel al intregii referinte. Numaratoarea e dupa pozitie, ca la `system_starts`.
    """
    starts = [1]
    for m21_part in m21_score.parts:
        for position, measure in enumerate(m21_part.getElementsByClass("Measure"), 1):
            layouts = measure.getElementsByClass("PageLayout")
            if any(layout.isNew for layout in layouts) and position not in starts:
                starts.append(position)
        break
    return sorted(starts)


def load_with_layout(
    path: str | Path, trust_short_measures: bool = True, min_voices: int | Sequence[int] = 1
) -> tuple[Score, list[int], list[int]]:
    """Partitura, rupturile ei de sistem si de pagina, dintr-o singura citire.

    Amandoua listele sunt numere de masura la care incepe ceva nou. Cele de
    pagina sunt intotdeauna si inceputuri de sistem, deci se regasesc si in
    prima lista. `min_voices` merge mai departe la `from_music21`.
    """
    import music21  # lazy: vezi antetul modulului

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"nu exista fisierul {path}")
    parsed = music21.converter.parse(str(path))
    return (
        from_music21(parsed, trust_short_measures, min_voices),
        system_starts(parsed),
        page_starts(parsed),
    )


def load(path: str | Path, trust_short_measures: bool = True) -> Score:
    """Citeste un fisier MusicXML (`.musicxml`, `.xml` sau `.mxl`).

    Numerotarea masurilor se normalizeaza la 1..n. Numerele originale n-ar
    ajuta: comparatia se face prin aliniere, tocmai ca sa nu depinda de ele.
    """
    import music21  # lazy: vezi antetul modulului

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"nu exista fisierul {path}")
    return from_music21(music21.converter.parse(str(path)), trust_short_measures)
