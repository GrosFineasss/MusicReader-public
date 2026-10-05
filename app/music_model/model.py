"""Reprezentarea muzicala interna.

Acest modul nu stie nimic despre imagini, modele AI sau MusicXML. Este logica
pura peste care se sprijina validarea si exportul.

Duratele sunt tinute exact, ca `Fraction`, niciodata ca `float`. Validatorul
verifica daca suma duratelor dintr-o masura este *exact* cat cere metrica, iar
0.1 + 0.2 != 0.3 ar transforma acea regula intr-o sursa de bug-uri.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Iterator

# --------------------------------------------------------------------------
# Inaltime
# --------------------------------------------------------------------------

STEPS = ("C", "D", "E", "F", "G", "A", "B")

# Semitonuri de la C, pentru fiecare treapta.
_STEP_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


class Accidental(str, Enum):
    """Alteratie *scrisa*, asa cum apare pe portativ."""

    SHARP = "sharp"
    FLAT = "flat"
    NATURAL = "natural"

    @property
    def alter(self) -> int:
        return {"sharp": 1, "flat": -1, "natural": 0}[self.value]


@dataclass(frozen=True, order=True)
class Pitch:
    """O inaltime notata: treapta + octava + alteratie efectiva.

    `alter` este alteratia *efectiva* (dupa aplicarea armurii si a alteratiilor
    din masura), nu neaparat cea scrisa langa nota. `accidental` retine ce s-a
    scris efectiv, pentru ca exportul MusicXML trebuie sa redea pagina, nu doar
    sunetul.
    """

    step: str
    octave: int
    alter: int = 0
    accidental: Accidental | None = None

    def __post_init__(self) -> None:
        if self.step not in STEPS:
            raise ValueError(f"treapta invalida: {self.step!r}")
        if not -2 <= self.alter <= 2:
            raise ValueError(f"alteratie invalida: {self.alter}")

    @property
    def midi(self) -> int:
        """Numar de nota MIDI. C4 = 60."""
        return (self.octave + 1) * 12 + _STEP_SEMITONES[self.step] + self.alter

    @property
    def diatonic_index(self) -> int:
        """Pozitia diatonica absoluta — cat de sus sta nota *pe portativ*.

        Ignora alteratiile intentionat: Fa si Fa-diez ocupa aceeasi linie.
        """
        return self.octave * 7 + STEPS.index(self.step)

    def __str__(self) -> str:
        mark = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}[self.alter]
        return f"{self.step}{mark}{self.octave}"


# --------------------------------------------------------------------------
# Durata
# --------------------------------------------------------------------------


class NoteType(str, Enum):
    """Valoarea de nota nepunctata."""

    WHOLE = "whole"
    HALF = "half"
    QUARTER = "quarter"
    EIGHTH = "eighth"
    SIXTEENTH = "16th"
    THIRTY_SECOND = "32nd"

    @property
    def quarter_length(self) -> Fraction:
        return {
            "whole": Fraction(4),
            "half": Fraction(2),
            "quarter": Fraction(1),
            "eighth": Fraction(1, 2),
            "16th": Fraction(1, 4),
            "32nd": Fraction(1, 8),
        }[self.value]


@dataclass(frozen=True)
class Tuplet:
    """Un grup neregulat: `actual` note scrise in timpul a `normal`.

    Trioletul e cazul obisnuit — 3 in timpul a 2. Se tine ca doua numere
    intregi, nu ca un raport deja simplificat, fiindca MusicXML cere exact
    perechea asta (`actual-notes` / `normal-notes`), iar 6:4 si 3:2 se scriu
    diferit pe pagina desi suna la fel.
    """

    actual: int
    normal: int

    def __post_init__(self) -> None:
        if self.actual < 1 or self.normal < 1:
            raise ValueError(f"grup neregulat invalid: {self.actual}:{self.normal}")

    @property
    def ratio(self) -> Fraction:
        """Cu cat se inmulteste durata *scrisa* ca sa iasa cea sunata."""
        return Fraction(self.normal, self.actual)

    def __str__(self) -> str:
        return f"{self.actual}:{self.normal}"


@dataclass(frozen=True)
class Duration:
    """Durata unei note sau pauze: valoare de baza + puncte + grup neregulat."""

    note_type: NoteType
    dots: int = 0
    tuplet: Tuplet | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.dots <= 3:
            raise ValueError(f"numar de puncte invalid: {self.dots}")

    @property
    def quarter_length(self) -> Fraction:
        """Durata in patrimi. Fiecare punct adauga jumatate din ce s-a acumulat.

        Grupul neregulat se aplica la sfarsit, ca inmultire exacta cu
        `normal/actual`. Cu `Fraction`, o patrime de triolet e *exact* o
        treime, deci regula „suma duratelor = cat cere metrica" ramane o
        egalitate, nu o aproximatie. Cu virgula mobila, trei treimi n-ar da
        niciodata fix o patrime, si fiecare masura cu triolete ar fi raportata
        gresit de validator.
        """
        written = self.note_type.quarter_length * (Fraction(2) - Fraction(1, 2**self.dots))
        return written if self.tuplet is None else written * self.tuplet.ratio

    def __str__(self) -> str:
        base = self.note_type.value + "." * self.dots
        return base if self.tuplet is None else f"{base} ({self.tuplet})"


# --------------------------------------------------------------------------
# Elemente de portativ
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Clef:
    sign: str = "G"  # G, F sau C
    line: int = 2
    octave_change: int = 0  # -1 pentru cheia Sol de tenor

    @staticmethod
    def treble() -> "Clef":
        return Clef("G", 2)

    @staticmethod
    def bass() -> "Clef":
        return Clef("F", 4)

    @staticmethod
    def tenor_vocal() -> "Clef":
        """Cheia Sol cu 8 dedesubt — standard pentru tenor in partiturile de cor."""
        return Clef("G", 2, octave_change=-1)


@dataclass(frozen=True)
class KeySignature:
    """Armura, ca numar de alteratii. Pozitiv = diezi, negativ = bemoli."""

    fifths: int = 0
    mode: str = "major"

    def __post_init__(self) -> None:
        if not -7 <= self.fifths <= 7:
            raise ValueError(f"armura invalida: {self.fifths}")

    @property
    def altered_steps(self) -> dict[str, int]:
        """Ce trepte sunt alterate de armura, si cu cat."""
        order = ("F", "C", "G", "D", "A", "E", "B")
        if self.fifths >= 0:
            return {s: 1 for s in order[: self.fifths]}
        return {s: -1 for s in reversed(order[7 + self.fifths :])}


@dataclass(frozen=True)
class TimeSignature:
    beats: int = 4
    beat_type: int = 4

    def __post_init__(self) -> None:
        if self.beats <= 0 or self.beat_type <= 0:
            raise ValueError(f"metrica invalida: {self.beats}/{self.beat_type}")

    @property
    def measure_quarter_length(self) -> Fraction:
        """Cate patrimi incape intr-o masura completa."""
        return Fraction(4 * self.beats, self.beat_type)

    def __str__(self) -> str:
        return f"{self.beats}/{self.beat_type}"


class BarlineStyle(str, Enum):
    REGULAR = "regular"
    DOUBLE = "light-light"
    FINAL = "light-heavy"


@dataclass(frozen=True)
class Barline:
    """Bara din dreapta masurii.

    `repeat` = bara **inchide** o repetitie (`:|`, punctele in stanga ei). Cea
    care deschide una (`|:`) e `Measure.repeat_start`, pe masura de dupa ea:
    asa o tine si MusicXML (`<barline location="left">`), si doar asa se poate
    scrie repetitia de la inceputul piesei, unde nu exista masura dinainte — cum
    incepe `pagina3`. Directia a devenit necesara pe 2026-10-04, cand paginile de
    antrenare au inceput sa poarte repetitiile pieselor (`app/training/bach.py`).
    """

    style: BarlineStyle = BarlineStyle.REGULAR
    repeat: bool = False


@dataclass
class Note:
    pitch: Pitch
    duration: Duration
    tie_start: bool = False  # incepe o legatura de durata spre nota urmatoare
    tie_stop: bool = False  # incheie o legatura inceputa anterior

    @property
    def quarter_length(self) -> Fraction:
        return self.duration.quarter_length


@dataclass
class Rest:
    duration: Duration

    @property
    def quarter_length(self) -> Fraction:
        return self.duration.quarter_length


Event = Note | Rest


def rests_for(quarter_length: Fraction | int) -> list[Rest]:
    """Pauze care umplu exact `quarter_length` patrimi, de la mare la mic.

    Folosita ori de cate ori trebuie *acoperita* o durata pentru care partitura
    nu scrie nimic: masurile completate ca vocile sa aiba aceeasi lungime
    (`app/recognition/page.py`) si golurile dinauntrul unei voci, pe care
    MuseScore le scrie ca `<forward>` in loc de pauze
    (`app/evaluation/reference.py`).

    O durata care nu se poate scrie exact — o treime de patrime, dintr-un
    triolet rupt — **refuza**, nu aproximeaza. O pauza cu o durata gresita ar
    decala tot ce urmeaza in voce, tacut.
    """
    rests: list[Rest] = []
    remaining = Fraction(quarter_length)
    if remaining < 0:
        raise ValueError(f"durata negativa: {remaining}")
    for note_type in (NoteType.WHOLE, NoteType.HALF, NoteType.QUARTER,
                      NoteType.EIGHTH, NoteType.SIXTEENTH, NoteType.THIRTY_SECOND):
        length = note_type.quarter_length
        while remaining >= length:
            rests.append(Rest(Duration(note_type)))
            remaining -= length
    if remaining:
        raise ValueError(
            f"raman {remaining} patrimi care nu se pot scrie ca pauze intregi"
        )
    return rests


# --------------------------------------------------------------------------
# Masura, parte, partitura
# --------------------------------------------------------------------------


@dataclass
class Measure:
    """O masura a unei singure voci.

    Atributele (cheie, armura, metrica) apar doar acolo unde se *schimba*;
    starea efectiva la un moment dat se obtine cu `Part.context_at`.
    """

    number: int
    events: list[Event] = field(default_factory=list)
    clef: Clef | None = None
    key: KeySignature | None = None
    time: TimeSignature | None = None
    barline: Barline | None = None
    # Masura e mai scurta decat cere metrica **din scriitura**, nu din greseala:
    # anacruza si perechea care o completeaza la bara de repetitie. Se pune doar
    # acolo unde se **citeste** din partitura (`app/evaluation/reference.py`) —
    # niciodata pe o transcriere, unde o masura scurta e o eroare pana la proba
    # contrarie.
    partial: bool = False
    # Masura incepe cu semnul de repetitie `|:` (bara groasa-subtire si doua
    # puncte). Perechea lui, `:|`, e `barline.repeat` pe masura care inchide.
    repeat_start: bool = False

    @property
    def quarter_length(self) -> Fraction:
        return sum((e.quarter_length for e in self.events), Fraction(0))

    def notes(self) -> Iterator[Note]:
        return (e for e in self.events if isinstance(e, Note))


@dataclass
class Part:
    """O voce. Intr-o partitura de cor: Sopran, Alto, Tenor sau Bas."""

    id: str
    name: str = ""
    measures: list[Measure] = field(default_factory=list)

    def context_at(self, measure_number: int) -> tuple[Clef, KeySignature, TimeSignature]:
        """Cheia, armura si metrica in vigoare la masura data.

        Se propaga inainte de la ultima schimbare; daca partea nu declara
        nimic, se folosesc valorile implicite (Sol, Do major, 4/4).
        """
        clef, key, time = Clef.treble(), KeySignature(), TimeSignature()
        for measure in self.measures:
            if measure.number > measure_number:
                break
            clef = measure.clef or clef
            key = measure.key or key
            time = measure.time or time
        return clef, key, time

    def measure(self, number: int) -> Measure:
        for m in self.measures:
            if m.number == number:
                return m
        raise KeyError(f"partea {self.id!r} nu are masura {number}")


@dataclass
class Score:
    """Partitura completa. Mai multe parti = sistem de cor."""

    parts: list[Part] = field(default_factory=list)
    title: str = ""
    composer: str = ""
    # Ce voci stau pe fiecare portativ, **de sus in jos**, ca pe pagina. Pentru
    # o partitura stransa corala: [["S", "A"], ["T", "B"]].
    #
    # Exista fiindca `parts` e o lista plata, iar asezarea pe portative nu se
    # poate deduce din ea: patru voci pot sta pe doua portative sau pe patru, si
    # o cantare cu solo are un portativ in plus, cu una sau doua voci pe el.
    # Fara campul asta, `to_kern_text` nu are de unde sti pe ce coloane sa
    # scrie, si presupunea SATB — adica refuza orice altceva.
    #
    # Goala inseamna „nu se stie", nu „nu are portative": partiturile construite
    # in teste sau venite pe alte drumuri raman valide, iar scriitorul cade
    # inapoi pe partitura stransa corala.
    staves: list[list[str]] = field(default_factory=list)

    def part(self, part_id: str) -> Part:
        for p in self.parts:
            if p.id == part_id:
                return p
        raise KeyError(f"nu exista partea {part_id!r}")

    @property
    def measure_count(self) -> int:
        return max((len(p.measures) for p in self.parts), default=0)

    def to_dict(self) -> dict:
        """Serializare pentru `data/annotations/`.

        Fractiile nu sunt JSON-abile, dar nici nu sunt stocate: duratele se
        salveaza ca (note_type, dots), din care se recalculeaza exact.
        """
        return asdict(self)


# --------------------------------------------------------------------------
# Ajutoare pentru muzica de cor
# --------------------------------------------------------------------------

SATB = (
    ("S", "Soprano", Clef.treble()),
    ("A", "Alto", Clef.treble()),
    ("T", "Tenor", Clef.tenor_vocal()),
    ("B", "Bass", Clef.bass()),
)


def empty_satb(
    measures: int,
    time: TimeSignature | None = None,
    key: KeySignature | None = None,
) -> Score:
    """Schelet SATB gol — util in teste si ca tinta pentru pasul de merge."""
    time = time or TimeSignature()
    key = key or KeySignature()
    score = Score(staves=[["S", "A"], ["T", "B"]])
    for part_id, name, clef in SATB:
        part = Part(id=part_id, name=name)
        for n in range(1, measures + 1):
            first = n == 1
            part.measures.append(
                Measure(
                    number=n,
                    clef=clef if first else None,
                    key=key if first else None,
                    time=time if first else None,
                )
            )
        score.parts.append(part)
    return score


def measures_slice(score: Score, first: int, last: int) -> Score:
    """Bucata de partitura dintre doua masuri, inclusiv, gata de scris singura.

    Folosita ca sa taie o referinta de pagina in **sisteme**, fiecare pereche cu
    decupajul lui din poza (§15.7). Un sistem trebuie sa se poata citi si scrie
    de sine statator, deci prima lui masura primeste **cheia, armura si metrica
    in vigoare acolo** — altfel un sistem care incepe la masura 6 s-ar citi in
    cheia Sol implicita, iar basul ar aparea cu o octava gresita.

    Numerele de masura se pastreaza: raportul trebuie sa poata spune „masura 8",
    nu „a treia masura a sistemului doi".
    """
    if first > last:
        raise ValueError(f"interval gol de masuri: {first}..{last}")

    cut = Score(
        title=score.title,
        composer=score.composer,
        staves=[list(group) for group in score.staves],
    )
    for part in score.parts:
        taken = Part(id=part.id, name=part.name)
        clef, key, time = part.context_at(first)
        for measure in part.measures:
            if not first <= measure.number <= last:
                continue
            opening = measure.number == first
            taken.measures.append(
                Measure(
                    number=measure.number,
                    events=list(measure.events),
                    clef=measure.clef or (clef if opening else None),
                    key=measure.key or (key if opening else None),
                    time=measure.time or (time if opening else None),
                    barline=measure.barline,
                    partial=measure.partial,
                    repeat_start=measure.repeat_start,
                )
            )
        cut.parts.append(taken)
    return cut


def drop_silent_staves(score: Score, keep: int) -> Score:
    """Bucata de partitura cu doar `keep` portative: scoate portativele care tac.

    Un sistem din manuscris nu deseneaza portativul de solo acolo unde solo-ul
    tace, iar MuseScore il ascunde la fel. Referinta il are insa pe toata piesa,
    deci tinta unui sistem de doua portative iesea cu trei — unul plin de pauze,
    pe care modelul l-ar fi invatat sa-l scrie fara sa-l vada (§15.9, „Doamne bun
    asculta", 2026-10-04). Se scot doar portative in care **toate** vocile au numai
    pauze, de sus in jos, cat sa ramana `keep`; daca nu ajung, e o nepotrivire
    adevarata intre decupaj si referinta, si se refuza.
    """
    if not score.staves or len(score.staves) == keep:
        return score
    if len(score.staves) < keep:
        # Decupajul arata mai multe portative decat are referinta: nepotrivire, nu „nimic de scos".
        raise ValueError(f"{keep} portative in decupaj, doar {len(score.staves)} in referinta")

    def silent(voices: list[str]) -> bool:
        return all(
            isinstance(event, Rest)
            for voice in voices
            for measure in score.part(voice).measures
            for event in measure.events
        )

    drop: set[int] = set()
    for index, voices in enumerate(score.staves):
        if len(score.staves) - len(drop) == keep:
            break
        if silent(voices):
            drop.add(index)
    if len(score.staves) - len(drop) != keep:
        raise ValueError(
            f"{len(score.staves)} portative in referinta, {keep} in decupaj, si doar "
            f"{len(drop)} tac pe tot sistemul"
        )
    gone = {voice for index in drop for voice in score.staves[index]}
    return Score(
        title=score.title,
        composer=score.composer,
        staves=[list(v) for i, v in enumerate(score.staves) if i not in drop],
        parts=[part for part in score.parts if part.id not in gone],
    )
