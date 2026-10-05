"""Motor de validare muzicala — cod determinist, fara AI.

Principiul din PROJECT.md: modelul nu e singura sursa de adevar. Validatorul
nu incearca sa *repare* nimic; spune **unde** e problema, ca sa putem cere o
corectie umana si ca sa stim pe ce sa ne uitam cand modelul greseste.

Nu importa nimic din `recognition/`. Primeste un `Score` si intoarce
constatari, indiferent daca partitura a venit dintr-o poza sau a fost scrisa
manual in cod.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction


from app.music_model.model import (
    Clef,
    KeySignature,
    Measure,
    Note,

    Pitch,
    Score,
)


class Severity(str, Enum):
    ERROR = "error"      # partitura e sigur gresita
    WARNING = "warning"  # suspect, dar poate fi intentionat


@dataclass(frozen=True)
class Finding:
    """O constatare, localizata cat de precis se poate."""

    severity: Severity
    rule: str
    message: str
    part_id: str | None = None
    measure: int | None = None
    event_index: int | None = None

    def __str__(self) -> str:
        where = []
        if self.part_id:
            where.append(self.part_id)
        if self.measure is not None:
            where.append(f"m{self.measure}")
        if self.event_index is not None:
            where.append(f"#{self.event_index}")
        location = ".".join(where) or "partitura"
        return f"[{self.severity.value}] {location}: {self.message} ({self.rule})"


# --------------------------------------------------------------------------
# Ambitusul vocilor, in inaltimi *sunate* (MIDI). Larg intentionat: scopul e
# sa prinda o nota citita cu o octava gresit, nu sa judece scriitura corala.
# --------------------------------------------------------------------------

VOICE_RANGES: dict[str, tuple[int, int]] = {
    "S": (60, 81),  # C4  - A5
    "A": (53, 77),  # F3  - F5
    "T": (48, 69),  # C3  - A4
    "B": (40, 64),  # E2  - E4
}


def sounding_midi(pitch: Pitch, clef: Clef) -> int:
    """Inaltimea reala. Cheia Sol de tenor scrie cu o octava mai sus decat suna."""
    return pitch.midi + 12 * clef.octave_change


# --------------------------------------------------------------------------
# Reguli
# --------------------------------------------------------------------------


def check_measure_durations(score: Score) -> list[Finding]:
    """Suma duratelor dintr-o masura = ce cere metrica.

    Prima masura are voie sa fie mai scurta (anacruza) — o semnalam ca warning,
    nu ca eroare, chiar daca v1 nu suporta anacruze.

    O masura marcata `partial` si mai scurta decat metrica nu e nici macar
    semnalata: e scrisa asa dinadins, iar marcajul vine din partitura, nu din
    ghicit (vezi `Measure.partial`). Mai *lunga* decat metrica ramane insa
    eroare oricum ar fi marcata — asta nu e niciodata intentionat.
    """
    findings: list[Finding] = []
    for part in score.parts:
        for measure in part.measures:
            _, _, time = part.context_at(measure.number)
            expected = time.measure_quarter_length
            actual = measure.quarter_length
            if actual == expected:
                continue
            if measure.partial and actual < expected:
                continue
            anacrusis = measure is part.measures[0] and actual < expected
            findings.append(
                Finding(
                    Severity.WARNING if anacrusis else Severity.ERROR,
                    "durata-masura",
                    f"masura are {actual} patrimi, metrica {time} cere {expected}",
                    part.id,
                    measure.number,
                )
            )
    return findings


def check_partial_measures_pair_up(score: Score) -> list[Finding]:
    """Masurile scurte dinadins se completeaza intre ele.

    Regula pe care o descrie orice partitura cu repetitie: anacruza de la
    inceput plus masura de dinaintea barei de repetitie fac impreuna o masura
    intreaga. Pe pagina 3: m1 are 1 timp, m9 are 3.

    Verificarea e cea mai buna dovada ca marcajul `partial` a fost citit corect.
    Daca ramane o masura scurta neimperecheata, ori partitura chiar e asa — se
    intampla pe manuscris — ori am pierdut ceva la citire. In ambele cazuri
    merita spus, si merita spus ca **avertisment**: nu putem sti care din doua.
    """
    findings: list[Finding] = []
    for part in score.parts:
        pending = Fraction(0)
        opened: Measure | None = None
        for measure in part.measures:
            if not measure.partial:
                continue
            _, _, time = part.context_at(measure.number)
            if opened is None:
                opened = measure
            pending += measure.quarter_length
            while pending >= time.measure_quarter_length:
                pending -= time.measure_quarter_length
                opened = None
            if pending == 0:
                opened = None
        if pending and opened is not None:
            findings.append(
                Finding(
                    Severity.WARNING,
                    "masura-scurta-neimperecheata",
                    f"masura scurta de {pending} patrimi nu e completata de alta",
                    part.id,
                    opened.number,
                )
            )
    return findings


def check_measure_numbering(score: Score) -> list[Finding]:
    """Numerele de masura cresc cu 1, fara sarituri si fara duplicate."""
    findings: list[Finding] = []
    for part in score.parts:
        expected = None
        for measure in part.measures:
            if expected is not None and measure.number != expected:
                findings.append(
                    Finding(
                        Severity.ERROR,
                        "numerotare-masuri",
                        f"dupa masura {expected - 1} urmeaza {measure.number}",
                        part.id,
                        measure.number,
                    )
                )
            expected = measure.number + 1
    return findings


def check_system_alignment(score: Score) -> list[Finding]:
    """Toate vocile unui sistem coral au acelasi numar de masuri.

    Regula cea mai utila pe muzica de cor: daca o voce iese cu 11 masuri si
    restul cu 12, aproape sigur un portativ a fost citit gresit.
    """
    if len(score.parts) < 2:
        return []
    counts = {part.id: len(part.measures) for part in score.parts}
    reference = max(set(counts.values()), key=list(counts.values()).count)
    return [
        Finding(
            Severity.ERROR,
            "aliniere-sistem",
            f"vocea are {count} masuri, restul sistemului are {reference}",
            part_id,
        )
        for part_id, count in counts.items()
        if count != reference
    ]


def check_voice_ranges(score: Score) -> list[Finding]:
    """Fiecare nota cade in ambitusul vocii ei."""
    findings: list[Finding] = []
    for part in score.parts:
        bounds = VOICE_RANGES.get(part.id)
        if bounds is None:
            continue
        low, high = bounds
        for measure in part.measures:
            clef, _, _ = part.context_at(measure.number)
            for index, event in enumerate(measure.events):
                if not isinstance(event, Note):
                    continue
                midi = sounding_midi(event.pitch, clef)
                if low <= midi <= high:
                    continue
                direction = "sub" if midi < low else "peste"
                findings.append(
                    Finding(
                        Severity.WARNING,
                        "ambitus",
                        f"{event.pitch} suna {direction} ambitusul vocii {part.id}",
                        part.id,
                        measure.number,
                        index,
                    )
                )
    return findings




def check_accidental_propagation(score: Score) -> list[Finding]:
    """Alteratia scrisa tine pana la bara de masura, pe aceeasi treapta si octava.

    Verificam ca `alter`-ul efectiv al fiecarei note corespunde cu ce ar trebui
    sa fie dupa armura si alteratiile deja scrise in masura. O nepotrivire
    inseamna ca recunoasterea a produs o partitura incoerenta cu ea insasi.
    """
    findings: list[Finding] = []
    for part in score.parts:
        for measure in part.measures:
            _, key, _ = part.context_at(measure.number)
            from_key = key.altered_steps
            state: dict[tuple[str, int], int] = {}
            for index, event in enumerate(measure.events):
                if not isinstance(event, Note):
                    continue
                pitch = event.pitch
                slot = (pitch.step, pitch.octave)
                if pitch.accidental is not None:
                    state[slot] = pitch.accidental.alter
                expected = state.get(slot, from_key.get(pitch.step, 0))
                if pitch.alter != expected:
                    findings.append(
                        Finding(
                            Severity.ERROR,
                            "propagare-alteratii",
                            f"{pitch.step}{pitch.octave} are alter={pitch.alter}, "
                            f"dar armura si masura cer {expected}",
                            part.id,
                            measure.number,
                            index,
                        )
                    )
    return findings


def check_ties(score: Score) -> list[Finding]:
    """Fiecare legatura de durata inceputa se si inchide, pe aceeasi inaltime."""
    findings: list[Finding] = []
    for part in score.parts:
        pending: tuple[Pitch, int, int] | None = None
        for measure in part.measures:
            for index, event in enumerate(measure.events):
                if not isinstance(event, Note):
                    if pending is not None:
                        pitch, m, i = pending
                        findings.append(
                            Finding(
                                Severity.ERROR,
                                "legatura-durata",
                                f"legatura pe {pitch} nu se inchide (urmeaza o pauza)",
                                part.id,
                                m,
                                i,
                            )
                        )
                        pending = None
                    continue
                if pending is not None:
                    pitch, m, i = pending
                    if not event.tie_stop:
                        findings.append(
                            Finding(
                                Severity.ERROR,
                                "legatura-durata",
                                f"legatura pe {pitch} nu e preluata de nota urmatoare",
                                part.id,
                                m,
                                i,
                            )
                        )
                    elif event.pitch.midi != pitch.midi:
                        findings.append(
                            Finding(
                                Severity.ERROR,
                                "legatura-durata",
                                f"legatura de la {pitch} continua pe {event.pitch}",
                                part.id,
                                measure.number,
                                index,
                            )
                        )
                    pending = None
                elif event.tie_stop:
                    findings.append(
                        Finding(
                            Severity.ERROR,
                            "legatura-durata",
                            f"{event.pitch} inchide o legatura care nu a inceput",
                            part.id,
                            measure.number,
                            index,
                        )
                    )
                if event.tie_start:
                    pending = (event.pitch, measure.number, index)
        if pending is not None:
            pitch, m, i = pending
            findings.append(
                Finding(
                    Severity.ERROR,
                    "legatura-durata",
                    f"legatura pe {pitch} ramane deschisa la finalul vocii",
                    part.id,
                    m,
                    i,
                )
            )
    return findings


def check_empty_measures(score: Score) -> list[Finding]:
    """O masura fara niciun eveniment e aproape sigur un portativ citit gresit."""
    return [
        Finding(Severity.ERROR, "masura-goala", "masura nu contine nicio nota sau pauza",
                part.id, measure.number)
        for part in score.parts
        for measure in part.measures
        if not measure.events
    ]


RULES = (
    check_measure_durations,
    check_partial_measures_pair_up,
    check_measure_numbering,
    check_system_alignment,
    check_voice_ranges,
    check_accidental_propagation,
    check_ties,
    check_empty_measures,
)


def validate(score: Score) -> list[Finding]:
    """Ruleaza toate regulile. Erorile intai, apoi avertismentele."""
    findings: list[Finding] = []
    for rule in RULES:
        findings.extend(rule(score))
    findings.sort(key=lambda f: (f.severity != Severity.ERROR, f.part_id or "", f.measure or 0))
    return findings


def errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity is Severity.ERROR]
