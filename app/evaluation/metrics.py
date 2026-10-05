"""Cat de bine seamana o transcriere cu partitura de referinta.

PROJECT.md §9 cere metrici separate, nu un „procent de acuratete" global:
**note corecte (pitch + durata)**, **masuri valide**, si fisiere care se
deschid in MuseScore. Modulul asta le calculeaza pe primele doua.

De ce sunt despartite inaltimea si ritmul: §12.4 a aratat ca slabiciunea reala
a lui `smt-grandstaff` e ritmul — 29% dintre masuri-voce nu se inchid la 4/4 —
in timp ce structura pe voci iese corecta. O singura cifra de acuratete ar
amesteca cele doua si ar ascunde exact lucrul pe care trebuie sa-l repare
fine-tuningul. Deci: o nota citita corect ca inaltime dar gresit ca durata se
numara altfel decat una citita gresit de tot.

„Masurile valide" nu se recalculeaza aici — regula exista deja, in
`app/validation/rules.py`, si se cheama de acolo pe **amandoua** partiturile.
Cifra de pe referinta nu e redundanta: daca referinta insasi are masuri care nu
se inchid, problema e in ground truth, si e mai bine sa se vada inainte de a
trage concluzii despre model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction

from app.evaluation.align import Op, OpKind, align, normalized_cost
from app.music_model.model import Clef, Event, Measure, Note, Part, Score
from app.validation.rules import Finding, check_measure_durations


class Category(str, Enum):
    """Cum a iesit o nota din referinta, dupa ce a fost pusa fata in fata."""

    MATCHED = "matched"
    PITCH_WRONG = "pitch_wrong"
    DURATION_WRONG = "duration_wrong"
    BOTH_WRONG = "both_wrong"


def pitch_key(event: Event) -> tuple[str, int, int] | None:
    """Inaltimea care conteaza la comparatie. `None` pentru pauza.

    `Pitch.accidental` — alteratia *scrisa* — e lasata deoparte intentionat:
    aceeasi nota poate fi notata cu sau fara alteratie de curtoazie, iar
    diferenta aceea e de tipografie, nu de muzica. `alter`, alteratia efectiva,
    intra insa in cheie: un fa-diez citit ca fa e o greseala adevarata.

    O pauza nu se potriveste niciodata cu o nota, si asta e voit: acolo unde
    referinta are o nota iar modelul a scris pauza, inaltimea e gresita.
    """
    if isinstance(event, Note):
        return (event.pitch.step, event.pitch.octave, event.pitch.alter)
    return None


def classify(ref: Event, hyp: Event) -> Category:
    """Categoria unei perechi aliniate."""
    same_pitch = pitch_key(ref) == pitch_key(hyp)
    # Se compara durata *sunata*, nu scrierea ei, iar `Fraction` face
    # egalitatea exacta — cu virgula mobila, o patrime cu punct ar fi putut
    # iesi diferita de ea insasi.
    same_duration = ref.quarter_length == hyp.quarter_length
    if same_pitch and same_duration:
        return Category.MATCHED
    if same_duration:
        return Category.PITCH_WRONG
    if same_pitch:
        return Category.DURATION_WRONG
    return Category.BOTH_WRONG


_COST = {
    Category.MATCHED: Fraction(0),
    Category.PITCH_WRONG: Fraction(1, 2),
    Category.DURATION_WRONG: Fraction(1, 2),
    Category.BOTH_WRONG: Fraction(1),
}


def event_cost(ref: Event, hyp: Event) -> Fraction:
    """Cost de substituire pentru aliniere: 0, 1/2 sau 1."""
    return _COST[classify(ref, hyp)]


def align_events(ref: list[Event], hyp: list[Event]) -> list[Op]:
    return align(ref, hyp, event_cost)


def _measure_cost_with_cache(cache: dict[tuple[int, int], list[Op]]):
    """Costul dintre doua masuri = cat de mult difera continuturile lor.

    Alinierea interioara se calculeaza o singura data pentru fiecare pereche si
    se pastreaza: programarea dinamica de la nivelul masurilor o cere in faza de
    umplere, iar raportul o cere din nou dupa ce s-a ales drumul.
    """

    def cost(ref: Measure, hyp: Measure) -> Fraction:
        key = (id(ref), id(hyp))
        ops = cache.get(key)
        if ops is None:
            ops = align_events(ref.events, hyp.events)
            cache[key] = ops
        return normalized_cost(ops, len(ref.events), len(hyp.events))

    return cost


@dataclass
class VoiceComparison:
    """Rezultatul pentru o singura voce."""

    part_id: str
    ref_events: int = 0
    hyp_events: int = 0
    #: Cate din evenimentele fiecarei parti sunt pauze. Se tin fiindca `note%`
    #: se raporteaza la **note adevarate**, nu la evenimente: pe pagina 3, din
    #: 182 de evenimente ale referintei 26 sunt pauze, iar o pagina de tacere
    #: care nimereste tacerea nu e o pagina citita (§15.42).
    ref_rests: int = 0
    hyp_rests: int = 0
    matched: int = 0
    #: Din `matched`, cate sunt **pauze** potrivite cu pauze. O pauza nu are
    #: inaltime, deci `pitch_key` da `None` de ambele parti si perechea iese
    #: „corecta" — dar o pagina de tacere care nimereste tacerea referintei nu
    #: e o pagina citita. Aceeasi capcana ca §15.28, mutata din SER in `corecte`.
    #: Se raporteaza, nu se scade: pauzele sunt in referinta (26 din 182 pe
    #: pagina 3), deci a le sterge din numitor ar fi tot o alegere, si nu a mea.
    matched_rests: int = 0
    pitch_wrong: int = 0
    duration_wrong: int = 0
    both_wrong: int = 0
    missing: int = 0
    extra: int = 0
    ref_measures: int = 0
    hyp_measures: int = 0
    measures_exact: int = 0
    measures_missing: int = 0
    measures_extra: int = 0
    context: list[str] = field(default_factory=list)

    @property
    def ref_notes(self) -> int:
        """Note adevarate in referinta — numitorul lui `note_accuracy`."""
        return self.ref_events - self.ref_rests

    @property
    def hyp_notes(self) -> int:
        """Note adevarate scrise de model — numitorul preciziei."""
        return self.hyp_events - self.hyp_rests

    @property
    def matched_notes(self) -> int:
        """Note adevarate potrivite — `matched` fara perechile de pauze."""
        return self.matched - self.matched_rests

    @property
    def substitutions(self) -> int:
        """Note prezente de ambele parti, dar diferite prin ceva."""
        return self.pitch_wrong + self.duration_wrong + self.both_wrong

    def _rate(self, correct: int) -> Fraction:
        """Raportat mereu la **referinta**: notele lipsa se numara ca gresite.

        O referinta goala e reprodusa perfect doar de o transcriere goala.
        """
        if self.ref_events == 0:
            return Fraction(1) if self.hyp_events == 0 else Fraction(0)
        return Fraction(correct, self.ref_events)

    @property
    def note_accuracy(self) -> Fraction:
        """Cat din **muzica** paginii s-a citit: note adevarate, nu evenimente.

        Numitorul e `ref_notes`, nu `ref_events`, si numaratorul `matched_notes`,
        nu `matched`. Motivul e in §15.42: doua pauze de aceeasi durata ies
        „identice" fiindca niciuna n-are inaltime, deci intrau amandoua in
        cifra. Pe pagina 3, 16 „corecte" erau 5 note si 11 pauze — 8,8% se
        citea ca acuratete si era, de fapt, 3,2%.

        Definitia unei note corecte ramane cea din §9: inaltime **si** durata.
        Ce s-a schimbat e ce se numara, nu cand se numara corect.
        """
        if self.ref_notes == 0:
            return Fraction(1) if self.hyp_notes == 0 else Fraction(0)
        return Fraction(self.matched_notes, self.ref_notes)

    @property
    def precision(self) -> Fraction:
        """Cat din ce **scrie** modelul e corect — perechea lui `note_accuracy`.

        Cele doua raspund la intrebari diferite, si abia impreuna descriu o
        transcriere: `note_accuracy` spune cat din pagina a fost citita,
        precizia spune cat din ce s-a scris nu trebuie sters. Un model amutit
        le desparte violent — 79% precizie cu 6% din pagina citita (§15.28).

        O transcriere goala are precizia 0, nu 1: n-a nimerit nimic. Exceptia e
        referinta goala, unde golul e raspunsul corect.

        Se numara **note adevarate** de ambele parti, ca sa fie perechea reala a
        lui `note_accuracy`: doua cifre pe numitori diferiti n-ar mai descrie
        aceeasi transcriere.
        """
        if self.hyp_notes == 0:
            return Fraction(1) if self.ref_notes == 0 else Fraction(0)
        return Fraction(self.matched_notes, self.hyp_notes)

    @property
    def pitch_accuracy(self) -> Fraction:
        """Note cu inaltimea corecta, indiferent de durata."""
        return self._rate(self.matched + self.duration_wrong)

    @property
    def rhythm_accuracy(self) -> Fraction:
        """Note cu durata corecta, indiferent de inaltime."""
        return self._rate(self.matched + self.pitch_wrong)

    @property
    def ser(self) -> Fraction:
        """Symbol Error Rate — (S+D+I)/N, metrica standard din literatura OMR.

        E pastrata tocmai pentru ca lucrarile SMT raporteaza tot SER: fara ea,
        cifrele noastre n-ar putea fi puse langa ale lor. Poate depasi 1, cand
        transcrierea inventeaza mai mult decat exista pe pagina.
        """
        if self.ref_events == 0:
            return Fraction(0) if self.hyp_events == 0 else Fraction(self.extra)
        return Fraction(self.substitutions + self.missing + self.extra, self.ref_events)

    @property
    def exact(self) -> bool:
        return (
            self.substitutions == 0
            and self.missing == 0
            and self.extra == 0
            and self.measures_missing == 0
            and self.measures_extra == 0
            and not self.context
        )


def _sum_voices(voices: list[VoiceComparison], part_id: str) -> VoiceComparison:
    total = VoiceComparison(part_id=part_id)
    for voice in voices:
        total.ref_events += voice.ref_events
        total.hyp_events += voice.hyp_events
        total.ref_rests += voice.ref_rests
        total.hyp_rests += voice.hyp_rests
        total.matched += voice.matched
        total.matched_rests += voice.matched_rests
        total.pitch_wrong += voice.pitch_wrong
        total.duration_wrong += voice.duration_wrong
        total.both_wrong += voice.both_wrong
        total.missing += voice.missing
        total.extra += voice.extra
        total.ref_measures += voice.ref_measures
        total.hyp_measures += voice.hyp_measures
        total.measures_exact += voice.measures_exact
        total.measures_missing += voice.measures_missing
        total.measures_extra += voice.measures_extra
        total.context.extend(voice.context)
    return total


@dataclass
class Signature:
    """*Cum* greseste modelul, nu doar cat.

    Cifra „11% note corecte" nu spune nimic despre ce trebuie reparat. Aceleasi
    aliniere din care ies procentele da si distributia greselilor, si acolo se
    vede tiparul: pe pagina 3, din 27 de durate gresite, **19 sunt patrime
    citita ca doime** — modelul citeste sistematic notele mai lungi decat sunt.
    Inaltimile, in schimb, sunt imprastiate, fara un decalaj dominant.

    Distinctia conteaza pentru ce urmeaza: un decalaj **constant** de inaltime ar
    fi un bug determinist (cheie sau octava citita gresit), pe cand unul
    imprastiat e ceva ce numai antrenarea repara.
    """

    durations: Counter = field(default_factory=Counter)
    pitches: Counter = field(default_factory=Counter)

    def add(self, ref: Event, hyp: Event) -> None:
        if ref.quarter_length != hyp.quarter_length:
            self.durations[(str(ref.quarter_length), str(hyp.quarter_length))] += 1
        if isinstance(ref, Note) and isinstance(hyp, Note):
            self.pitches[hyp.pitch.diatonic_index - ref.pitch.diatonic_index] += 1


@dataclass
class Comparison:
    """Rezultatul complet, cu tot ce trebuie ca sa poata fi si citit, si crezut."""

    voices: list[VoiceComparison] = field(default_factory=list)
    structure: list[str] = field(default_factory=list)
    ref_findings: list[Finding] = field(default_factory=list)
    hyp_findings: list[Finding] = field(default_factory=list)
    signature: Signature = field(default_factory=Signature)

    @property
    def total(self) -> VoiceComparison:
        return _sum_voices(self.voices, "TOTAL")

    @property
    def exact(self) -> bool:
        """Adevarat doar daca nu s-a gasit nicio diferenta, de niciun fel."""
        return not self.structure and all(voice.exact for voice in self.voices)


def clef_name(clef: Clef) -> str:
    """Numele cheii, cu tot cu schimbarea de octava.

    Fara sufix, cheia Sol de tenor si cheia Sol obisnuita se scriu amandoua
    „G2", iar raportul ar spune „referinta G2, transcriere G2" in dreptul unei
    nepotriviri reale — adica exact genul de rand care te trimite sa cauti in
    partea gresita.
    """
    suffix = {-1: "/8vb", 1: "/8va"}.get(clef.octave_change, "")
    return f"{clef.sign}{clef.line}{suffix}"


def _context_mismatches(ref: Part, hyp: Part) -> list[str]:
    """Cheia, armura si metrica in vigoare la inceputul vocii.

    Merita verificate separat, inainte de note: o armura citita gresit muta
    toate inaltimile de dupa ea. Fara randul asta, raportul ar arata zeci de
    note gresite si niciun indiciu ca toate au aceeasi cauza.
    """
    if not ref.measures or not hyp.measures:
        return []
    ref_clef, ref_key, ref_time = ref.context_at(ref.measures[0].number)
    hyp_clef, hyp_key, hyp_time = hyp.context_at(hyp.measures[0].number)
    mismatches = []
    if ref_clef != hyp_clef:
        mismatches.append(
            f"cheie: referinta {clef_name(ref_clef)}, transcriere {clef_name(hyp_clef)}"
        )
    if ref_key.fifths != hyp_key.fifths:
        mismatches.append(
            f"armura: referinta {ref_key.fifths}, transcriere {hyp_key.fifths}"
        )
    if (ref_time.beats, ref_time.beat_type) != (hyp_time.beats, hyp_time.beat_type):
        mismatches.append(f"metrica: referinta {ref_time}, transcriere {hyp_time}")
    return mismatches


def compare_parts(ref: Part, hyp: Part, signature: Signature | None = None) -> VoiceComparison:
    """Compara o voce cu perechea ei, pe doua niveluri: masuri, apoi note."""
    voice = VoiceComparison(
        part_id=ref.id,
        ref_events=sum(len(m.events) for m in ref.measures),
        hyp_events=sum(len(m.events) for m in hyp.measures),
        ref_rests=sum(1 for m in ref.measures for e in m.events if pitch_key(e) is None),
        hyp_rests=sum(1 for m in hyp.measures for e in m.events if pitch_key(e) is None),
        ref_measures=len(ref.measures),
        hyp_measures=len(hyp.measures),
        context=_context_mismatches(ref, hyp),
    )

    cache: dict[tuple[int, int], list[Op]] = {}
    measure_ops = align(ref.measures, hyp.measures, _measure_cost_with_cache(cache))

    for op in measure_ops:
        if op.kind is OpKind.MISSING:
            voice.measures_missing += 1
            voice.missing += len(ref.measures[op.ref].events)
            continue
        if op.kind is OpKind.EXTRA:
            voice.measures_extra += 1
            voice.extra += len(hyp.measures[op.hyp].events)
            continue

        ref_measure = ref.measures[op.ref]
        hyp_measure = hyp.measures[op.hyp]
        inner = cache[(id(ref_measure), id(hyp_measure))]
        for event_op in inner:
            if event_op.kind is OpKind.MISSING:
                voice.missing += 1
            elif event_op.kind is OpKind.EXTRA:
                voice.extra += 1
            else:
                pair = (
                    ref_measure.events[event_op.ref],
                    hyp_measure.events[event_op.hyp],
                )
                category = classify(*pair)
                setattr(voice, category.value, getattr(voice, category.value) + 1)
                if category is Category.MATCHED and pitch_key(pair[0]) is None:
                    voice.matched_rests += 1
                if signature is not None and category is not Category.MATCHED:
                    signature.add(*pair)
        if all(item.kind is OpKind.MATCH for item in inner):
            voice.measures_exact += 1

    return voice


def _rests(part: Part) -> int:
    """Cate evenimente ale vocii sunt pauze.

    Se numara si pentru vocile nepereche, altfel `ref_notes` si `hyp_notes` ies
    prea mari exact acolo unde modelul a inventat voci intregi de tacere — adica
    fix cazul pe care §15.42 il face vizibil.
    """
    return sum(1 for m in part.measures for e in m.events if pitch_key(e) is None)


def _all_missing(part: Part) -> VoiceComparison:
    """O voce pe care transcrierea nu a produs-o deloc.

    Se numara ca pierdere totala, nu se sare peste. A o omite ar umfla
    acuratetea exact in cazul cel mai grav — modelul care a citit trei voci din
    patru ar parea la fel de bun ca unul care le-a citit pe toate.
    """
    events = sum(len(m.events) for m in part.measures)
    return VoiceComparison(
        part_id=part.id,
        ref_events=events,
        ref_rests=_rests(part),
        missing=events,
        ref_measures=len(part.measures),
        measures_missing=len(part.measures),
    )


def _all_extra(part: Part) -> VoiceComparison:
    """O voce pe care transcrierea a inventat-o."""
    events = sum(len(m.events) for m in part.measures)
    return VoiceComparison(
        part_id=part.id,
        hyp_events=events,
        hyp_rests=_rests(part),
        extra=events,
        hyp_measures=len(part.measures),
        measures_extra=len(part.measures),
    )


def _pair_parts(
    reference: Score, hypothesis: Score, structure: list[str]
) -> list[tuple[Part | None, Part | None]]:
    """Imperecheaza vocile — dupa identitate daca se poate, altfel pozitional.

    Nu ghicim care voce e care. Daca cele doua partituri au exact aceleasi
    id-uri de parte, imperecherea dupa id e un fapt, nu o presupunere. Daca nu,
    imperechem pozitional si **spunem** ca am facut-o, la fel cum
    `parse_open_system` raporteaza neconcordanta de portative in loc s-o
    rezolve singur (§12.4).
    """
    ref_ids = [p.id for p in reference.parts]
    hyp_ids = [p.id for p in hypothesis.parts]

    if len(ref_ids) != len(hyp_ids):
        structure.append(
            f"numar de voci diferit: referinta {len(ref_ids)}, transcriere {len(hyp_ids)}"
        )

    if sorted(ref_ids) == sorted(hyp_ids):
        by_id = {p.id: p for p in hypothesis.parts}
        return [(part, by_id[part.id]) for part in reference.parts]

    if set(ref_ids) != set(hyp_ids):
        structure.append(
            f"id-uri de voce diferite: referinta {ref_ids}, transcriere {hyp_ids}"
            " — imperechere pozitionala"
        )

    pairs: list[tuple[Part | None, Part | None]] = []
    for index in range(max(len(ref_ids), len(hyp_ids))):
        ref_part = reference.parts[index] if index < len(reference.parts) else None
        hyp_part = hypothesis.parts[index] if index < len(hypothesis.parts) else None
        pairs.append((ref_part, hyp_part))
    return pairs


def compare(reference: Score, hypothesis: Score) -> Comparison:
    """Compara doua partituri. Referinta e adevarul; transcrierea e judecata."""
    result = Comparison(
        ref_findings=check_measure_durations(reference),
        hyp_findings=check_measure_durations(hypothesis),
    )

    for ref_part, hyp_part in _pair_parts(reference, hypothesis, result.structure):
        if ref_part is None:
            result.voices.append(_all_extra(hyp_part))
        elif hyp_part is None:
            result.voices.append(_all_missing(ref_part))
        else:
            result.voices.append(compare_parts(ref_part, hyp_part, result.signature))

    return result
