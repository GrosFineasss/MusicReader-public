"""Criteriul etapei 5: motorul de reguli prinde erori injectate deliberat.

Fiecare test strica partitura de referinta intr-un mod anume si verifica atat
ca greseala e prinsa, cat si ca e localizata corect. Un validator care spune
doar "e ceva in neregula" nu ne ajuta la corectia umana.
"""

import pytest

from app.music_model.model import (
    Accidental,
    Duration,
    KeySignature,
    Measure,
    Note,
    NoteType,
    Pitch,
    Rest,
)
from app.validation.rules import Severity, errors, validate
from scripts.make_reference_score import build

Q = Duration(NoteType.QUARTER)
H = Duration(NoteType.HALF)


@pytest.fixture
def score():
    return build()


def rules_fired(findings) -> set[str]:
    return {f.rule for f in findings}


class TestCleanScore:
    def test_reference_score_has_no_errors(self, score):
        found = validate(score)
        assert errors(found) == [], "\n".join(str(f) for f in found)


class TestMeasureDurations:
    def test_note_too_many_is_caught(self, score):
        score.part("S").measure(1).events.append(Note(Pitch("C", 5), Q))
        found = errors(validate(score))
        assert "durata-masura" in rules_fired(found)
        assert found[0].part_id == "S"
        assert found[0].measure == 1

    def test_missing_beat_is_caught(self, score):
        score.part("A").measure(2).events.pop()
        assert "durata-masura" in rules_fired(errors(validate(score)))

    def test_short_first_measure_is_only_a_warning(self, score):
        """Anacruza e in afara scopului v1, dar nu e o eroare de citire."""
        for part in score.parts:
            part.measures[0].events = [Note(Pitch("C", 4), Q)]
        found = validate(score)
        duration_findings = [f for f in found if f.rule == "durata-masura"]
        assert duration_findings
        assert all(f.severity is Severity.WARNING for f in duration_findings)


class TestSystemAlignment:
    def test_voice_with_a_missing_measure_is_caught(self, score):
        """Cazul tipic pe cor: un portativ din sistem a fost citit gresit."""
        score.part("T").measures.pop()
        found = errors(validate(score))
        assert "aliniere-sistem" in rules_fired(found)
        assert any(f.part_id == "T" for f in found if f.rule == "aliniere-sistem")

    def test_single_part_score_is_never_misaligned(self, score):
        score.parts = [score.part("S")]
        assert "aliniere-sistem" not in rules_fired(validate(score))


class TestMeasureNumbering:
    def test_gap_in_numbering_is_caught(self, score):
        score.part("B").measures[1].number = 5
        assert "numerotare-masuri" in rules_fired(errors(validate(score)))


class TestVoiceRanges:
    def test_octave_error_in_bass_is_caught(self, score):
        """Cea mai frecventa greseala de OMR: nota citita cu o octava alaturi."""
        measure = score.part("B").measure(1)
        measure.events[0] = Note(Pitch("C", 6), Duration(NoteType.WHOLE))
        found = validate(score)
        assert "ambitus" in rules_fired(found)

    def test_tenor_written_pitch_is_read_as_sounding_an_octave_lower(self, score):
        """G4 scris in cheia Sol de tenor suna G3 — in ambitus, deci fara warning."""
        found = [f for f in validate(score) if f.rule == "ambitus" and f.part_id == "T"]
        assert found == []


class TestAccidentals:
    def test_alter_inconsistent_with_key_is_caught(self, score):
        """Nota cu alter=1 fara diez scris si fara armura: partitura e incoerenta."""
        measure = score.part("S").measure(1)
        measure.events[0] = Note(Pitch("C", 5, 1), Q)
        assert "propagare-alteratii" in rules_fired(errors(validate(score)))

    def test_written_accidental_propagates_to_the_barline(self, score):
        measure = score.part("S").measure(1)
        measure.events[0] = Note(Pitch("C", 5, 1, Accidental.SHARP), Q)
        measure.events[1] = Note(Pitch("C", 5, 1), Q)
        assert "propagare-alteratii" not in rules_fired(errors(validate(score)))

    def test_accidental_does_not_leak_across_the_barline(self, score):
        score.part("S").measure(1).events[0] = Note(Pitch("D", 5, 1, Accidental.SHARP), Q)
        score.part("S").measure(2).events[0] = Note(Pitch("D", 5, 1), Q)
        found = errors(validate(score))
        assert any(f.rule == "propagare-alteratii" and f.measure == 2 for f in found)

    def test_key_signature_supplies_the_alteration(self, score):
        """In Fa major, un Si fara bemol scris e totusi Si bemol."""
        for part in score.parts:
            part.measures[0].key = KeySignature(-1)
        measure = score.part("A").measure(1)
        measure.events[0] = Note(Pitch("B", 4, -1), H)
        assert "propagare-alteratii" not in rules_fired(errors(validate(score)))


class TestTies:
    def test_unclosed_tie_is_caught(self, score):
        score.part("A").measure(2).events[1].tie_stop = False
        assert "legatura-durata" in rules_fired(errors(validate(score)))

    def test_tie_to_a_different_pitch_is_caught(self, score):
        score.part("A").measure(2).events[1] = Note(Pitch("G", 4), Q, tie_stop=True)
        found = errors(validate(score))
        assert any("continua pe" in f.message for f in found)

    def test_tie_stop_without_a_start_is_caught(self, score):
        score.part("S").measure(1).events[0].tie_stop = True
        assert "legatura-durata" in rules_fired(errors(validate(score)))

    def test_tie_running_into_a_rest_is_caught(self, score):
        score.part("T").measure(1).events[-1].tie_start = True
        assert "legatura-durata" in rules_fired(errors(validate(score)))


class TestEmptyMeasures:
    def test_empty_measure_is_caught(self, score):
        score.part("B").measure(2).events.clear()
        found = errors(validate(score))
        assert "masura-goala" in rules_fired(found)


class TestFindingFormat:
    def test_finding_names_its_location(self, score):
        score.part("S").measure(1).events.append(Rest(Q))
        text = str(errors(validate(score))[0])
        assert "S" in text and "m1" in text


# --------------------------------------------------------------------------
# Masuri scurte dinadins (§14, pagina 3: m1 are 1 timp, m9 are 3)
# --------------------------------------------------------------------------


def set_measure(score, number, durations, partial):
    """Rescrie masura `number` in toate vocile, cu pauzele date."""
    for part in score.parts:
        measure = part.measure(number)
        measure.events = [Rest(duration) for duration in durations]
        measure.partial = partial


class TestPartialMeasures:
    def test_a_short_measure_is_an_error_when_nothing_says_otherwise(self, score):
        set_measure(score, 2, [Q], partial=False)
        assert "durata-masura" in rules_fired(errors(validate(score)))

    def test_a_measure_marked_partial_is_not_an_error(self, score):
        """Marcajul vine din partitura, deci nu e o greseala de citit."""
        set_measure(score, 2, [Q], partial=True)
        assert "durata-masura" not in rules_fired(errors(validate(score)))

    def test_a_partial_measure_longer_than_the_meter_stays_an_error(self, score):
        """Prea lunga nu e niciodata intentionat, oricum ar fi marcata."""
        set_measure(score, 2, [H, H, Q], partial=True)
        assert "durata-masura" in rules_fired(errors(validate(score)))

    def test_two_partial_measures_that_complete_each_other_are_silent(self, score):
        """Anacruza plus masura de dinaintea repetitiei: 1 + 3 = 4."""
        set_measure(score, 1, [Q], partial=True)
        set_measure(score, 2, [H, Q], partial=True)
        assert "masura-scurta-neimperecheata" not in rules_fired(validate(score))

    def test_a_partial_measure_left_alone_is_reported(self, score):
        set_measure(score, 1, [Q], partial=True)
        found = [f for f in validate(score) if f.rule == "masura-scurta-neimperecheata"]
        assert {f.measure for f in found} == {1}
        assert {f.part_id for f in found} == {"S", "A", "T", "B"}
        assert all(f.severity is Severity.WARNING for f in found)

    def test_the_leftover_is_reported_where_it_opened_not_at_the_end(self, score):
        """Locul contat e masura care a ramas deschisa, nu ultima din voce."""
        set_measure(score, 1, [Q], partial=True)
        set_measure(score, 2, [H, Q], partial=True)
        score.part("S").measure(2).events = [Rest(Q)]
        found = [f for f in validate(score) if f.rule == "masura-scurta-neimperecheata"]
        assert [(f.part_id, f.measure) for f in found] == [("S", 1)]
