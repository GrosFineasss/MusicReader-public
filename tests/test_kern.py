"""Conversia `**kern` -> model intern. Logica pura, fara model si fara imagini.

Exemplele de mai jos sunt scrise in stilul in care scoate efectiv modelul SMT
(inclusiv coloana `**mxhm` de acorduri, pe care trebuie s-o ignoram).
"""

from fractions import Fraction

import pytest

from app.music_model.model import (
    Accidental,
    Clef,
    KeySignature,
    Note,
    NoteType,
    Rest,
    TimeSignature,
)
from app.recognition.kern import (
    KernError,
    merge_voices,
    parse_duration,
    parse_key,
    parse_pitch,
    parse_spine,
)

SOPRANO = """**kern\t**mxhm
*clefG2\t*
*k[f#]\t*
*M4/4\t*
=\t=
4g\t.
4a\t.
2b\t.
=\t=
[2cc\t.
2cc]\t.
==\t=
*-\t*-
"""


class TestPitch:
    @pytest.mark.parametrize(
        "letters,expected",
        [("c", "C4"), ("cc", "C5"), ("ccc", "C6"), ("C", "C3"), ("CC", "C2"), ("CCC", "C1")],
    )
    def test_octave_convention(self, letters, expected):
        assert str(parse_pitch(letters, "")) == expected

    def test_bass_range_that_jazzmus_cannot_express(self):
        """Mi2 si Sol2 — note obisnuite de bas coral."""
        assert parse_pitch("EE", "").midi == 40
        assert parse_pitch("GG", "").midi == 43

    def test_sharp_and_flat(self):
        assert parse_pitch("f", "#").alter == 1
        assert parse_pitch("b", "-").alter == -1
        assert parse_pitch("b", "--").alter == -2

    def test_written_accidental_is_kept(self):
        assert parse_pitch("f", "#").accidental is Accidental.SHARP
        assert parse_pitch("b", "n").accidental is Accidental.NATURAL
        assert parse_pitch("g", "").accidental is None

    def test_inconsistent_letters_are_rejected(self):
        with pytest.raises(KernError):
            parse_pitch("cd", "")


class TestDuration:
    @pytest.mark.parametrize(
        "digits,dots,expected",
        [("4", "", Fraction(1)), ("2", "", Fraction(2)), ("1", "", Fraction(4)),
         ("8", "", Fraction(1, 2)), ("2", ".", Fraction(3)), ("4", "..", Fraction(7, 4))],
    )
    def test_quarter_length(self, digits, dots, expected):
        assert parse_duration(digits, dots).quarter_length == expected

    @pytest.mark.parametrize(
        "digits,expected",
        # Durata Humdrum e reciproca, deci sunatul trebuie sa iasa mereu 4/valoare.
        [("12", Fraction(1, 3)), ("6", Fraction(2, 3)), ("3", Fraction(4, 3)),
         ("24", Fraction(1, 6)), ("20", Fraction(1, 5))],
    )
    def test_grupurile_neregulate_dau_durata_exacta(self, digits, expected):
        assert parse_duration(digits, "").quarter_length == expected

    def test_trioletul_e_scris_ca_optime_in_grup_de_trei(self):
        # Ce se *vede* pe pagina e o optime; grupul spune ca trei tin cat doua.
        duration = parse_duration("12", "")
        assert duration.note_type is NoteType.EIGHTH
        assert (duration.tuplet.actual, duration.tuplet.normal) == (3, 2)

    def test_trei_optimi_de_triolet_fac_exact_o_patrime(self):
        # Egalitatea asta e motivul pentru care duratele sunt `Fraction`: cu
        # virgula mobila, validatorul ar raporta gresit fiecare masura cu triolete.
        assert sum(parse_duration("12", "").quarter_length for _ in range(3)) == Fraction(1)

    def test_o_durata_nereprezentabila_tot_e_refuzata(self):
        with pytest.raises(KernError):
            parse_duration("64", "")  # sub 32-ime, in afara scopului v1


class TestKeySignature:
    @pytest.mark.parametrize(
        "content,fifths",
        [("", 0), ("f#", 1), ("f#c#", 2), ("f#c#g#", 3), ("b-", -1), ("b-e-a-", -3)],
    )
    def test_parses_both_directions(self, content, fifths):
        assert parse_key(content).fifths == fifths

    def test_mixed_signature_is_refused(self):
        with pytest.raises(KernError):
            parse_key("f#b-")


class TestClef:
    def test_treble(self):
        result = parse_spine("**kern\n*clefG2\n=\n4c\n")
        assert result.part.measures[0].clef == Clef.treble()

    def test_bass(self):
        result = parse_spine("**kern\n*clefF4\n=\n4C\n")
        assert result.part.measures[0].clef == Clef.bass()

    def test_tenor_octave_transposing(self):
        """`*clefGv2` = cheia Sol cu 8 dedesubt, cea a tenorului de cor."""
        result = parse_spine("**kern\n*clefGv2\n=\n4c\n")
        assert result.part.measures[0].clef == Clef.tenor_vocal()


class TestSpine:
    @pytest.fixture
    def parsed(self):
        return parse_spine(SOPRANO, part_id="S", name="Soprano")

    def test_nothing_is_silently_dropped(self, parsed):
        assert parsed.unrecognised == []

    def test_leading_barline_does_not_create_an_empty_measure(self, parsed):
        assert len(parsed.part.measures) == 2
        assert parsed.part.measures[0].events

    def test_attributes_land_on_the_first_measure(self, parsed):
        first = parsed.part.measures[0]
        assert first.clef == Clef.treble()
        assert first.key == KeySignature(1)
        assert first.time == TimeSignature(4, 4)

    def test_measures_are_complete(self, parsed):
        assert [m.quarter_length for m in parsed.part.measures] == [Fraction(4), Fraction(4)]

    def test_notes_are_read_in_order(self, parsed):
        pitches = [str(e.pitch) for e in parsed.part.measures[0].events]
        assert pitches == ["G4", "A4", "B4"]

    def test_ties_are_paired(self, parsed):
        first, second = parsed.part.measures[1].events
        assert (first.tie_start, first.tie_stop) == (True, False)
        assert (second.tie_start, second.tie_stop) == (False, True)

    def test_chord_column_is_ignored(self, parsed):
        """Coloana `**mxhm` contine acorduri de jazz. Nu ne intereseaza."""
        assert all(isinstance(e, Note) for m in parsed.part.measures for e in m.events)

    def test_final_barline_is_recorded(self, parsed):
        assert parsed.part.measures[-1].barline is not None


class TestRests:
    def test_rest_is_parsed(self):
        result = parse_spine("**kern\n*M4/4\n=\n4r\n4c\n2r\n")
        events = result.part.measures[0].events
        assert [type(e).__name__ for e in events] == ["Rest", "Note", "Rest"]
        assert result.part.measures[0].quarter_length == Fraction(4)


class TestUnknownTokens:
    def test_unknown_interpretations_are_reported(self):
        result = parse_spine("**kern\n*staff1\n*IVox\n=\n4c\n")
        assert result.unrecognised == ["*staff1", "*IVox"]

    def test_ornaments_do_not_block_the_note(self):
        """`;` fermata, `L`/`J` legaturi de codite — desen, nu continut."""
        result = parse_spine("**kern\n=\n4cL\n4dJ;\n")
        assert [str(e.pitch) for e in result.part.measures[0].events] == ["C4", "D4"]


class TestMergeVoices:
    def test_builds_a_choral_score(self):
        spines = [parse_spine(SOPRANO) for _ in range(4)]
        score = merge_voices(
            spines, [("S", "Soprano"), ("A", "Alto"), ("T", "Tenor"), ("B", "Bass")]
        )
        assert [p.id for p in score.parts] == ["S", "A", "T", "B"]
        assert score.measure_count == 2

    def test_voice_count_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="portative"):
            merge_voices([parse_spine(SOPRANO)], [("S", "Soprano"), ("A", "Alto")])


class TestEndToEnd:
    def test_kern_becomes_musicxml_that_music21_can_read(self, tmp_path):
        """Lantul complet, fara AI: tokeni -> model intern -> MusicXML -> parsare."""
        music21 = pytest.importorskip("music21")
        from app.musicxml import export

        score = merge_voices(
            [parse_spine(SOPRANO) for _ in range(4)],
            [("S", "Soprano"), ("A", "Alto"), ("T", "Tenor"), ("B", "Bass")],
        )
        path = export.write(score, tmp_path / "coral.musicxml")
        parsed = music21.converter.parse(str(path))

        assert len(parsed.parts) == 4
        assert [n.pitch.nameWithOctave for n in parsed.parts[0].flatten().notes][:3] == [
            "G4", "A4", "B4"
        ]

    def test_the_result_passes_validation(self):
        from app.validation.rules import errors, validate

        score = merge_voices(
            [parse_spine(SOPRANO) for _ in range(4)],
            [("S", "Soprano"), ("A", "Alto"), ("T", "Tenor"), ("B", "Bass")],
        )
        # Vocile joase primesc aici linia sopranei, deci ambitusul va fi
        # semnalat — dar ca avertisment, nu ca eroare de structura.
        assert errors(validate(score)) == []
