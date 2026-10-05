"""Etapa 4: exportul MusicXML.

Testul care conteaza e round-trip-ul: scriem partitura, o citim inapoi cu o
implementare *independenta* de a noastra (`music21`) si verificam ca a ajuns
acolo aceeasi muzica. Un export care se serializeaza frumos dar pe care nimeni
altcineva nu-l intelege nu valoreaza nimic.
"""

import re
from fractions import Fraction

import pytest

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
)
from app.musicxml import export
from scripts.make_reference_score import build

music21 = pytest.importorskip("music21")


@pytest.fixture(scope="module")
def reference():
    return build()


@pytest.fixture(scope="module")
def parsed(reference, tmp_path_factory):
    path = tmp_path_factory.mktemp("musicxml") / "reference.musicxml"
    export.write(reference, path)
    return music21.converter.parse(str(path))


class TestDivisions:
    def test_divisions_cover_the_shortest_value(self):
        score = Score(parts=[Part("S", measures=[Measure(1, [
            Note(Pitch("C", 4), Duration(NoteType.SIXTEENTH)),
        ])])])
        assert export.divisions_for(score) == 4

    def test_triplet_free_score_needs_no_extra_divisions(self):
        score = Score(parts=[Part("S", measures=[Measure(1, [
            Note(Pitch("C", 4), Duration(NoteType.QUARTER)),
        ])])])
        assert export.divisions_for(score) == 1

    def test_dotted_eighth_forces_finer_divisions(self):
        score = Score(parts=[Part("S", measures=[Measure(1, [
            Note(Pitch("C", 4), Duration(NoteType.EIGHTH, 1)),
        ])])])
        assert export.divisions_for(score) == 4


class TestDocumentShape:
    def test_declares_musicxml_4(self, reference):
        text = export.to_string(reference)
        assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>')
        assert "score-partwise" in text
        assert 'version="4.0"' in text

    def test_has_a_doctype_museScore_expects(self, reference):
        assert "DOCTYPE score-partwise PUBLIC" in export.to_string(reference)

    def test_divisions_appear_once_per_part(self, reference):
        assert export.to_string(reference).count("<divisions>") == len(reference.parts)


class TestRoundTrip:
    def test_all_voices_survive(self, parsed, reference):
        assert len(parsed.parts) == len(reference.parts)

    def test_part_names_survive(self, parsed):
        names = [p.partName for p in parsed.parts]
        assert names == ["Soprano", "Alto", "Tenor", "Bass"]

    def test_pitches_and_durations_survive(self, parsed, reference):
        for original, roundtripped in zip(reference.parts, parsed.parts):
            expected = [
                (str(e.pitch), e.quarter_length) if isinstance(e, Note) else ("rest", e.quarter_length)
                for measure in original.measures
                for e in measure.events
            ]
            actual = [
                ("rest" if n.isRest else n.pitch.nameWithOctave, Fraction(n.duration.quarterLength))
                for n in roundtripped.flatten().notesAndRests
            ]
            assert actual == expected, f"vocea {original.id}"

    def test_measure_lengths_survive(self, parsed):
        for part in parsed.parts:
            for measure in part.getElementsByClass("Measure"):
                assert measure.duration.quarterLength == 4.0

    def test_tenor_clef_octave_change_survives(self, parsed):
        tenor = parsed.parts[2]
        clef = tenor.flatten().getElementsByClass("Clef")[0]
        assert clef.sign == "G"
        assert clef.octaveChange == -1

    def test_tie_survives(self, parsed):
        alto = parsed.parts[1]
        tied = [n for n in alto.flatten().notes if n.tie is not None]
        assert [n.tie.type for n in tied] == ["start", "stop"]

    def test_final_barline_survives(self, parsed):
        for part in parsed.parts:
            last = list(part.getElementsByClass("Measure"))[-1]
            assert last.rightBarline is not None
            assert last.rightBarline.type == "final"


class TestAccidentals:
    def test_written_accidental_is_serialised(self):
        score = Score(parts=[Part("S", "Soprano", [Measure(
            1,
            [Note(Pitch("F", 4, 1, Accidental.SHARP), Duration(NoteType.WHOLE))],
            clef=Clef.treble(),
            key=KeySignature(0),
            time=TimeSignature(4, 4),
        )])])
        text = export.to_string(score)
        assert "<alter>1</alter>" in text
        assert "<accidental>sharp</accidental>" in text

    def test_alteration_from_key_is_not_written_as_accidental(self):
        """In Sol major, Fa-diez ia diezul din armura; nu se scrie langa nota."""
        score = Score(parts=[Part("S", "Soprano", [Measure(
            1,
            [Note(Pitch("F", 4, 1), Duration(NoteType.WHOLE))],
            clef=Clef.treble(),
            key=KeySignature(1),
            time=TimeSignature(4, 4),
        )])])
        text = export.to_string(score)
        assert "<alter>1</alter>" in text
        assert "<accidental>" not in text


class TestErrorHandling:
    def test_unrepresentable_duration_is_refused(self):
        """Mai bine o exceptie explicita decat un MusicXML rotunjit pe tacute."""
        with pytest.raises(ValueError, match="nu e reprezentabila"):
            export._duration_ticks(Fraction(1, 3), 2)


class TestPartialMeasures:
    """Masura scurta dinadins — anacruza si perechea ei (§14, pagina 3)."""

    def _score(self):
        part = Part(id="S", name="Soprano")
        part.measures.append(
            Measure(
                number=1,
                events=[Note(Pitch("C", 5), Duration(NoteType.QUARTER))],
                clef=Clef.treble(),
                key=KeySignature(0),
                time=TimeSignature(4, 4),
                partial=True,
            )
        )
        part.measures.append(
            Measure(number=2, events=[Note(Pitch("D", 5), Duration(NoteType.WHOLE))])
        )
        score = Score()
        score.parts.append(part)
        return score

    def test_a_partial_measure_is_written_as_implicit(self, tmp_path):
        path = export.write(self._score(), tmp_path / "anacruza.musicxml")
        text = path.read_text(encoding="utf-8")
        assert '<measure number="1" implicit="yes">' in text
        assert '<measure number="2">' in text

    def test_the_flag_survives_the_round_trip(self, tmp_path):
        """Citita inapoi, masura trebuie sa ramana scurta *si* marcata asa.

        Fara marcaj, orice cititor — al nostru sau MuseScore — o socoteste
        stricata, si o partitura corecta ar aparea cu erori.
        """
        from app.evaluation.reference import load

        path = export.write(self._score(), tmp_path / "anacruza.musicxml")
        back = load(path)
        measures = back.parts[0].measures
        assert [m.partial for m in measures] == [True, False]
        assert measures[0].quarter_length == Fraction(1)


def test_doua_parti_cu_acelasi_id_nu_se_suprapun(tmp_path):
    """Id-urile MusicXML se dau **pozitional**, nu dupa `part.id`.

    Pe pagina 3, recunoasterea a dat doua voci numite „X3"; cheiate pe id, cele
    doua se suprapuneau, iar fisierul iesea cu `P6` de doua ori si fara `P3` —
    adica MusicXML invalid, dintr-o partitura pe care noi o construisem.
    """
    score = Score()
    for _ in range(2):
        part = Part(id="X3", name="X3")
        part.measures.append(
            Measure(
                number=1,
                events=[Note(Pitch("C", 5), Duration(NoteType.WHOLE))],
                clef=Clef.treble(),
                key=KeySignature(0),
                time=TimeSignature(4, 4),
            )
        )
        score.parts.append(part)

    text = export.write(score, tmp_path / "duplicat.musicxml").read_text(encoding="utf-8")
    ids = re.findall(r'<score-part id="([^"]+)"', text)
    assert ids == ["P1", "P2"]
    assert re.findall(r'<part id="([^"]+)"', text) == ["P1", "P2"]
