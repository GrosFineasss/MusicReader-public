"""Teste pentru reprezentarea muzicala interna. Logica pura, fara AI, fara I/O."""

from fractions import Fraction

import pytest

from app.music_model.model import (
    Accidental,
    Clef,
    Duration,
    KeySignature,
    Measure,
    Note,
    NoteType,
    Part,
    Pitch,
    Rest,
    TimeSignature,
    empty_satb,
)


class TestDuration:
    @pytest.mark.parametrize(
        "note_type,dots,expected",
        [
            (NoteType.WHOLE, 0, Fraction(4)),
            (NoteType.HALF, 0, Fraction(2)),
            (NoteType.HALF, 1, Fraction(3)),
            (NoteType.QUARTER, 0, Fraction(1)),
            (NoteType.QUARTER, 1, Fraction(3, 2)),
            (NoteType.QUARTER, 2, Fraction(7, 4)),
            (NoteType.EIGHTH, 0, Fraction(1, 2)),
            (NoteType.SIXTEENTH, 1, Fraction(3, 8)),
        ],
    )
    def test_quarter_length(self, note_type, dots, expected):
        assert Duration(note_type, dots).quarter_length == expected

    def test_durations_stay_exact(self):
        """Trei optimi fac exact o patrime si jumatate — nu 1.4999999."""
        total = sum((Duration(NoteType.EIGHTH).quarter_length for _ in range(3)), Fraction(0))
        assert total == Fraction(3, 2)
        assert isinstance(total, Fraction)

    def test_dots_are_bounded(self):
        with pytest.raises(ValueError):
            Duration(NoteType.QUARTER, 4)


class TestPitch:
    def test_middle_c_is_60(self):
        assert Pitch("C", 4).midi == 60

    def test_alter_shifts_midi(self):
        assert Pitch("F", 4, 1).midi == Pitch("F", 4).midi + 1
        assert Pitch("B", 3, -1).midi == Pitch("B", 3).midi - 1

    def test_accidental_does_not_move_staff_position(self):
        """Fa si Fa-diez stau pe aceeasi linie a portativului."""
        assert Pitch("F", 4).diatonic_index == Pitch("F", 4, 1).diatonic_index

    def test_diatonic_index_orders_by_staff_position(self):
        assert Pitch("B", 3).diatonic_index < Pitch("C", 4).diatonic_index

    def test_rejects_invalid_step(self):
        with pytest.raises(ValueError):
            Pitch("H", 4)

    def test_str(self):
        assert str(Pitch("E", 5, -1, Accidental.FLAT)) == "Eb5"


class TestKeySignature:
    def test_c_major_alters_nothing(self):
        assert KeySignature(0).altered_steps == {}

    def test_sharps_follow_the_circle_of_fifths(self):
        assert KeySignature(3).altered_steps == {"F": 1, "C": 1, "G": 1}

    def test_flats_follow_the_reverse_order(self):
        assert KeySignature(-2).altered_steps == {"B": -1, "E": -1}

    def test_rejects_impossible_signature(self):
        with pytest.raises(ValueError):
            KeySignature(8)


class TestTimeSignature:
    @pytest.mark.parametrize(
        "beats,beat_type,expected",
        [(4, 4, Fraction(4)), (3, 4, Fraction(3)), (6, 8, Fraction(3)), (2, 2, Fraction(4))],
    )
    def test_measure_capacity(self, beats, beat_type, expected):
        assert TimeSignature(beats, beat_type).measure_quarter_length == expected


class TestMeasureAndPart:
    def test_measure_length_sums_events(self):
        measure = Measure(
            1,
            [
                Note(Pitch("C", 4), Duration(NoteType.QUARTER)),
                Rest(Duration(NoteType.EIGHTH)),
                Note(Pitch("D", 4), Duration(NoteType.EIGHTH)),
            ],
        )
        assert measure.quarter_length == Fraction(2)

    def test_attributes_propagate_forward(self):
        part = Part("S", "Soprano")
        part.measures.append(Measure(1, clef=Clef.bass(), time=TimeSignature(3, 4)))
        part.measures.append(Measure(2))
        part.measures.append(Measure(3, time=TimeSignature(4, 4)))

        clef, _, time = part.context_at(2)
        assert clef == Clef.bass()
        assert time == TimeSignature(3, 4)

        _, _, time = part.context_at(3)
        assert time == TimeSignature(4, 4)

    def test_defaults_when_nothing_declared(self):
        part = Part("S", measures=[Measure(1)])
        clef, key, time = part.context_at(1)
        assert (clef, key, time) == (Clef.treble(), KeySignature(0), TimeSignature(4, 4))


class TestSATB:
    def test_skeleton_has_four_voices(self):
        score = empty_satb(3)
        assert [p.id for p in score.parts] == ["S", "A", "T", "B"]
        assert score.measure_count == 3

    def test_tenor_uses_octave_transposing_clef(self):
        """Tenorul de cor se scrie in cheia Sol, dar suna cu o octava mai jos."""
        clef, _, _ = empty_satb(1).part("T").context_at(1)
        assert clef.sign == "G"
        assert clef.octave_change == -1


class TestTaiereaInSisteme:
    """`measures_slice`: o bucata de partitura care se poate citi singura."""

    def _score(self):
        from app.music_model.model import empty_satb

        score = empty_satb(6, TimeSignature(4, 4), KeySignature(2))
        for part in score.parts:
            for measure in part.measures:
                measure.events = [Rest(Duration(NoteType.WHOLE))]
        return score

    def test_pastreaza_numerele_de_masura(self):
        from app.music_model.model import measures_slice

        cut = measures_slice(self._score(), 3, 5)
        assert [m.number for m in cut.part("S").measures] == [3, 4, 5]

    def test_prima_masura_primeste_contextul_in_vigoare(self):
        """Un sistem care incepe la masura 3 trebuie sa-si stie cheia si armura.

        Fara asta s-ar citi in cheia Sol implicita, iar basul ar iesi cu o
        octava gresita.
        """
        from app.music_model.model import measures_slice

        cut = measures_slice(self._score(), 3, 5)
        opening = cut.part("B").measures[0]
        assert opening.clef == Clef.bass()
        assert opening.key == KeySignature(2)
        assert opening.time == TimeSignature(4, 4)

    def test_masurile_de_la_mijloc_nu_primesc_declaratii_inutile(self):
        from app.music_model.model import measures_slice

        cut = measures_slice(self._score(), 3, 5)
        assert cut.part("S").measures[1].clef is None

    def test_un_interval_gol_e_refuzat(self):
        from app.music_model.model import measures_slice

        with pytest.raises(ValueError, match="interval gol"):
            measures_slice(self._score(), 5, 3)


class TestPortativeleCareTac:
    """`drop_silent_staves`: tinta are doar portativele desenate in sistem.

    „Doamne bun asculta" (2026-10-04): solo-ul tace pe primele trei sisteme, iar
    manuscrisul nu-i deseneaza portativul; tinta iesea totusi cu trei portative.
    """

    def _cu_solo(self, solo_canta: bool):
        from app.music_model.model import Score

        def parte(voce, nota):
            evenimente = [Note(Pitch("C", 4), Duration(NoteType.WHOLE))] if nota else [
                Rest(Duration(NoteType.WHOLE))
            ]
            return Part(id=voce, measures=[Measure(number=1, events=evenimente)])

        return Score(
            parts=[parte("X1", solo_canta), parte("S", True), parte("A", True),
                   parte("T", True), parte("B", True)],
            staves=[["X1"], ["S", "A"], ["T", "B"]],
        )

    def test_portativul_care_tace_iese(self):
        from app.music_model.model import drop_silent_staves

        doua = drop_silent_staves(self._cu_solo(solo_canta=False), 2)
        assert doua.staves == [["S", "A"], ["T", "B"]]
        assert [p.id for p in doua.parts] == ["S", "A", "T", "B"]

    def test_cand_decupajul_are_toate_portativele_nu_se_schimba_nimic(self):
        from app.music_model.model import drop_silent_staves

        score = self._cu_solo(solo_canta=False)
        assert drop_silent_staves(score, 3) is score

    def test_mai_multe_portative_in_decupaj_decat_in_referinta_e_refuzat(self):
        from app.music_model.model import drop_silent_staves

        with pytest.raises(ValueError, match="in referinta"):
            drop_silent_staves(self._cu_solo(solo_canta=True), 4)

    def test_un_portativ_care_canta_nu_se_scoate(self):
        """Doua portative in decupaj, dar solo-ul canta: nepotrivire, nu ghicit."""
        from app.music_model.model import drop_silent_staves

        with pytest.raises(ValueError, match="tac"):
            drop_silent_staves(self._cu_solo(solo_canta=True), 2)
