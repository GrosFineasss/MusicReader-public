"""Asamblarea paginii: portative decupate -> partitura SATB continua."""

import pytest

from fractions import Fraction

from app.recognition.page import assemble_page, group_into_systems, part_measure_counts
from tests.test_closed_score import BASS_STAFF, TREBLE_STAFF

SYSTEM = [TREBLE_STAFF, BASS_STAFF]


class TestGrouping:
    def test_pairs_staves_into_systems(self):
        assert group_into_systems(["a", "b", "c", "d"]) == [["a", "b"], ["c", "d"]]

    def test_odd_number_of_staves_is_refused(self):
        """Un portativ in plus sau in minus inseamna detectie gresita."""
        with pytest.raises(ValueError, match="nu se impart"):
            group_into_systems(["a", "b", "c"])


class TestAssembly:
    def test_a_single_system_becomes_four_voices(self):
        page = assemble_page(SYSTEM)
        assert [p.id for p in page.score.parts] == ["S", "A", "T", "B"]
        assert page.systems == 1

    def test_systems_are_concatenated_not_stacked(self):
        """Al doilea sistem continua aceleasi voci, nu adauga voci noi."""
        one = assemble_page(SYSTEM)
        two = assemble_page(SYSTEM * 2)
        assert len(two.score.parts) == len(one.score.parts) == 4
        assert two.score.measure_count == one.score.measure_count * 2

    def test_measure_numbers_run_continuously(self):
        page = assemble_page(SYSTEM * 3)
        numbers = [m.number for m in page.score.part("S").measures]
        assert numbers == list(range(1, len(numbers) + 1))

    def test_all_voices_stay_aligned(self):
        page = assemble_page(SYSTEM * 2)
        assert len(set(part_measure_counts(page.score).values())) == 1

    def test_title_is_carried_through(self):
        assert assemble_page(SYSTEM, title="Cantec").score.title == "Cantec"


class TestDamageReporting:
    def test_an_unreadable_system_is_reported_not_hidden(self):
        """Un sistem fara `*^` e sarit — dar partitura spune ca lipseste."""
        broken = ["**kern\n*clefG2\n=\n4c\n==\n*-\n", BASS_STAFF]
        page = assemble_page(SYSTEM + broken)
        assert page.systems == 1
        assert any("sistemul 2" in problem for problem in page.problems)
        assert any("1 din 2" in problem for problem in page.problems)


class TestFallback:
    """Cand pagina nu se organizeaza in sisteme corale, tot trebuie sa iasa ceva.

    Un MusicXML imperfect, pe care il poti deschide si corecta in MuseScore,
    e mai util decat un esec din care nu iese niciun fisier.
    """

    def test_an_odd_number_of_staves_still_produces_a_score(self):
        page = assemble_page(SYSTEM + [TREBLE_STAFF])
        assert page.score is not None
        # Trei portative, fiecare cu `*^`, deci doua voci fiecare.
        assert len(page.score.parts) == 6

    def test_the_fallback_says_what_it_did(self):
        page = assemble_page(SYSTEM + [TREBLE_STAFF])
        assert any("o parte per portativ" in problem for problem in page.problems)
        assert any("nu se impart" in problem for problem in page.problems)

    def test_systems_is_zero_so_nothing_looks_assembled(self):
        assert assemble_page(SYSTEM + [TREBLE_STAFF]).systems == 0

    def test_staves_without_voice_splits_fall_back_too(self):
        """Cazul real: modelul nu emite `*^`, deci nicio pagina nu se imparte."""
        single = "**kern\n*clefG2\n*M4/4\n=\n4c\n4d\n4e\n4f\n==\n*-\n"
        page = assemble_page([single, single])
        assert len(page.score.parts) == 2
        assert any("o parte per portativ" in problem for problem in page.problems)

    def test_the_fallback_result_is_valid_musicxml(self, tmp_path):
        music21 = pytest.importorskip("music21")
        from app.musicxml import export

        page = assemble_page(SYSTEM + [TREBLE_STAFF], title="Pagina")
        parsed = music21.converter.parse(str(export.write(page.score, tmp_path / "f.musicxml")))
        assert len(parsed.parts) == 6

    def test_parts_are_named_after_their_staff(self):
        page = assemble_page(SYSTEM + [TREBLE_STAFF])
        assert all(p.name.startswith("Portativ") for p in page.score.parts)


class TestEqualLengthParts:
    """MusicXML nu admite parti cu numere diferite de masuri.

    O partitura are N masuri si fiecare parte trebuie sa le contina pe toate.
    MuseScore chiar **refuza sa deschida** fisierul altfel — descoperit pe o
    pagina reala, ale carei sapte portative se deschideau separat, dar nu si
    impreuna.
    """

    @pytest.fixture
    def uneven(self):
        short = "**kern\n*clefG2\n*M4/4\n=\n4c\n4d\n4e\n4f\n==\n*-\n"
        long = short.replace("==\n", "=\n4g\n4a\n4b\n4cc\n==\n")
        return [short, long]

    def test_all_parts_end_up_with_the_same_measure_count(self, uneven):
        counts = set(part_measure_counts(assemble_page(uneven).score).values())
        assert len(counts) == 1

    def test_the_padding_is_rests_not_silence_in_the_data(self, uneven):
        from app.music_model.model import Rest

        score = assemble_page(uneven).score
        shortest = min(score.parts, key=lambda p: sum(len(m.events) for m in p.measures))
        assert any(isinstance(e, Rest) for m in shortest.measures for e in m.events)

    def test_padded_measures_are_complete(self, uneven):
        from fractions import Fraction

        for part in assemble_page(uneven).score.parts:
            assert all(m.quarter_length == Fraction(4) for m in part.measures)

    def test_nothing_is_padded_when_parts_already_match(self):
        from app.recognition.page import pad_parts_to_equal_length

        assert pad_parts_to_equal_length(assemble_page(SYSTEM).score) == 0

    def test_padding_reports_how_much_it_added(self):
        """Cate masuri s-au completat = cat de multa muzica lipseste."""
        from app.recognition.page import pad_parts_to_equal_length

        score = assemble_page(SYSTEM).score
        score.part("B").measures.pop()
        assert pad_parts_to_equal_length(score) == 1
        assert len(set(part_measure_counts(score).values())) == 1


class TestEndToEnd:
    def test_a_two_system_page_becomes_musicxml(self, tmp_path):
        music21 = pytest.importorskip("music21")
        from app.musicxml import export

        page = assemble_page(SYSTEM * 2, title="Pagina")
        parsed = music21.converter.parse(str(export.write(page.score, tmp_path / "p.musicxml")))

        assert len(parsed.parts) == 4
        for part in parsed.parts:
            assert len(part.getElementsByClass("Measure")) == page.score.measure_count

    def test_the_assembled_page_has_no_structural_errors(self):
        from app.validation.rules import errors, validate

        page = assemble_page(SYSTEM * 2)
        found = errors(validate(page.score))
        assert found == [], "\n".join(str(f) for f in found)


# --------------------------------------------------------------------------
# Masuri taiate de hotarul dintre sisteme (§14, observatie de pe pagina 3)
# --------------------------------------------------------------------------

# Un sistem care se opreste la jumatatea ultimei masuri: doua patrimi scrise,
# doua raman pe randul urmator. Exact ce se vede pe manuscris.
CUT_TREBLE_FIRST = """**kern
*clefG2
*M4/4
=
*^
4g\t4d
4a\t4d
2b\t2d
=\t=
2cc\t2e
*v\t*v
==
*-
"""

CUT_BASS_FIRST = """**kern
*clefF4
*M4/4
=
*^
4d\t4G
4c#\t4A
2B\t2G
=\t=
2c\t2E
*v\t*v
==
*-
"""

# Sistemul urmator incepe cu restul acelei masuri — tot doua patrimi — si
# continua cu o masura intreaga. Cheia si metrica se rescriu, cum face modelul
# la fiecare sistem; nefiind schimbate, nu impiedica lipirea.
CUT_TREBLE_SECOND = """**kern
*clefG2
*M4/4
=
*^
2b\t2g
=\t=
4g\t4d
4a\t4d
4b\t4d
4cc\t4e
*v\t*v
==
*-
"""

CUT_BASS_SECOND = """**kern
*clefF4
*M4/4
=
*^
2B\t2G
=\t=
4d\t4G
4e\t4A
4B\t4G
4c\t4E
*v\t*v
==
*-
"""

CUT_FIRST = [CUT_TREBLE_FIRST, CUT_BASS_FIRST]
CUT_SECOND = [CUT_TREBLE_SECOND, CUT_BASS_SECOND]


def _lengths(score, part_id="S"):
    return [m.quarter_length for m in score.part(part_id).measures]


class TestMeasureCutByTheSystemBreak:
    def test_the_two_halves_become_one_measure(self):
        score = assemble_page(CUT_FIRST + CUT_SECOND).score
        assert _lengths(score) == [Fraction(4), Fraction(4), Fraction(4)]

    def test_every_voice_is_joined_not_just_the_first(self):
        score = assemble_page(CUT_FIRST + CUT_SECOND).score
        for part in score.parts:
            assert _lengths(score, part.id) == [Fraction(4)] * 3

    def test_the_join_is_reported(self):
        page = assemble_page(CUT_FIRST + CUT_SECOND)
        assert any("taiata de hotarul" in problem for problem in page.problems)

    def test_measure_numbers_stay_unbroken(self):
        score = assemble_page(CUT_FIRST + CUT_SECOND).score
        for part in score.parts:
            assert [m.number for m in part.measures] == [1, 2, 3]

    def test_the_joined_measure_keeps_the_notes_of_both_halves(self):
        """Lipirea nu trebuie sa piarda nimic: 3 note + 1 nota + 1 nota."""
        score = assemble_page(CUT_FIRST + CUT_SECOND).score
        assert [len(m.events) for m in score.part("S").measures] == [3, 2, 4]

    def test_the_result_has_no_structural_errors(self):
        from app.validation.rules import errors, validate

        assert errors(validate(assemble_page(CUT_FIRST + CUT_SECOND).score)) == []

    def test_a_page_without_cut_measures_is_untouched(self):
        """Regresie: cazul obisnuit nu are voie sa se schimbe."""
        page = assemble_page(SYSTEM * 2)
        assert part_measure_counts(page.score) == {"S": 4, "A": 4, "T": 4, "B": 4}
        assert not any("taiata" in problem for problem in page.problems)

    def test_one_voice_that_does_not_close_stops_the_join(self):
        """Daca o singura voce nu inchide masura, nu se lipeste nimic.

        Vocile desincronizate sunt greseala cea mai costisitoare (§1.1), deci
        raspunsul se ia pe toate deodata sau deloc.
        """
        odd = CUT_TREBLE_SECOND.replace("2b\t2g", "4b\t2g")
        page = assemble_page(CUT_FIRST + [odd, CUT_BASS_SECOND])
        assert part_measure_counts(page.score)["S"] == 4
        assert not any("taiata" in problem for problem in page.problems)

    def test_nothing_is_joined_across_a_repeat_barline(self):
        """3 + 1 face tot 4, dar la repetitie masura scurta e scrisa dinadins."""
        before = CUT_TREBLE_FIRST.replace("2cc\t2e", "4.cc\t4.e").replace("==", "=:|!")
        bass = CUT_BASS_FIRST.replace("2c\t2E", "4.c\t4.E").replace("==", "=:|!")
        after = CUT_TREBLE_SECOND.replace("2b\t2g", "4b\t4g")
        after_bass = CUT_BASS_SECOND.replace("2B\t2G", "4B\t4G")
        page = assemble_page([before, bass] + [after, after_bass])
        assert not any("taiata" in problem for problem in page.problems)


class TestVociGoale:
    """O voce fara nicio masura nu are voie sa opreasca pagina (§15.26).

    Gasit in timpul unei antrenari: modelul poate emite un spine care se deschide
    si se inchide **fara nimic intre**, iar asamblarea cadea cu `IndexError` —
    deci dintr-o poza intreaga nu mai iesea nimic, din cauza unei singure voci
    goale. `_joinable` verifica deja cazul; propagarea cheii spre vocea noua, nu.

    Se testeaza direct pe `_append_system`, unde a fost greseala: ca sa se ajunga
    acolo prin `assemble_systems` ar trebui construit Humdrum-ul care produce
    exact defectul, iar testul ar spune atunci mai mult despre parser.
    """

    def masura(self, numar, cu_note=True):
        from app.music_model.model import Duration, Measure, Note, NoteType, Pitch

        evenimente = (
            [Note(pitch=Pitch(step="C", octave=4), duration=Duration(NoteType.WHOLE))]
            if cu_note else []
        )
        return Measure(number=numar, events=evenimente)

    def partitura(self, voci):
        from app.music_model.model import Part, Score

        return Score(parts=[
            Part(id=id_, name=id_, measures=[self.masura(i + 1) for i in range(cate)])
            for id_, cate in voci
        ])

    def test_o_voce_noua_fara_masuri_nu_opreste_asamblarea(self):
        from app.recognition.page import _append_system

        target = self.partitura([("S", 1)])
        addition = self.partitura([("S", 1), ("B", 0)])  # B intra goala

        _append_system(target, addition)  # nu trebuie sa ridice

        assert {p.id for p in target.parts} == {"S", "B"}

    def test_golul_se_completeaza_si_se_raporteaza(self):
        """Vocea goala lasa o gaura, iar gaura se **spune** — nu se ascunde.

        `_append_system` nu egalizeaza singur: `assemble_systems` cheama
        `pad_parts_to_equal_length` la final si pune in raport cate masuri a
        completat. Asa, o voce pe care modelul n-a citit-o ramane vizibila in
        rezultat, in loc sa treaca drept muzica scrisa asa.
        """
        from app.recognition.page import _append_system, pad_parts_to_equal_length

        target = self.partitura([("S", 1)])
        _append_system(target, self.partitura([("S", 1), ("B", 0)]))

        completate = pad_parts_to_equal_length(target)
        assert completate > 0, "gaura n-ar fi fost raportata"
        assert len({len(p.measures) for p in target.parts}) == 1
