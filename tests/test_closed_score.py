"""Partitura stransa: doua voci pe acelasi portativ, cu codite opuse.

E forma in care scrie utilizatorul, si e cazul greu: un singur portativ trebuie
sa produca doua voci independente. In Humdrum asta se scrie prin impartirea
coloanei — `*^` desface, `*v` lipeste la loc.
"""

from fractions import Fraction

import pytest

from app.music_model.model import Clef, KeySignature, TimeSignature
from app.recognition.kern import (
    parse_closed_score,
    parse_document,
    split_voices,
)

TREBLE_STAFF = """**kern
*clefG2
*k[f#]
*M4/4
=
*^
4g\t4d
4a\t4d
2b\t2d
=\t=
*v\t*v
1cc
==
*-
"""

BASS_STAFF = """**kern
*clefF4
*k[f#]
*M4/4
=
*^
4d\t4G
4c#\t4A
2B\t2G
=\t=
*v\t*v
1C
==
*-
"""


class TestSplitVoices:
    def test_a_split_staff_yields_two_voices(self):
        assert len(split_voices(TREBLE_STAFF)) == 2

    def test_an_unsplit_staff_yields_one_voice(self):
        assert len(split_voices("**kern\n*clefG2\n=\n4c\n==\n*-\n")) == 1

    def test_the_chord_column_never_becomes_a_voice(self):
        """`**mxhm` sunt acorduri de jazz, nu o voce corala."""
        doc = "**kern\t**mxhm\n*clefG2\t*\n=\t=\n4c\t.\n"
        assert len(split_voices(doc)) == 1


class TestVoiceIndependence:
    @pytest.fixture
    def voices(self):
        return parse_document(TREBLE_STAFF, [("S", "Soprano"), ("A", "Alto")])

    def test_each_voice_reads_its_own_notes(self, voices):
        soprano, alto = voices
        assert [str(e.pitch) for e in soprano.part.measures[0].events] == ["G4", "A4", "B4"]
        assert [str(e.pitch) for e in alto.part.measures[0].events] == ["D4", "D4", "D4"]

    def test_nothing_is_silently_dropped(self, voices):
        assert all(v.unrecognised == [] for v in voices)

    def test_both_voices_have_complete_measures(self, voices):
        for voice in voices:
            assert all(m.quarter_length == Fraction(4) for m in voice.part.measures)

    def test_the_new_voice_inherits_clef_key_and_meter(self):
        """Vocea de jos apare dupa declaratii; fara mostenire ar fi citita gresit."""
        _, bass = parse_document(BASS_STAFF, [("T", "Tenor"), ("B", "Bass")])
        clef, key, time = bass.part.context_at(1)
        assert clef == Clef.bass()
        assert key == KeySignature(1)
        assert time == TimeSignature(4, 4)

    def test_a_bass_clef_voice_is_not_read_in_treble(self):
        _, bass = parse_document(BASS_STAFF, [("T", "Tenor"), ("B", "Bass")])
        assert [str(e.pitch) for e in bass.part.measures[0].events] == ["G3", "A3", "G3"]


class TestUnisonAfterJoin:
    """Cand coloanele se reunesc, nota e cantata de amandoua vocile."""

    @pytest.fixture
    def voices(self):
        return parse_document(TREBLE_STAFF, [("S", "Soprano"), ("A", "Alto")])

    def test_the_shared_note_reaches_both_voices(self, voices):
        for voice in voices:
            assert [str(e.pitch) for e in voice.part.measures[1].events] == ["C5"]

    def test_no_voice_is_left_with_an_incomplete_measure(self, voices):
        """Daca unisonul ar cadea intr-o singura voce, cealalta ar avea gauri."""
        for voice in voices:
            assert voice.part.measures[1].quarter_length == Fraction(4)

    def test_both_voices_end_with_the_same_measure_count(self, voices):
        assert len({len(v.part.measures) for v in voices}) == 1


class TestClosedScoreSATB:
    @pytest.fixture
    def score(self):
        return parse_closed_score([TREBLE_STAFF, BASS_STAFF])

    def test_two_staves_become_four_parts(self, score):
        assert [p.id for p in score.parts] == ["S", "A", "T", "B"]

    def test_voices_are_ordered_top_to_bottom(self, score):
        """Coditele in sus = vocea de sus. `*^` pune ramura noua la dreapta."""
        first = {p.id: [str(e.pitch) for e in p.measures[0].events] for p in score.parts}
        assert first["S"] == ["G4", "A4", "B4"]
        assert first["A"] == ["D4", "D4", "D4"]
        # In kern inaltimea e absoluta, independenta de cheie: `d` e Re4,
        # `D` e Re3. Cheia spune unde se *deseneaza* nota, nu care e.
        assert first["T"] == ["D4", "C#4", "B3"]
        assert first["B"] == ["G3", "A3", "G3"]

    def test_the_system_is_aligned(self, score):
        assert len({len(p.measures) for p in score.parts}) == 1

    def test_wrong_number_of_staves_is_refused(self):
        with pytest.raises(ValueError, match="2 portative"):
            parse_closed_score([TREBLE_STAFF])

    def test_a_staff_without_a_split_is_reported(self):
        """Daca modelul rateaza `*^`, vrem o eroare, nu o voce lipsa in tacere."""
        with pytest.raises(ValueError, match="in loc de 2"):
            parse_closed_score(["**kern\n*clefG2\n=\n4c\n==\n*-\n", BASS_STAFF])


class TestEndToEnd:
    def test_closed_score_becomes_valid_musicxml(self, tmp_path):
        music21 = pytest.importorskip("music21")
        from app.musicxml import export

        score = parse_closed_score([TREBLE_STAFF, BASS_STAFF])
        parsed = music21.converter.parse(str(export.write(score, tmp_path / "cor.musicxml")))

        assert len(parsed.parts) == 4
        assert [p.partName for p in parsed.parts] == ["Soprano", "Alto", "Tenor", "Bass"]

    def test_the_result_has_no_structural_errors(self):
        from app.validation.rules import errors, validate

        score = parse_closed_score([TREBLE_STAFF, BASS_STAFF])
        found = errors(validate(score))
        assert found == [], "\n".join(str(f) for f in found)


class TestImpartireLaMijloc:
    """`*^` care apare dupa ce muzica a inceput.

    Cazul apare la fiecare pagina de cor unde T&B intra mai tarziu, sau unde un
    portativ se desparte in divisi dupa un pasaj la unison. Era limitarea
    cunoscuta din PROJECT.md §11.5, gasita pe prima transcriere de sistem
    intreg facuta de `smt-grandstaff`.
    """

    STAFF = """**kern
*clefF4
*M4/4
=
1G
=
1A
=
*^
2c\t2G
2d\t2A
==
*v\t*v
*-
"""

    def test_vocea_noua_continua_numerotarea_masurilor(self):
        voices = parse_document(self.STAFF, [("T", "Tenor"), ("B", "Bass")])
        assert [len(result.part.measures) for result in voices] == [3, 3]
        assert voices[1].part.measures[-1].number == 3

    def test_vocea_noua_mosteneste_muzica_dinaintea_impartirii(self):
        # Pana la impartire coloana era una singura, deci vocile cantau la
        # unison: notele de dinainte sunt ale amandurora.
        voices = parse_document(self.STAFF, [("T", "Tenor"), ("B", "Bass")])
        assert voices[0].part.measures[0].events == voices[1].part.measures[0].events

    def test_vocea_noua_mosteneste_cheia(self):
        voices = parse_document(self.STAFF, [("T", "Tenor"), ("B", "Bass")])
        assert voices[1].part.measures[0].clef == Clef("F", 4)


def test_vocile_in_plus_primesc_nume_unice_pe_tot_sistemul():
    """A treia voce de sus si a treia de jos nu pot fi amandoua „X3".

    Doua parti cu acelasi id fac `Score.part()` sa intoarca mereu prima, iar
    exportul scria doua `score-part` cu acelasi id — MusicXML invalid. Vazut pe
    pagina 3, unde modelul a deschis mai multe voci decat are partitura (§15.6).
    """
    from app.recognition.kern import parse_open_system

    def row(*fields: str) -> str:
        return "\t".join(fields)

    system = "\n".join(
        [
            row(*["**kern"] * 6),
            row(*["*clefG2"] * 3, *["*clefF4"] * 3),
            row(*["*M4/4"] * 6),
            row(*["="] * 6),
            row(*["4c"] * 6),
            "",
        ]
    )

    parsed = parse_open_system(system)
    ids = [part.id for part in parsed.score.parts]
    assert len(ids) == 6
    assert len(ids) == len(set(ids)), ids


class TestUnisonSiDespartireDinNou:
    """`*^` dupa `*v`: vocile unite se dau inapoi, nu se inventeaza altele.

    Cazul apare pe orice pagina corala: S si A canta la unison cateva masuri,
    apoi se despart iar. Pana la reparatia din §15.7, fiecare astfel de loc
    adauga o voce fantoma — de acolo veneau cele 8-11 voci ale paginii 3.
    """

    DOCUMENT = (
        "**kern\n"
        "*^\n"
        "4c\t4e\n"
        "*v\t*v\n"
        "4g\n"
        "*^\n"
        "4c\t4e\n"
    )

    def test_numarul_de_voci_ramane_doi(self):
        assert len(split_voices(self.DOCUMENT)) == 2

    def test_fiecare_voce_isi_primeste_propria_linie(self):
        """Proba care conteaza: nu numarul, ci **continutul**.

        Inainte, vocea a doua primea `4c` — linia primei — iar `4e` ajungea la
        vocea inventata.
        """
        first, second = split_voices(self.DOCUMENT)
        assert first == ["4c", "4g", "4c"]
        assert second == ["4e", "4g", "4e"]

    def test_unisonul_se_scrie_in_amandoua(self):
        """Nota dintre `*v` si `*^` e a amandurora — regula unisonului."""
        for voice in split_voices(self.DOCUMENT):
            assert voice.count("4g") == 1

    def test_o_despartire_fara_unison_inainte_creeaza_o_voce(self):
        """Regresie: cazul obisnuit nu are voie sa se schimbe."""
        assert len(split_voices("**kern\n*^\n4c\t4e\n")) == 2

    def test_trei_voci_unite_se_despart_una_si_restul(self):
        document = (
            "**kern\n*^\n*\t*^\n4c\t4e\t4g\n*v\t*v\t*v\n4c\n*^\n4c\t4e\n"
        )
        voices = split_voices(document)
        assert len(voices) == 3
        assert voices[0][-1] == "4c"
        assert voices[1][-1] == voices[2][-1] == "4e"


# ---------------------------------------------------------------------------
# Asezarea declarata (PROJECT.md §15.10)
# ---------------------------------------------------------------------------

def _document(*rows: str) -> str:
    """Un document Humdrum scris citibil: `|` tine locul tabului.

    Fixturile scrise ca literal cu escape-uri ies pe un singur rand de 160
    de caractere, in care nu se vede structura — si structura e tot ce
    testeaza clasele de mai jos.
    """
    return chr(10).join(row.replace(chr(124), chr(9)) for row in rows) + chr(10)


# Trei portative — jos in cheia Fa, mijloc si sus in cheia Sol — iar portativul
# de jos se desparte in doua voci. Cazul unei cantari cu solo: doua portative
# impart aceeasi cheie, deci cheia nu mai poate spune care e care.
TREI_PORTATIVE = _document(
    "**kern|**kern|**kern",
    "*clefF4|*clefG2|*clefG2",
    "*M4/4|*M4/4|*M4/4",
    "*^|*|*",
    "=|=|=|=",
    "4C|4c|4g|4cc",
    "==|==|==|==",
    "*-|*-|*-|*-",
)

# Acelasi sistem, dar cu cheia Fa pe coloana din mijloc: modelul a scris
# portativele in alta ordine decat cea de pe pagina.
COLOANE_RASTURNATE = _document(
    "**kern|**kern|**kern",
    "*clefG2|*clefF4|*clefG2",
    "*M4/4|*M4/4|*M4/4",
    "=|=|=",
    "4c|4C|4g",
    "==|==|==",
    "*-|*-|*-",
)




class TestOriginileVocilor:
    """`split_voices` numeroteaza vocile in ordinea crearii, nu a paginii."""

    def test_o_voce_nascuta_din_split_mosteneste_portativul(self):
        from app.recognition.kern import split_voices

        origins: list[int] = []
        voices = split_voices(TREI_PORTATIVE, origins=origins)
        assert len(voices) == 4
        # A patra voce s-a nascut din coloana 0; nu e un portativ nou.
        assert origins == [0, 1, 2, 0]

    def test_pozitia_singura_ar_minti(self):
        # Gruparea „primele doua coloane = portativul de jos" ar pune impreuna
        # vocile 0 si 1, care vin din portative diferite.
        from app.recognition.kern import split_voices

        origins: list[int] = []
        split_voices(TREI_PORTATIVE, origins=origins)
        assert origins[:2] != [0, 0]


class TestAsezareDeclarata:
    def _parse(self, layout):
        from app.recognition.kern import parse_open_system

        return parse_open_system(TREI_PORTATIVE, layout=layout)

    def test_vocile_ajung_pe_portativele_declarate(self):
        parsed = self._parse([["Solo"], ["S"], ["T", "B"]])
        assert [p.id for p in parsed.score.parts] == ["Solo", "S", "T", "B"]
        assert not parsed.problems

    def test_cheia_prinde_coloanele_scrise_in_alta_ordine(self):
        # Modelul a pus cheia Fa pe coloana din mijloc, adica deasupra uneia in
        # cheia Sol. Gruparea reuseste — se face pe coloana de origine — dar
        # numele declarate ar cadea pe alta muzica, deci se semnaleaza.
        from app.recognition.kern import parse_open_system

        parsed = parse_open_system(COLOANE_RASTURNATE, layout=[["S"], ["A"], ["B"]])
        assert any("ordinea de pe pagina" in problem for problem in parsed.problems)

    def test_ordinea_corecta_nu_se_plange(self):
        parsed = self._parse([["Solo"], ["S"], ["T", "B"]])
        assert not any("ordinea de pe pagina" in problem for problem in parsed.problems)

    def test_o_coloana_pe_un_portativ_de_doua_voci_e_unison(self):
        parsed = self._parse([["Solo", "Solo2"], ["S"], ["T", "B"]])
        assert any("ca la unison" in problem for problem in parsed.problems)
        assert [p.id for p in parsed.score.parts] == ["Solo", "Solo2", "S", "T", "B"]

    def test_un_portativ_peste_declaratie_nu_se_pierde(self):
        parsed = self._parse([["S"], ["T", "B"]])
        assert any("peste asezarea declarata" in problem for problem in parsed.problems)
        assert len(parsed.score.parts) == 4  # muzica ramane, cu nume generic

    def test_un_portativ_care_nu_primeste_nimic_se_raporteaza(self):
        parsed = self._parse([["A"], ["Solo"], ["S"], ["T", "B"]])
        assert any("n-a primit nicio coloana" in problem for problem in parsed.problems)


class TestCitireaDeclaratieiDeAsezare:
    def test_partitura_stransa(self):
        from app.staff_detection.staves import parse_layout

        # Intotdeauna o lista de asezari; una singura = „la fel peste tot".
        assert parse_layout("S,A/T,B") == [[["S", "A"], ["T", "B"]]]

    def test_cantare_cu_solo(self):
        from app.staff_detection.staves import parse_layout

        assert parse_layout("X1/S,A/T,B") == [[["X1"], ["S", "A"], ["T", "B"]]]

    def test_cate_o_asezare_pe_sistem(self):
        from app.staff_detection.staves import parse_layout

        # Vocea de solo intra abia la al doilea sistem.
        assert parse_layout("S,A/T,B;X1/S,A/T,B") == [
            [["S", "A"], ["T", "B"]],
            [["X1"], ["S", "A"], ["T", "B"]],
        ]

    def test_id_urile_duplicate_se_refuza(self):
        from app.staff_detection.staves import parse_layout

        with pytest.raises(ValueError, match="de doua ori"):
            parse_layout("S,A/S,B")

    def test_gol_se_refuza(self):
        from app.staff_detection.staves import parse_layout

        with pytest.raises(ValueError, match="despartite prin"):
            parse_layout("//")
