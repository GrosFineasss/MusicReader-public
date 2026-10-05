"""Semnele de repetitie, de la sursa pana la tinta de antrenare (2026-10-04).

`pagina3` incepe cu `|:` si o anacruza, iar modelul lipea anacruza de masura 2.
Paginile generate nu aveau niciodata repetitii: `from_music21` nu citea barele,
desi 40% din coralele Bach si 30% din piesele PDMX le au. Testele urmaresc
semnul pe tot drumul: MusicXML -> `Score` -> MusicXML generat -> recitire ->
`**kern` -> inapoi in `Score`, plus lipirea masurilor peste hotarul de sistem.
"""

from pathlib import Path

import pytest

from app.evaluation.reference import load, load_with_layout
from app.music_model.model import measures_slice
from app.recognition.kern import parse_open_system
from app.recognition.kern_writer import to_kern_text
from app.recognition.page import assemble_page
from app.training.bach import musicxml_strans, repetitii
from tests.test_page import CUT_BASS_FIRST, CUT_BASS_SECOND, CUT_TREBLE_FIRST, CUT_TREBLE_SECOND

PAGINA3 = Path(__file__).resolve().parents[1] / "data" / "validation" / "pagina3.musicxml"

pytestmark = pytest.mark.skipif(not PAGINA3.exists(), reason="lipseste pagina3.musicxml")


def deschise(score):
    return {p.id: [m.number for m in p.measures if m.repeat_start] for p in score.parts}


def inchise(score):
    return {
        p.id: [m.number for m in p.measures if m.barline is not None and m.barline.repeat]
        for p in score.parts
    }


def bare(text: str) -> list[str]:
    """Tokenii de bara ai documentului, cate unul pe rand (coloanele repeta acelasi token)."""
    return [line.split("\t")[0] for line in text.splitlines() if line.startswith("=")]


@pytest.fixture(scope="module")
def pagina3():
    return load(PAGINA3)


class TestCitireaDinMusicXML:
    def test_pagina3_deschide_la_1_si_10(self, pagina3):
        assert deschise(pagina3) == {voce: [1, 10] for voce in "SATB"}

    def test_pagina3_inchide_la_9_si_14(self, pagina3):
        assert inchise(pagina3) == {voce: [9, 14] for voce in "SATB"}

    def test_repetitiile_sunt_ale_partiturii(self, pagina3):
        assert repetitii(pagina3) == ({1, 10}, {9, 14})


class TestTinta:
    """Sistemele lui `pagina3` incep la masurile 1, 6 si 12."""

    def test_primul_sistem_incepe_cu_repetitie(self, pagina3):
        text = to_kern_text(measures_slice(pagina3, 1, 5), ties=False)
        assert bare(text) == ["=!|:", "=", "=", "=", "=", "=="]

    def test_bara_care_inchide_si_deschide_e_un_singur_token(self, pagina3):
        text = to_kern_text(measures_slice(pagina3, 6, 11), ties=False)
        assert bare(text) == ["=", "=", "=", "=", "=:|!|:", "=", "=="]

    def test_ultima_bara_inchide_repetitia(self, pagina3):
        text = to_kern_text(measures_slice(pagina3, 12, 14), ties=False)
        assert bare(text) == ["=", "=", "=", "==:|!"]

    def test_fara_repetitii_nu_se_schimba_nimic(self, pagina3):
        """Regresie: masurile 2-5 dau exact barele de dinainte de repetitii."""
        text = to_kern_text(measures_slice(pagina3, 2, 5), ties=False)
        assert bare(text) == ["=", "=", "=", "=", "=="]

    @pytest.mark.parametrize("prima, ultima", [(1, 5), (6, 11), (12, 14)])
    def test_tinta_se_citeste_inapoi_cu_aceleasi_repetitii(self, pagina3, prima, ultima):
        sistem = measures_slice(pagina3, prima, ultima)
        citit = parse_open_system(to_kern_text(sistem, ties=False)).score
        decalaj = prima - 1
        assert deschise(citit)["S"] == [n - decalaj for n in deschise(sistem)["S"]]
        assert inchise(citit)["S"] == [n - decalaj for n in inchise(sistem)["S"]]


class TestPaginaGenerata:
    @pytest.mark.parametrize("acorduri", [False, True])
    def test_musicxml_strans_le_scrie_si_recitirea_le_gaseste(self, pagina3, tmp_path, acorduri):
        pagina3.staves = [["S", "A"], ["T", "B"]]
        xml = musicxml_strans(pagina3, [(1, 5), (6, 11), (12, 14)], True, acorduri)
        assert xml.count('direction="forward"') == 2 * 2  # doua masuri, doua portative
        assert xml.count('direction="backward"') == 2 * 2
        cale = tmp_path / "pagina.musicxml"
        cale.write_text(xml, encoding="utf-8")
        recitit, _, _ = load_with_layout(cale, min_voices=2)
        assert deschise(recitit) == {voce: [1, 10] for voce in "SATB"}
        assert inchise(recitit) == {voce: [9, 14] for voce in "SATB"}


class TestLipirea:
    def test_nu_se_lipeste_peste_repetitia_de_la_inceputul_sistemului(self):
        """2 + 2 patrimi ar face o masura, dar `|:` spune ca e anacruza unei repetitii.

        MuseScore deseneaza `|:` dupa cheia sistemului urmator, deci tokenul vine
        primul in documentul acela, nu ultimul in cel dinainte.
        """
        after = CUT_TREBLE_SECOND.replace("*M4/4\n=\n", "*M4/4\n=!|:\n")
        after_bass = CUT_BASS_SECOND.replace("*M4/4\n=\n", "*M4/4\n=!|:\n")
        page = assemble_page([CUT_TREBLE_FIRST, CUT_BASS_FIRST, after, after_bass])
        assert not any("taiata" in problem for problem in page.problems)
        assert [m.number for m in page.score.part("S").measures if m.repeat_start] == [3]

    def test_fara_semn_masura_se_lipeste_ca_inainte(self):
        """Martorul testului de mai sus: aceleasi bucati, fara `|:`, se lipesc."""
        page = assemble_page([CUT_TREBLE_FIRST, CUT_BASS_FIRST, CUT_TREBLE_SECOND, CUT_BASS_SECOND])
        assert any("taiata" in problem for problem in page.problems)
