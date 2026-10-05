"""Fereastra de corectare: citirea celulelor si drumul spre antrenare (§15.24).

Widgetul in sine se vede cu ochiul. Ce se testeaza aici sunt cele doua locuri in
care se poate pierde munca de corectare fara ca cineva sa observe:

- **traducerea celula <-> muzica** — o inaltime scrisa `F#3` care se citeste
  gresit strica nota in tacere, fiindca tabelul o va afisa oricum cum a inteles-o;
- **salvarea** — o corectie care nu supravietuieste drumului MusicXML inseamna ca
  omul a corectat degeaba, si asta se descopera abia la antrenare.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from app.correction.editare import evenimente, eveniment_la  # noqa: E402
from app.music_model.model import Duration, NoteType, Pitch  # noqa: E402
from app.ui.corectare import (  # noqa: E402
    citeste_durata,
    citeste_inaltimea,
    numele_inaltimii,
    salveaza_corectat,
)
from scripts.make_reference_score import build  # noqa: E402


class TestInaltimile:
    @pytest.mark.parametrize("treapta", list("CDEFGAB"))
    @pytest.mark.parametrize("alter", [-1, 0, 1])
    @pytest.mark.parametrize("octava", [2, 4, 6])
    def test_dus_intors(self, treapta, alter, octava):
        pitch = Pitch(step=treapta, octave=octava, alter=alter)
        assert citeste_inaltimea(numele_inaltimii(pitch)) == pitch

    def test_se_accepta_si_scris_cu_litera_mica(self):
        assert citeste_inaltimea("f#3") == Pitch(step="F", octave=3, alter=1)

    @pytest.mark.parametrize(
        "text, cuvant",
        [("", "inaltimea"), ("H4", "treapta"), ("C$4", "alteratia"), ("C#x", "octava")],
    )
    def test_mesajul_spune_ce_n_a_mers(self, text, cuvant):
        # Cine corecteaza o suta de note nu are chef sa ghiceasca formatul.
        with pytest.raises(ValueError, match=cuvant):
            citeste_inaltimea(text)


class TestDuratele:
    def test_o_durata_obisnuita(self):
        assert citeste_durata("quarter", "1") == Duration(NoteType.QUARTER, dots=1)

    def test_o_durata_inexistenta_enumera_ce_se_poate(self):
        with pytest.raises(ValueError, match="quarter"):
            citeste_durata("sfert", "0")

    @pytest.mark.parametrize("puncte", ["3", "-1", "x", ""])
    def test_puncte_imposibile(self, puncte):
        with pytest.raises(ValueError, match="punctele"):
            citeste_durata("quarter", puncte)


class TestSalvarea:
    def test_scrie_alaturi_fara_sa_stearga_originalul(self, tmp_path):
        """Diferenta dintre ce a citit modelul si ce era de fapt e chiar
        informatia din care se invata. Cine suprascrie originalul o pierde."""
        original = tmp_path / "pagina.musicxml"
        original.write_text("<!-- iesirea modelului -->", encoding="utf-8")

        cale = salveaza_corectat(build(), original)

        assert cale != original
        assert original.read_text(encoding="utf-8") == "<!-- iesirea modelului -->"
        assert cale.parent == original.parent and "corectat" in cale.name

    def test_corectia_supravietuieste_drumului_spre_antrenare(self, tmp_path):
        """Miezul: corectat -> MusicXML -> citit inapoi -> aceeasi corectie.

        Asta e exact drumul pe care `scripts/dataset.py` transforma pagina
        corectata in perechi de antrenare. Daca schimbarea se pierde aici, omul a
        corectat degeaba si n-ar afla decat mult mai tarziu.
        """
        from app.correction.editare import schimba_inaltimea
        from app.evaluation.reference import load

        score = build()
        adresa = evenimente(score)[0][0]
        tinta = Pitch(step="G", octave=4)
        corectat = schimba_inaltimea(score, adresa, tinta)

        cale = salveaza_corectat(corectat, tmp_path / "pagina.musicxml")
        recitit = load(cale)

        assert eveniment_la(recitit, adresa).pitch == tinta

    def test_si_o_durata_schimbata_supravietuieste(self, tmp_path):
        from app.correction.editare import schimba_durata
        from app.evaluation.reference import load

        score = build()
        adresa = evenimente(score)[0][0]
        corectat = schimba_durata(score, adresa, Duration(NoteType.EIGHTH))

        recitit = load(salveaza_corectat(corectat, tmp_path / "p.musicxml"))
        assert eveniment_la(recitit, adresa).duration.note_type is NoteType.EIGHTH


class TestFereastra:
    def test_se_deschide_si_arata_toate_notele(self, tmp_path, qt_app):
        """Constructia ferestrei pe o transcriere adevarata, fara ecran."""
        from app.ui.corectare import FereastraCorectare

        musicxml = salveaza_corectat(build(), tmp_path / "pagina.musicxml")
        imagine = tmp_path / "pagina.png"
        imagine.write_bytes(b"")  # poza lipsa nu trebuie sa impiedice corectarea

        fereastra = FereastraCorectare(imagine, musicxml)
        assert fereastra.tabel.rowCount() == len(evenimente(fereastra.istoric.curenta))
        assert fereastra.istoric.modificari == 0


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])
