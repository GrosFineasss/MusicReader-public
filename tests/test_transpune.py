"""Transpozitia referintelor, si cele trei straji ale ei (§15.23, §15.29).

De ce exista scriptul: aveam 13 sisteme de antrenat, si atat. Randand fiecare in
mai multe variante de stil ies mai multe imagini, dar **aceeasi muzica**. Urcata
cu doua semitonuri, e alta tinta, la fel de valida.

De ce are straji, si de ce sunt testate: transpozitia cromatica poate da o
partitura corecta ca sunet si stricata ca notatie. Masurat pe referintele reale,
mutarea cu **un semiton** a produs o armura de 8 diezi la una si alteratii care
nu se mai potrivesc cu armura la cealalta. Amandoua ar fi intrat tacut in datele
de antrenare, iar modelul ar fi invatat notatie gresita cu incredere deplina.
"""

from pathlib import Path

import pytest

from scripts.transpune import (
    MAX_MIDI,
    MIN_MIDI,
    ambitus,
    erori_de_notatie,
    ritm,
    scrie,
    transpune,
)

REFERINTA = Path("data/validation/pagina3.musicxml")


def _referinta() -> Path:
    """Referinta paginii 3; fisier local, deci testul se sare intr-o copie fara date."""
    if not REFERINTA.exists():
        pytest.skip("lipseste data/validation/pagina3.musicxml (date locale, nu se publica)")
    return REFERINTA


class TestAmbitus:
    def test_marginile_sunt_cele_de_cor(self):
        # Mi2 - La5, confirmat de utilizator pentru muzica de cor.
        assert (MIN_MIDI, MAX_MIDI) == (40, 81)

    def test_se_ia_din_note_nu_din_altceva(self):
        from music21 import note, stream

        s = stream.Score()
        p = stream.Part()
        for inaltime in ("C4", "G5", "E3"):
            p.append(note.Note(inaltime))
        s.append(p)
        assert ambitus(s) == (52, 79)  # E3 = 52, G5 = 79

    def test_o_partitura_fara_note_se_plange(self):
        from music21 import stream

        with pytest.raises(ValueError, match="nicio nota"):
            ambitus(stream.Score())


class TestTranspozitia:
    @pytest.mark.parametrize("semitonuri", [-3, -2, 2, 3])
    def test_muta_fiecare_nota_cu_exact_atat(self, semitonuri):
        _, (jos, sus) = transpune(_referinta(), 0)
        _, (jos_nou, sus_nou) = transpune(_referinta(), semitonuri)
        assert (jos_nou - jos, sus_nou - sus) == (semitonuri, semitonuri)


class TestStrajaDeNotatie:
    def test_un_fisier_care_nu_se_citeste_da_minus_unu(self, tmp_path):
        stricat = tmp_path / "stricat.musicxml"
        stricat.write_text("<score-partwise></score-partwise>", encoding="utf-8")
        erori, mesaj = erori_de_notatie(stricat)
        assert erori == -1 and mesaj

    def test_referinta_buna_n_are_erori(self):
        erori, _ = erori_de_notatie(_referinta())
        assert erori == 0

    @pytest.mark.slow
    def test_cazul_care_a_pornit_straja(self, tmp_path):
        """Un semiton in sus strica notatia — cazul masurat, tinut ca regresie.

        Daca `music21` invata candva sa aleaga scrierea enarmonica, testul asta
        cade, si atunci straja poate fi relaxata. Pana atunci, e dovada ca e
        nevoie de ea.
        """
        import warnings

        mutat, _ = transpune(_referinta(), 1)
        iesire = tmp_path / "plus-unu.musicxml"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mutat.write("musicxml", fp=str(iesire))

        erori, prima = erori_de_notatie(iesire)
        assert erori > 0, "transpozitia cu +1 nu mai strica notatia — straja se poate relaxa"
        assert "alter" in prima or "armura" in prima


class TestRitm:
    """A treia straja: transpozitia muta inaltimi, si nimic altceva."""

    @pytest.mark.slow
    def test_scrierea_pastreaza_ritmul(self, tmp_path):
        """Ce trebuie sa fie adevarat dupa reparatie, pe drumul real de scriere."""
        mutat, _ = transpune(_referinta(), 2)
        iesire = tmp_path / "plus-doi.musicxml"
        scrie(mutat, iesire)

        assert ritm(iesire) == ritm(_referinta())

    @pytest.mark.slow
    def test_scrierea_repara_totusi_alteratiile(self, tmp_path):
        """Ritmul neatins nu trebuie platit cu notatie stricata (§15.33).

        Oprind `makeNotation` de tot se opreste si partea care chiar trebuia:
        transpozitia cromatica lasa note a caror alteratie nu se mai potriveste
        cu armura. Masurat pe +1, cazul cel mai rau: 56 de erori fara nimic, 2
        cu `makeAccidentals` chemata pe fiecare parte.

        Testul nu cere zero — +1 ramane un semiton care strica notatia (§15.23)
        — ci cere ca reparatia sa se fi facut: sub zece erori in loc de zeci.
        """
        mutat, _ = transpune(_referinta(), 1)
        iesire = tmp_path / "plus-unu.musicxml"
        scrie(mutat, iesire)

        erori, _ = erori_de_notatie(iesire)
        assert erori < 10, f"alteratiile nu mai sunt reparate: {erori} erori"
        assert ritm(iesire) == ritm(_referinta())

    @pytest.mark.slow
    def test_scrierea_implicita_rescrie_ritmul(self, tmp_path):
        """Cazul care a pornit straja a treia, tinut ca regresie (§15.29).

        `music21` isi „repara" notatia la scriere: contopeste pauze alaturate
        (`4r 8r 16r` -> `4..r`, care nici macar nu e in vocabularul modelului)
        si puncteaza note care in original nu erau. Masurat: **toate** cele opt
        transpozitii ale celor doua referinte au iesit cu alt ritm.

        Daca `music21` invata candva sa scrie fara sa repare, testul asta cade,
        si atunci `makeNotation=False` devine de prisos. Pana atunci, e dovada
        ca 30% din corpusul de antrenare invata alt ritm decat arata pagina.
        """
        import warnings

        mutat, _ = transpune(_referinta(), 2)
        iesire = tmp_path / "plus-doi-implicit.musicxml"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mutat.write("musicxml", fp=str(iesire))

        assert ritm(iesire) != ritm(_referinta()), (
            "music21 nu mai rescrie ritmul la scriere — `makeNotation=False` "
            "si straja de ritm pot fi relaxate"
        )
