"""Corectarea transcrierii (etapa 12, §15.24).

Ce se testeaza aici e **drumul datelor**, nu fereastra. Widgetul se vede cu
ochiul; ce se strica tacut e o corectie care pare aplicata si nu e, sau una care
nu supravietuieste drumului spre materialul de antrenare — si atunci munca de
corectare se pierde fara ca cineva sa observe.

Regula pe care se sprijina tot restul: **fiecare modificare intoarce o partitura
noua**, iar cea primita ramane neatinsa. Daca s-ar modifica pe loc, istoricul de
„inapoi" ar arata versiuni care intre timp s-au schimbat si ele.
"""

from itertools import groupby

import pytest

from app.correction.editare import (
    Adresa,
    Istoric,
    catre_nota,
    catre_pauza,
    evenimente,
    eveniment_la,
    insereaza,
    probleme,
    schimba_durata,
    schimba_inaltimea,
    sterge,
)
from app.music_model.model import Duration, Note, NoteType, Pitch, Rest
from app.validation.rules import Severity
from scripts.make_reference_score import build

Q = Duration(NoteType.QUARTER)
H = Duration(NoteType.HALF)


@pytest.fixture
def score():
    return build()


@pytest.fixture
def prima(score):
    """Adresa primului eveniment — punctul de plecare al majoritatii probelor."""
    return evenimente(score)[0][0]


class TestAdresele:
    def test_ordinea_e_cea_de_citire(self, score):
        """Voce cu voce, iar in fiecare voce masurile si notele in ordine.

        **Nu** ordinea alfabetica a vocilor: partitura are S, A, T, B in ordinea
        asta, iar sortarea dupa nume ar da A, B, S, T. De asta `Adresa` nu e
        ordonabila — vezi comentariul din `editare.py`.
        """
        adrese = [a for a, _ in evenimente(score)]
        blocuri = [voce for voce, _ in groupby(a.part_id for a in adrese)]
        assert blocuri == [p.id for p in score.parts]
        for part in score.parts:
            ale_ei = [(a.measure, a.index) for a in adrese if a.part_id == part.id]
            assert ale_ei == sorted(ale_ei)

    def test_fiecare_adresa_gaseste_evenimentul_ei(self, score):
        for adresa, event in evenimente(score):
            assert eveniment_la(score, adresa) is event

    def test_o_adresa_inexistenta_spune_unde(self, score, prima):
        gresita = Adresa(prima.part_id, prima.measure, 999)
        with pytest.raises(KeyError, match="#999"):
            eveniment_la(score, gresita)


class TestModificarile:
    def test_partitura_primita_ramane_neatinsa(self, score, prima):
        inainte = eveniment_la(score, prima)
        noua = schimba_durata(score, prima, H)
        assert eveniment_la(score, prima) is inainte, "s-a modificat originalul"
        assert eveniment_la(noua, prima).duration == H

    def test_inaltimea_se_schimba_iar_durata_ramane(self, score, prima):
        vechi = eveniment_la(score, prima)
        tinta = Pitch(step="G", octave=4)
        noua = schimba_inaltimea(score, prima, tinta)
        schimbat = eveniment_la(noua, prima)
        assert schimbat.pitch == tinta
        assert schimbat.duration == vechi.duration

    def test_o_pauza_n_are_inaltime(self, score, prima):
        cu_pauza = catre_pauza(score, prima)
        with pytest.raises(TypeError, match="pauza"):
            schimba_inaltimea(cu_pauza, prima, Pitch(step="G", octave=4))

    def test_nota_devine_pauza_pastrand_durata(self, score, prima):
        durata = eveniment_la(score, prima).duration
        noua = catre_pauza(score, prima)
        schimbat = eveniment_la(noua, prima)
        assert isinstance(schimbat, Rest) and schimbat.duration == durata

    def test_pauza_devine_nota_pastrand_durata(self, score, prima):
        durata = eveniment_la(score, prima).duration
        cu_pauza = catre_pauza(score, prima)
        inapoi = catre_nota(cu_pauza, prima, Pitch(step="A", octave=4))
        schimbat = eveniment_la(inapoi, prima)
        assert isinstance(schimbat, Note) and schimbat.duration == durata

    def test_stergerea_scoate_exact_un_eveniment(self, score, prima):
        noua = sterge(score, prima)
        assert len(evenimente(noua)) == len(evenimente(score)) - 1

    def test_inserarea_pune_inaintea_pozitiei(self, score, prima):
        vechi = eveniment_la(score, prima)
        noua = insereaza(score, prima, Rest(duration=Q))
        assert isinstance(eveniment_la(noua, prima), Rest)
        urmatorul = Adresa(prima.part_id, prima.measure, prima.index + 1)
        assert eveniment_la(noua, urmatorul).duration == vechi.duration


class TestProblemele:
    def test_o_partitura_curata_nu_are_erori(self, score):
        toate = [f for lista in probleme(score).values() for f in lista]
        assert [f for f in toate if f.severity is Severity.ERROR] == []

    def test_o_stergere_face_masura_scurta(self, score):
        """Dintr-o masura din mijloc, nu din prima.

        O **prima** masura scurta e anacruza si se raporteaza doar ca
        avertisment (§15.2) — deci stergerea de acolo n-ar fi dovedit nimic.
        """
        adrese = [a for a, _ in evenimente(score) if a.measure > 1]
        gasite = probleme(sterge(score, adrese[0]))
        toate = [f for lista in gasite.values() for f in lista]
        assert any(f.severity is Severity.ERROR for f in toate)

    def test_constatarile_se_leaga_de_adrese(self, score, prima):
        """Miezul: fereastra trebuie sa poata arata problema **pe nota**.

        `Finding` si `Adresa` au dinadins aceleasi campuri, deci legatura se face
        fara traducere. Daca cineva schimba unul din ele, testul cade.
        """
        stricat = schimba_durata(score, prima, Duration(NoteType.WHOLE))
        gasite = probleme(stricat)
        cu_adresa = [a for a in gasite if a is not None]
        assert all(
            eveniment_la(stricat, a) is not None for a in cu_adresa
        ), "o constatare arata spre un eveniment care nu exista"


class TestIstoricul:
    def test_la_inceput_nu_se_poate_inapoi(self, score):
        istoric = Istoric(score)
        assert not istoric.se_poate_inapoi and istoric.modificari == 0

    def test_inapoi_si_inainte(self, score, prima):
        istoric = Istoric(score)
        istoric.aplica(schimba_durata(score, prima, H))
        assert istoric.modificari == 1
        assert eveniment_la(istoric.curenta, prima).duration == H

        istoric.inapoi()
        assert eveniment_la(istoric.curenta, prima).duration != H
        istoric.inainte()
        assert eveniment_la(istoric.curenta, prima).duration == H

    def test_o_modificare_dupa_inapoi_taie_ce_era_inainte(self, score, prima):
        istoric = Istoric(score)
        istoric.aplica(schimba_durata(score, prima, H))
        istoric.inapoi()
        istoric.aplica(catre_pauza(istoric.curenta, prima))
        assert not istoric.se_poate_inainte
        assert isinstance(eveniment_la(istoric.curenta, prima), Rest)


class TestMarginile:
    """Doar inserarea are voie la pozitia de dupa ultimul eveniment."""

    def test_inserarea_la_coada_merge(self, score, prima):
        cate = len(score.part(prima.part_id).measure(prima.measure).events)
        coada = Adresa(prima.part_id, prima.measure, cate)
        noua = insereaza(score, coada, Rest(duration=Q))
        assert eveniment_la(noua, coada).duration == Q

    @pytest.mark.parametrize(
        "operatie",
        [
            lambda s, a: sterge(s, a),
            lambda s, a: catre_pauza(s, a),
            lambda s, a: schimba_durata(s, a, H),
        ],
    )
    def test_restul_refuza_pozitia_de_dupa_capat(self, score, prima, operatie):
        # Fara verificare stricta, ar fi cazut mai incolo cu `IndexError` — fara
        # sa spuna nici ce adresa, nici ce masura.
        cate = len(score.part(prima.part_id).measure(prima.measure).events)
        coada = Adresa(prima.part_id, prima.measure, cate)
        with pytest.raises(KeyError, match=f"#{cate}"):
            operatie(score, coada)
