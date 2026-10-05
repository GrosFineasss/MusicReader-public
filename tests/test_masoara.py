"""Agregarea masuratorilor pe mai multe pagini si maini (§15.21).

Testele nu incarca modelul: partea in care se poate gresi tacut e **adunarea**,
nu transcrierea. O medie de rate in loc de rata totalului da alt numar, iar
diferenta nu se vede cu ochiul — se vede doar cand cineva compara doua rulari.
"""

import pytest

from scripts.masoara import aduna


def pagina(ref_events, matched, missing=0, extra=0, pitch_wrong=0, masuri=(0, 0),
           ref_rests=0, hyp_rests=0, matched_rests=0):
    """O masuratoare de pagina, cu doar campurile care conteaza la adunare.

    Fara pauze, `note` si `evenimente` coincid — de aceea testele scrise inainte
    de corectia din §15.57 raman valabile neatinse.
    """
    return {
        "ref_rests": ref_rests,
        "hyp_rests": hyp_rests,
        "matched_rests": matched_rests,
        "ref_events": ref_events,
        "hyp_events": matched + extra,
        "matched": matched,
        "pitch_wrong": pitch_wrong,
        "duration_wrong": 0,
        "both_wrong": 0,
        "missing": missing,
        "extra": extra,
        "ref_measures": masuri[0],
        "hyp_measures": masuri[0],
        "measures_exact": masuri[1],
    }


class TestAdunarea:
    def test_componentele_se_aduna(self):
        total = aduna([pagina(100, 40), pagina(50, 10)])
        assert total["ref_events"] == 150
        assert total["matched"] == 50
        assert total["pagini"] == 2

    def test_acuratetea_e_pe_totalul_notelor(self):
        # 40/100 si 10/50 dau 50/150 = 33,3%, nu media (40% + 20%)/2 = 30%.
        total = aduna([pagina(100, 40), pagina(50, 10)])
        assert total["acuratete_note"] == pytest.approx(50 / 150)

    def test_o_pagina_lunga_cantareste_mai_mult(self):
        """Miezul: media de rate ar face o pagina de 10 note cat una de 1000.

        Cu media, o pagina scurta perfecta ar ascunde o pagina lunga proasta.
        """
        multa = pagina(1000, 100)   # 10%
        putina = pagina(10, 10)     # 100%
        total = aduna([multa, putina])
        media_ratelor = (100 / 1000 + 10 / 10) / 2
        assert total["acuratete_note"] == pytest.approx(110 / 1010)
        assert total["acuratete_note"] < media_ratelor / 2  # mult sub media

    def test_ser_se_recalculeaza_din_componente(self):
        # SER = (substituiri + lipsa + in plus) / note referinta.
        total = aduna([pagina(100, 80, missing=10, extra=5, pitch_wrong=10)])
        assert total["ser"] == pytest.approx((10 + 10 + 5) / 100)

    def test_ser_pe_doua_pagini(self):
        total = aduna([
            pagina(100, 90, missing=10),
            pagina(100, 50, missing=50, extra=20),
        ])
        assert total["ser"] == pytest.approx((10 + 50 + 20) / 200)

    def test_o_singura_pagina_da_exact_cifrele_ei(self):
        # Harnessul nou trebuie sa fie compatibil cu reperul inghetat, care e
        # masurat pe o singura pagina.
        una = pagina(182, 16, missing=100, extra=50, pitch_wrong=30)
        total = aduna([una])
        assert total["matched"] == 16 and total["ref_events"] == 182
        assert total["ser"] == pytest.approx((30 + 100 + 50) / 182)

    def test_un_set_gol_nu_imparte_la_zero(self):
        total = aduna([])
        assert total["pagini"] == 0 and total["acuratete_note"] == 0


class TestPauzeleNuSeNumaraCaNote:
    """Corectia §15.42, ajunsa si in unealta de mai multe pagini (§15.57).

    Doua pauze de aceeasi durata ies „identice" — niciuna n-are inaltime — deci
    o pereche pauza-pauza intra la `matched` fara sa insemne ca s-a citit vreo
    nota. `evaluate.py` a fost corectat atunci; `masoara.py` a ramas cu metrica
    veche, si a raportat ani la rand evenimente sub eticheta „note".
    """

    def test_pauzele_ies_din_numitor(self):
        total = aduna([pagina(100, 40, ref_rests=40, matched_rests=20)])
        assert total["ref_note"] == 60

    def test_pauzele_ies_din_numarator(self):
        total = aduna([pagina(100, 40, ref_rests=40, matched_rests=20)])
        assert total["corecte_note"] == 20

    def test_acuratetea_e_pe_note_nu_pe_evenimente(self):
        """40/100 = 40% pe evenimente, dar 20/60 = 33% pe note."""
        total = aduna([pagina(100, 40, ref_rests=40, matched_rests=20)])
        assert total["acuratete_note"] == pytest.approx(20 / 60)

    def test_o_pagina_numai_de_pauze_da_zero_nu_o_suta(self):
        """Cazul care facea cifra sa minta cel mai tare."""
        total = aduna([pagina(50, 50, ref_rests=50, matched_rests=50)])
        assert total["corecte_note"] == 0
        assert total["acuratete_note"] == 0

    def test_precizia_se_raporteaza_pe_notele_scrise(self):
        total = aduna([pagina(100, 40, extra=20, ref_rests=40,
                              hyp_rests=30, matched_rests=20)])
        assert total["hyp_note"] == 30  # hyp_events 40+20=60, minus 30 de pauze
        assert total["precizie_note"] == pytest.approx(20 / total["hyp_note"])

    def test_ser_ramane_pe_evenimente(self):
        """SER se compara cu literatura, deci pastreaza pauzele (§15.44)."""
        total = aduna([pagina(100, 80, missing=10, extra=5, pitch_wrong=10,
                              ref_rests=40, matched_rests=30)])
        assert total["ser"] == pytest.approx((10 + 10 + 5) / 100)


class TestAsezarea:
    """`asezare` din set ajunge la motor ca lista de asezari, in orice forma ar fi scrisa (2026-10-05)."""

    UNA = [["X1"], ["S", "A"], ["T", "B"]]

    def test_o_singura_asezare_se_impacheteaza(self):
        from scripts.masoara import asezare_din

        assert asezare_din({"asezare": self.UNA}) == [self.UNA]

    def test_textul_de_la_linia_de_comanda(self):
        from scripts.masoara import asezare_din

        assert asezare_din({"asezare": "X1/S,A/T,B"}) == [self.UNA]

    def test_cate_una_pe_sistem_ramane_neatinsa(self):
        from scripts.masoara import asezare_din

        pe_sisteme = [[["S", "A"], ["T", "B"]], self.UNA]
        assert asezare_din({"asezare": pe_sisteme}) == pe_sisteme

    def test_fara_asezare(self):
        from scripts.masoara import asezare_din

        assert asezare_din({}) is None
