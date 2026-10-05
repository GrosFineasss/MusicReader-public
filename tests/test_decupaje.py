"""Decuparea paginii pentru citit cu ochiul (scripts/decupaje.py).

Partile care se testeaza sunt cele care pot taia gresit fara sa se vada:
felierea orizontala si completarea portativelor pierdute de proiectie.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.preprocessing.staff_lines import Band  # noqa: E402
from scripts.decupaje import complete_bands, rotate_grey, slice_columns  # noqa: E402


def banda(top: int, spacing: int = 24) -> Band:
    lines = tuple(top + i * spacing for i in range(5))
    return Band(top=top, bottom=lines[-1], lines=lines)


class TestSliceColumns:
    def test_pagina_mai_ingusta_decat_felia_ramane_intreaga(self):
        assert slice_columns(900, 1200, 0.15) == [(0, 900)]

    def test_feliile_acopera_toata_latimea(self):
        felii = slice_columns(2397, 1200, 0.15)
        assert felii[0][0] == 0
        assert felii[-1][1] == 2397

    def test_feliile_vecine_se_suprapun(self):
        felii = slice_columns(2397, 1200, 0.15)
        for (_, sfarsit), (inceput, _) in zip(felii, felii[1:]):
            assert inceput < sfarsit, "o nota de la margine ar cadea intre felii"

    def test_suprapunerea_zero_da_felii_lipite(self):
        felii = slice_columns(2400, 1200, 0.0)
        assert felii == [(0, 1200), (1200, 2400)]


class TestCompleteBands:
    def test_fara_asteptare_nu_se_adauga_nimic(self):
        bands = [banda(100), banda(344)]
        assert complete_bands(bands, 0) == (bands, [])

    def test_un_portativ_lipsa_se_adauga_la_pasul_median(self):
        bands = [banda(100), banda(344), banda(588)]
        completate, adaugate = complete_bands(bands, 4)
        assert adaugate == [3]
        assert completate[3].top == 832
        assert len(completate[3].lines) == 5

    def test_banda_adaugata_pastreaza_forma_ultimei(self):
        bands = [banda(100), banda(344)]
        completate, _ = complete_bands(bands, 3)
        inaltimi = [b.bottom - b.top for b in completate]
        assert len(set(inaltimi)) == 1

    def test_nu_se_adauga_cand_sunt_destule(self):
        bands = [banda(100), banda(344), banda(588)]
        completate, adaugate = complete_bands(bands, 2)
        assert adaugate == []
        assert len(completate) == 3

    def test_o_singura_banda_nu_da_pas_deci_nu_se_extrapoleaza(self):
        bands = [banda(100)]
        assert complete_bands(bands, 4) == (bands, [])


class TestRotateGrey:
    def test_marginea_ramane_hartie_nu_cerneala(self):
        image = np.full((200, 200), 255, dtype=np.uint8)
        rotit = rotate_grey(image, 10.0)
        assert rotit[0, 0] > 200, "coltul umplut cu negru ar inghiti notele de la capat"

    def test_unghi_zero_lasa_imaginea_neschimbata(self):
        image = np.random.default_rng(0).integers(0, 255, (50, 50), dtype=np.uint8)
        assert np.array_equal(rotate_grey(image, 0.0), image)


@pytest.mark.parametrize("latime", [1, 1199, 1200, 1201, 5000])
def test_felierea_nu_iese_niciodata_din_pagina(latime):
    for inceput, sfarsit in slice_columns(latime, 1200, 0.15):
        assert 0 <= inceput < sfarsit <= latime


class TestInaltimi:
    """Numele inaltimilor, de sus in jos, pornind de la linia de sus a portativului.

    Cheia spune doar de unde se numara: pe cheia Sol linia de sus e F5, pe cheia
    Fa e A3. Restul e alfabetul muzical. Testat fiindca o grila decalata cu o
    treapta face **toata pagina** sa se citeasca gresit, coerent cu ea insasi si
    fara sa pice niciun alt test — exact defectul din §15.43.
    """

    def pozitii(self, cheie, sus=0, jos=0):
        from scripts.decupaje import inaltimi

        return dict((nume, pas) for nume, pas in inaltimi(cheie, sus, jos))

    def test_liniile_cheii_sol(self):
        p = self.pozitii("G")
        assert [p[n] for n in ("F5", "D5", "B4", "G4", "E4")] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_spatiile_cheii_sol(self):
        p = self.pozitii("G")
        assert [p[n] for n in ("E5", "C5", "A4", "F4")] == [0.5, 1.5, 2.5, 3.5]

    def test_liniile_cheii_fa(self):
        p = self.pozitii("F")
        assert [p[n] for n in ("A3", "F3", "D3", "B2", "G2")] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_octava_scade_la_trecerea_peste_do(self):
        p = self.pozitii("G")
        assert p["C5"] == 1.5 and p["B4"] == 2.0, "sub do se schimba octava"

    def test_treptele_de_deasupra_au_pozitii_negative(self):
        from scripts.decupaje import inaltimi

        peste = [nume for nume, pas in inaltimi("G", 4, 0) if pas < 0]
        assert peste == ["C6", "B5", "A5", "G5"]

    def test_treptele_de_dedesubt_continua_sub_portativ(self):
        p = self.pozitii("G", 0, 4)
        assert p["C4"] == 5.0 and p["A3"] == 6.0


class TestParseChei:
    def test_o_litera_inseamna_la_fel_peste_tot(self):
        from scripts.decupaje import parse_chei

        chei = parse_chei("G", 2)
        assert chei[0] == "G" and chei[17] == "G"

    def test_tiparul_cu_slash_se_repeta_pe_sisteme(self):
        from scripts.decupaje import parse_chei

        chei = parse_chei("G/F", 2)
        assert chei[:6] == ["G", "F", "G", "F", "G", "F"]

    def test_lista_completa_se_ia_ca_atare(self):
        from scripts.decupaje import parse_chei

        assert parse_chei("G,G,F", 3)[:3] == ["G", "G", "F"]

    def test_fara_optiune_nu_se_deseneaza_grila(self):
        from scripts.decupaje import parse_chei

        assert parse_chei(None, 2) is None
        assert parse_chei("", 2) is None


class TestIncruciseaza:
    """Barele gasite pe **toate** portativele, fata de cele gasite pe unul.

    O codita lunga umple si ea o coloana, dar numai pe portativul ei; o bara e
    trasa pe amandoua. Ce apare pe unul singur nu se arunca — bara anacruzei de
    pe pagina 3 e trasa doar pe portativul de jos — ci se raporteaza separat.
    """

    def test_ce_apare_pe_amandoua_e_bara(self):
        from scripts.decupaje import _incruciseaza

        sigure, singure = _incruciseaza([[100, 500], [102, 498]])
        assert sigure == [101, 499] and singure == []

    def test_ce_apare_pe_unul_singur_se_raporteaza_separat(self):
        from scripts.decupaje import _incruciseaza

        sigure, singure = _incruciseaza([[100, 300], [102]])
        assert sigure == [101] and singure == [300]

    def test_barele_apropiate_se_unesc(self):
        """Bara de repetitie e groasa+subtire+puncte, numarata de trei ori."""
        from scripts.decupaje import _incruciseaza

        sigure, _ = _incruciseaza([[2026, 2040, 2057], [2030, 2044, 2061]])
        assert len(sigure) == 1, "aceeasi bara, nu trei"

    def test_un_singur_portativ_nu_are_cu_ce_incrucisa(self):
        from scripts.decupaje import _incruciseaza

        assert _incruciseaza([[10, 20]]) == ([10, 20], [])


class TestBarlines:
    """Detectia pe o imagine construita, unde stim ce am desenat."""

    def portativ(self, latime=600, spacing=20):
        import numpy as np

        from app.preprocessing.staff_lines import Band

        inalt = 200
        binary = np.zeros((inalt, latime), dtype=np.uint8)
        top = 50
        linii = tuple(top + i * spacing for i in range(5))
        for y in linii:
            binary[y, :] = 255
        return binary, Band(top=linii[0], bottom=linii[-1], lines=linii)

    def test_o_coloana_plina_e_bara(self):
        from scripts.decupaje import barlines

        binary, band = self.portativ()
        binary[band.top : band.bottom + 1, 300] = 255
        assert barlines(binary, band) == [300]

    def test_o_codita_de_jumatate_de_portativ_nu_e_bara(self):
        from scripts.decupaje import barlines

        binary, band = self.portativ()
        mijloc = (band.top + band.bottom) // 2
        binary[band.top : mijloc, 300] = 255
        assert barlines(binary, band) == []

    def test_o_bara_groasa_da_o_singura_pozitie(self):
        from scripts.decupaje import barlines

        binary, band = self.portativ()
        binary[band.top : band.bottom + 1, 300:305] = 255
        assert barlines(binary, band) == [302]

    def test_pragul_coborat_prinde_bara_trasa_scurt(self):
        """Pe pagina 1 barele sunt trase scurt si nu ajung la 0,95."""
        from scripts.decupaje import barlines

        binary, band = self.portativ()
        inalt = band.bottom - band.top
        binary[band.top + inalt // 8 : band.bottom, 300] = 255
        assert barlines(binary, band, 0.95) == []
        assert barlines(binary, band, 0.80) == [300]
