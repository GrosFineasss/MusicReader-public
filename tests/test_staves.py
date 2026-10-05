"""Gruparea portativelor in sisteme.

Partea in care se poate gresi tacut: daca gruparea da 3 sisteme in loc de 4,
`smt-grandstaff` primeste imagini care nu seamana cu ce a vazut la antrenare si
transcrierea iese proasta din motive care n-au legatura cu modelul.
"""

import pytest

from app.staff_detection.staves import (
    bounding_box,
    group,
    group_by_gap,
    group_fixed,
    parse_grouping,
)

# (x1, y1, x2, y2) — doua sisteme de cate doua portative, cu spatiu mare intre
# ele si spatiu mic in interiorul lor.
PAGE = [
    (10, 100, 390, 150),
    (10, 158, 390, 208),
    (10, 300, 390, 350),
    (10, 358, 390, 408),
]


class TestGrupare:
    def test_doua_sisteme_de_cate_doua(self):
        assert [len(system) for system in group_by_gap(PAGE)] == [2, 2]

    def test_un_sistem_poate_avea_un_singur_portativ(self):
        # Primul sistem din „Iata-L pe Hristos" are doar S&A: T&B intra la
        # masura 6. O grupare „cate doua" ar lipi jumatati din sisteme diferite.
        page = [(10, 100, 390, 150)] + PAGE[2:]
        assert [len(system) for system in group_by_gap(page)] == [1, 2]

    def test_pragul_se_poate_schimba(self):
        # Cu prag zero, fiecare portativ e sistemul lui.
        assert len(group_by_gap(PAGE, gap=0)) == 4

    def test_pagina_goala_da_zero_sisteme(self):
        assert group_by_gap([]) == []


# Cutiile de pe pagina 1 a manuscrisului din §14.5, asa cum le-a dat detectorul.
# Sunt umflate de versuri: un portativ a carui intindere reala de linii e ~100 px
# primeste o cutie de 222-285 px, deci cutiile aproape se ating. Pagina are 6
# sisteme de cate doua portative.
MANUSCRIS = [
    (102, 90, 2397, 373),
    (102, 359, 2397, 591),
    (100, 607, 2388, 829),
    (93, 831, 2388, 1065),
    (77, 1067, 2380, 1291),
    (86, 1282, 2380, 1551),
    (86, 1532, 2379, 1769),
    (63, 1740, 2380, 2018),
    (51, 2007, 2372, 2241),
    (46, 2227, 2377, 2504),
    (51, 2501, 2371, 2731),
    (48, 2723, 2364, 3008),
]


class TestSpatiereaNuAjungePeScrisDeMana:
    """Cazul real care a rupt conducta — pastrat ca test de regresie (§9)."""

    def test_gruparea_dupa_spatiu_lipeste_zece_portative(self):
        # Nu e o problema de calibrare a pragului: spatiile din interiorul unui
        # sistem si cele dintre sisteme se suprapun complet.
        assert [len(s) for s in group_by_gap(MANUSCRIS)] == [2, 10]

    def test_niciun_prag_nu_separa_sistemele(self):
        # Daca ar exista un prag bun, undeva ar iesi sase sisteme de cate doua.
        assert not any(
            [len(s) for s in group_by_gap(MANUSCRIS, gap=g)] == [2] * 6
            for g in range(-40, 60)
        )

    def test_declaratia_da_sase_sisteme_de_cate_doua(self):
        assert [len(s) for s in group_fixed(MANUSCRIS, 2)] == [2] * 6


class TestGrupareDeclarata:
    def test_restul_devine_sistemul_lui_nu_se_pierde(self):
        # Cinci portative declarate cate doua: ultimul ramane singur, si trebuie
        # sa ajunga la `parse_open_system`, care raporteaza neconcordanta.
        assert [len(s) for s in group_fixed(MANUSCRIS[:5], 2)] == [2, 2, 1]

    def test_un_portativ_pe_sistem(self):
        assert [len(s) for s in group_fixed(PAGE, 1)] == [1, 1, 1, 1]

    def test_zero_e_refuzat(self):
        with pytest.raises(ValueError, match="cel putin un portativ"):
            group_fixed(PAGE, 0)

    def test_pagina_goala(self):
        assert group_fixed([], 2) == []


# Portativele de pe cele doua pagini din „Doamne bun asculta", asa cum le-a dat
# `staff_bands`. Pagina nu are acelasi numar de portative pe fiecare sistem:
# vocea solo primeste un portativ propriu abia din sistemul al patrulea, deci
# pagina 1 are 2, 2, 2, 3, 3, iar pagina 2 are 2, 3, 2.
#
# La baza paginii a doua sunt si patru portative goale, liniate. `staff_bands`
# le gaseste — de asta sunt 11 cutii aici pentru 7 portative cu muzica — pe cand
# detectorul YOLO din conducta nu le da deloc, si cu curatare si fara (masurat).
# Raman in fixtura fiindca fac vizibila regula de la coada listei.
DOAMNE_P1 = [
    (40, 162, 2400, 266), (40, 458, 2400, 558), (40, 760, 2400, 860),
    (40, 1056, 2400, 1156), (40, 1353, 2400, 1451), (40, 1651, 2400, 1750),
    (40, 1949, 2400, 2051), (40, 2247, 2400, 2347), (40, 2552, 2400, 2655),
    (40, 2862, 2400, 2968), (40, 3171, 2400, 3272), (40, 3366, 2400, 3474),
]
DOAMNE_P2 = [
    (40, 200, 2400, 302), (40, 496, 2400, 596), (40, 794, 2400, 893),
    (40, 1085, 2400, 1161), (40, 1369, 2400, 1461), (40, 1667, 2400, 1760),
    (40, 1953, 2400, 2047), (40, 2248, 2400, 2328), (40, 2532, 2400, 2625),
    (40, 2828, 2400, 2925), (40, 3128, 2400, 3221),
]


class TestDeclaratiePeFiecareSistem:
    """Cand numarul de portative nu e acelasi peste toata pagina."""

    def test_un_numar_fix_ar_taia_gresit_pagina(self):
        # Cazul care a cerut lista: declarata `2`, pagina iese in sase sisteme
        # de cate doua, iar sistemele de trei portative se rup in doua decupaje.
        assert [len(s) for s in group_fixed(DOAMNE_P1, 2)] == [2] * 6

    def test_lista_da_sistemele_adevarate(self):
        assert [len(s) for s in group_fixed(DOAMNE_P1, [2, 2, 2, 3, 3])] == [2, 2, 2, 3, 3]

    def test_ce_ramane_dupa_lista_nu_se_pierde(self):
        # Regula de la coada: portativele ramase peste declaratie devin un
        # sistem in plus, nu dispar si nu se lipesc de ultimul. Asa o declaratie
        # prea scurta se vede ca neconcordanta de numar de sisteme in
        # `dataset.py`, in loc sa imperecheze tacut alte masuri cu alta muzica.
        assert [len(s) for s in group_fixed(DOAMNE_P2, [2, 3, 2])] == [2, 3, 2, 4]

    def test_lista_mai_lunga_decat_pagina_nu_face_sisteme_goale(self):
        assert [len(s) for s in group_fixed(DOAMNE_P1[:5], [2, 3, 2, 2])] == [2, 3]

    def test_un_numar_si_lista_de_un_element_dau_acelasi_lucru(self):
        assert group_fixed(PAGE, 2) == group_fixed(PAGE, [2, 2])

    def test_zero_in_lista_e_refuzat(self):
        with pytest.raises(ValueError, match="cel putin un portativ"):
            group_fixed(PAGE, [2, 0])

    def test_lista_goala_e_refuzata(self):
        with pytest.raises(ValueError, match="nu poate fi goala"):
            group_fixed(PAGE, [])

    def test_pagina_goala(self):
        assert group_fixed([], [2, 3]) == []


class TestCitireaDeclaratiei:
    def test_un_numar(self):
        assert parse_grouping("2") == 2

    def test_o_lista(self):
        assert parse_grouping("2,2,2,3,3") == [2, 2, 2, 3, 3]

    def test_spatiile_nu_deranjeaza(self):
        assert parse_grouping(" 2 , 3 ") == [2, 3]

    def test_ce_nu_e_numar_se_refuza(self):
        with pytest.raises(ValueError, match="lista de numere"):
            parse_grouping("2,doua")

    def test_zero_se_refuza(self):
        with pytest.raises(ValueError, match="cel putin un portativ"):
            parse_grouping("2,0")


class TestAlegereaGruparii:
    def test_fara_declaratie_se_deduce_din_spatiere(self):
        assert group(PAGE) == group_by_gap(PAGE)

    def test_cu_declaratie_spatierea_e_ignorata(self):
        assert group(MANUSCRIS, per_system=2) == group_fixed(MANUSCRIS, 2)

    def test_declaratia_poate_fi_o_lista(self):
        assert group(DOAMNE_P1, per_system=[2, 2, 2, 3, 3]) == group_fixed(
            DOAMNE_P1, [2, 2, 2, 3, 3]
        )

    def test_tiparitul_nu_e_afectat(self):
        # Comportamentul implicit ramane cel de pana acum, inclusiv sistemul cu
        # un singur portativ din „Iata-L pe Hristos".
        page = [(10, 100, 390, 150)] + PAGE[2:]
        assert [len(s) for s in group(page)] == [1, 2]


class TestIncadrare:
    def test_cuprinde_toate_portativele_cu_margine(self):
        assert bounding_box(PAGE[:2], padding=4, size=(400, 500)) == (6, 96, 394, 212)

    def test_marginea_nu_iese_din_pagina(self):
        assert bounding_box(PAGE[:2], padding=200, size=(400, 500)) == (0, 0, 400, 408)
