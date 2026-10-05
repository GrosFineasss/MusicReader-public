"""Etapa 6: indreptare, gasirea liniilor, stergerea textului.

Testele lucreaza pe pagini sintetice desenate aici, nu pe scanuri: asa stim
exact ce ar trebui sa iasa, si raman rapide. Proportiile sunt cele masurate pe
scanul real din §14 — grosime : spatiu de 1:5, portativ de patru pasi — la
jumatate de scara, ca suita sa nu incetineasca.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.preprocessing import clean_page  # noqa: E402
from app.preprocessing.deskew import binarize, estimate_skew, rotate  # noqa: E402
from app.preprocessing.staff_lines import (  # noqa: E402
    group_lines,
    line_metrics,
    merge_split_lines,
    staff_bands,
)
from app.preprocessing.text import suppress_text  # noqa: E402

# Geometria scanului real, la jumatate de scara: acolo liniile sunt de 4 px si
# spatiile de 20, aici 2 si 10. Ce foloseste algoritmul sunt **rapoartele** —
# grosime : spatiu si inaltime = 4 pasi — iar ele sunt pastrate intocmai.
# Jumatatea de scara face paginile de patru ori mai mici, si nucleul morfologic
# de doua ori mai ingust; suita ramane rapida, cum cere §11.4.
THICKNESS = 2
SPACING = 10
PITCH = THICKNESS + SPACING  # distanta centru-la-centru intre doua linii
STAFF_HEIGHT = 4 * PITCH

PAGE_HEIGHT = 460
PAGE_WIDTH = 600
STAFF_TOPS = (40, 140, 240, 340)


def draw_staff(page: np.ndarray, top: int, left: int = 30, right: int = 570) -> None:
    for index in range(5):
        y = top + index * PITCH
        page[y:y + THICKNESS, left:right] = 0


def blank(height: int = PAGE_HEIGHT, width: int = PAGE_WIDTH) -> np.ndarray:
    """Pagina alba, cerneala inchisa — conventia imaginilor reale."""
    return np.full((height, width), 255, dtype=np.uint8)


def page_with_two_systems() -> np.ndarray:
    """Doua sisteme de cate doua portative, cu cate un rand de text sub fiecare.

    E aranjamentul manuscrisului real: portativ, versuri, portativ, versuri —
    spatiere uniforma, deci nimic nu separa sistemele geometric (§14.6).
    """
    page = blank()
    for top in STAFF_TOPS:
        draw_staff(page, top)
        # „versuri": pete mici, sub portativ, care nu-l ating
        for x in range(40, 560, 30):
            page[top + STAFF_HEIGHT + 20:top + STAFF_HEIGHT + 34, x:x + 15] = 0
    return page


class TestIndreptare:
    @pytest.mark.parametrize("angle", [-2.5, -1.0, 0.0, 1.5, 2.3])
    def test_unghiul_masurat_il_anuleaza_pe_cel_aplicat(self, angle):
        # Pagina se inclina deliberat, iar `estimate_skew` trebuie sa ceara
        # rotatia inversa. Toleranta e un pas fin de cautare.
        page = binarize(blank())
        for top in STAFF_TOPS:
            for i in range(5):
                page[top + i * PITCH:top + i * PITCH + THICKNESS, 30:570] = 255
        assert estimate_skew(rotate(page, angle)) == pytest.approx(-angle, abs=0.1)

    def test_o_pagina_dreapta_nu_se_roteste(self):
        assert estimate_skew(binarize(page_with_two_systems())) == pytest.approx(0.0, abs=0.1)


class TestMasuratori:
    def test_grosimea_si_spatiul_ies_din_imagine(self):
        thickness, spacing = line_metrics(binarize(page_with_two_systems()))
        assert thickness == THICKNESS
        assert spacing == SPACING

    def test_o_pagina_goala_e_refuzata_cu_mesaj(self):
        # Mai bine o eroare decat masuratori inventate din nimic.
        with pytest.raises(ValueError, match="cerneala"):
            line_metrics(binarize(blank()))


class TestLinii:
    def test_gaseste_toate_portativele(self):
        bands = staff_bands(binarize(page_with_two_systems()))
        assert len(bands) == 4
        assert all(len(band.lines) == 5 for band in bands)
        assert all(band.height == pytest.approx(STAFF_HEIGHT, abs=4) for band in bands)

    def test_o_linie_gasita_in_bucati_se_reunifica(self):
        """Cazul real: o linie groasa sau intrerupta apare ca doua randuri.

        Pe pagina 1 din scan asta rupsese portativele in fragmente de 28-56 px.
        Doua linii ale aceluiasi portativ nu pot fi la 3 px una de alta.
        """
        assert merge_split_lines([100, 103, 124, 148, 151, 172], PITCH) == [101, 124, 149, 172]

    def test_un_portativ_cu_linii_pierdute_e_tot_portativ(self):
        """Cazul real de pe scan: 3 linii din 5, la pasul paginii.

        Pe pagina 3 asa arata primul portativ, pe pagina 1 ultimul. Cand erau
        aruncate, pagina 1 iesea cu 11 portative in loc de 12 — deci gruparea
        declarata cate doua producea 5 sisteme si un rest orfan.
        """
        page = page_with_two_systems()
        # sterge ultimele doua linii ale primului portativ, ca pe o fotocopie
        top = STAFF_TOPS[0]
        for index in (3, 4):
            y = top + index * PITCH
            page[y:y + THICKNESS, :] = 255

        bands = staff_bands(binarize(page))
        assert len(bands) == 4
        assert len(bands[0].lines) == 3

    def test_portativul_incomplet_pastreaza_loc_pentru_liniile_lipsa(self):
        """Banda lui trebuie sa acopere si partea in care liniile s-au pierdut.

        Altfel `suppress_text` ar sterge tocmai notele de acolo — iar cerneala
        stearsa in minus e o nota care nu mai exista pentru nimeni.
        """
        page = page_with_two_systems()
        top = STAFF_TOPS[0]
        for index in (3, 4):
            page[top + index * PITCH:top + index * PITCH + THICKNESS, :] = 255

        band = staff_bands(binarize(page))[0]
        assert band.top <= top
        assert band.bottom >= top + STAFF_HEIGHT

    def test_doua_bare_alaturate_nu_devin_portativ(self):
        """Motivul pentru care pragul era 4: nu vrem sa inventam portative.

        Trei randuri la un pas care nu e al paginii raman resturi, nu portativ.
        """
        assert group_lines([100, 100 + PITCH * 2, 100 + PITCH * 4], PITCH) == []

    def test_un_grup_cu_alt_pas_nu_e_portativ(self):
        # Cinci linii ingramadite la un sfert din pasul paginii — text, o pata,
        # doua bare alaturate. Orice, numai portativ nu.
        squeezed = [100 + i * (PITCH // 4) for i in range(5)]
        assert group_lines(squeezed, PITCH) == []


class TestStergereaTextului:
    def test_versurile_dispar_si_portativele_raman(self):
        page = binarize(page_with_two_systems())
        bands = staff_bands(page)
        result = suppress_text(page, bands)

        assert result.removed > 0
        # Portativele trebuie sa fie toate acolo dupa curatare.
        assert len(staff_bands(result.image)) == len(bands)

    def test_cerneala_lipita_de_portativ_ramane(self):
        """O codita care atinge portativul e muzica, oriunde ar ajunge celalalt capat."""
        page = blank()
        draw_staff(page, 100)
        # codita in jos, din portativ pana adanc in banda de text
        page[100 + STAFF_HEIGHT:100 + STAFF_HEIGHT + 40, 300:303] = 0
        binary = binarize(page)
        result = suppress_text(binary, staff_bands(binary))
        assert result.image[100 + STAFF_HEIGHT + 35, 300:303].any()

    def test_nota_cu_linii_suplimentare_ramane(self):
        """Tenorul deasupra portativului de jos: nu atinge banda, dar e nota.

        Gasit pe paginile Bach randate (2026-10-02): cap si codita la trei
        spatii deasupra portativului, fara sa-l atinga — curatarea veche il
        stergea si lasa doar linia suplimentara.
        """
        page = blank()
        draw_staff(page, 200)
        head_top = 200 - 3 * PITCH
        page[head_top:head_top + 8, 300:312] = 0  # capul
        page[head_top - 28:head_top + 8, 310:312] = 0  # codita in sus, ~2,3 spatii
        page[head_top - 22:head_top + 10, 284:288] = 0  # alteratia din stanga, ~2,7 spatii
        page[head_top + 2:head_top + 6, 316:320] = 0  # punctul din dreapta
        binary = binarize(page)
        result = suppress_text(binary, staff_bands(binary))
        assert result.image[head_top + 2, 302:310].any()
        assert result.image[head_top - 15, 310:312].any()
        assert result.image[head_top - 10, 284:288].any()
        assert result.image[head_top + 4, 316:320].any()

    def test_versurile_fara_codita_tot_dispar(self):
        """Literele de langa nota salvata nu sunt salvate odata cu ea."""
        page = blank()
        draw_staff(page, 200)
        bottom = 200 + STAFF_HEIGHT
        page[bottom + 25:bottom + 39, 300:315] = 0  # o „litera", 1,2 spatii
        binary = binarize(page)
        result = suppress_text(binary, staff_bands(binary))
        assert not result.image[bottom + 25:bottom + 39, 300:315].any()

    def test_codita_departe_de_orice_portativ_dispare(self):
        """Peste `REACH_SPACES`, o trasatura verticala e text, nu nota."""
        page = blank(height=600)
        draw_staff(page, 100)
        far = 100 + STAFF_HEIGHT + 10 * PITCH
        page[far:far + 40, 300:303] = 0
        binary = binarize(page)
        result = suppress_text(binary, staff_bands(binary))
        assert not result.image[far:far + 40, 300:303].any()

    def test_fara_portative_nu_se_sterge_nimic(self):
        """O detectie esuata nu are voie sa goleasca pagina in tacere."""
        binary = binarize(page_with_two_systems())
        result = suppress_text(binary, [])
        assert result.removed == 0
        assert np.array_equal(result.image, binary)


class TestPaginaCuratata:
    def test_intoarce_masuratorile_nu_doar_imaginea(self):
        result = clean_page(page_with_two_systems())
        assert result.staves == 4
        assert result.angle == pytest.approx(0.0, abs=0.1)
        assert result.suppression.removed > 0

    def test_pastreaza_conventia_de_culoare(self):
        """Cerneala ramane inchisa pe fond deschis, ca in pagina originala."""
        result = clean_page(page_with_two_systems())
        assert result.image.max() == 255
        assert result.image.mean() > 128
