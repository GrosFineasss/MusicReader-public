"""Foaia de tiparit: harta casutelor si desenele de referinta (§15.22).

Prima foaie completata a aratat ca **numele nu ajunge**: „cap de doime" s-a citit
firesc ca „nota doime", si sapte casute din 36 s-au intors cu note intregi in loc
de bucati de nota. Foaia are acum, in casutele alea, un desen tiparit de copiat.

Doua lucruri se pot strica tacut de aici, si de asta sunt testate:

- **un desen care iese din casuta lui** ajunge peste casuta de scris de alaturi,
  iar segmentarea il decupeaza ca si cum l-ar fi scris omul;
- **casuta de exemplu ajunsa in harta** e mai rea: intra in repository ca stilul
  omului, si de acolo in fiecare pagina sintetica, cu desenul nostru in loc de
  scrisul lui.

Niciuna nu se vede uitandu-te la foaie.
"""

import numpy as np
import pytest

from scripts.foaie_glife import (
    CELL_DEFAULT,
    GAP_DEFAULT,
    EXEMPLE,
    REPETITIONS,
    SYMBOLS,
    build,
    deseneaza_exemplu,
)


@pytest.fixture(scope="module")
def pagini():
    return build()


def casute(pagini):
    return [cell for _, plan in pagini for cell in plan["cells"]]


class TestHarta:
    def test_fiecare_simbol_are_trei_casute(self, pagini):
        numarate = {}
        for cell in casute(pagini):
            numarate[cell["label"]] = numarate.get(cell["label"], 0) + 1
        assert numarate == {label: REPETITIONS for label, _ in SYMBOLS}

    def test_etichetele_cu_exemplu_sunt_de_pe_foaie(self):
        # O greseala de tipar in `EXEMPLE` n-ar da eroare: desenul pur si simplu
        # nu s-ar face, si simbolul care avea nevoie de el ar ramane fara.
        assert EXEMPLE <= {label for label, _ in SYMBOLS}

    def test_casutele_nu_se_suprapun(self, pagini):
        for _, plan in pagini:
            boxes = [cell["box"] for cell in plan["cells"]]
            for i, (ax1, ay1, ax2, ay2) in enumerate(boxes):
                for bx1, by1, bx2, by2 in boxes[i + 1:]:
                    assert ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1

    def test_casutele_nu_ating_reperele(self, pagini):
        for _, plan in pagini:
            for mx1, my1, mx2, my2 in plan["markers"]:
                for x1, y1, x2, y2 in (c["box"] for c in plan["cells"]):
                    assert x2 <= mx1 or mx2 <= x1 or y2 <= my1 or my2 <= y1


class TestCasutaDeExemplu:
    """Cea tiparita nu are voie sa fie in harta — altfel intra in repository."""

    def test_simbolul_cu_exemplu_are_tot_trei_casute_de_scris(self, pagini):
        numarate = {}
        for cell in casute(pagini):
            numarate[cell["label"]] = numarate.get(cell["label"], 0) + 1
        for label in EXEMPLE:
            assert numarate[label] == REPETITIONS

    def test_desenul_tiparit_nu_e_in_nicio_casuta(self, pagini):
        """Miezul: cerneala tiparita cade **in afara** oricarei casute din harta.

        Se cauta pe imaginea adevarata, nu pe coordonate: daca asezarea se
        schimba candva si exemplul aluneca peste o casuta de scris, aici se vede.
        """
        gasite = 0
        for image, plan in pagini:
            a = np.array(image.convert("L"))
            in_casuta = np.zeros(a.shape, dtype=bool)
            for x1, y1, x2, y2 in (c["box"] for c in plan["cells"]):
                in_casuta[y1:y2, x1:x2] = True

            for label in {c["label"] for c in plan["cells"]} & EXEMPLE:
                boxes = sorted(c["box"] for c in plan["cells"] if c["label"] == label)
                x1, y1, x2, y2 = boxes[0]
                latime = x2 - x1
                # Exemplul e casuta dinaintea primei de scris.
                ex = (x1 - latime - GAP_DEFAULT, y1, x1 - GAP_DEFAULT, y2)
                bucata = a[ex[1]:ex[3], ex[0]:ex[2]]
                assert (bucata < 250).any(), f"{label}: casuta de exemplu e goala"
                assert not in_casuta[ex[1]:ex[3], ex[0]:ex[2]].any()
                gasite += 1
        assert gasite == len(EXEMPLE)


class TestDesenele:
    """Fiecare desen trebuie sa incapa in casuta lui."""

    @pytest.mark.parametrize("eticheta", sorted(EXEMPLE))
    @pytest.mark.parametrize("casuta", [160, CELL_DEFAULT, 340])
    def test_desenul_nu_iese_din_casuta(self, eticheta, casuta):
        """Si intr-o casuta mai mica: foaia de supliment le micsoreaza.

        Numerele din desene sunt alese pentru `CELL_DEFAULT` si inmultite cu un
        factor. Inainte de scalare, cinci din opt desene ieseau din casuta la
        latura 200 — masurat, nu presupus.
        """
        from PIL import Image, ImageDraw

        # Rama de jur imprejur: ce iese se vede ca cerneala pe ea.
        rama = 40
        image = Image.new("L", (casuta + 2 * rama, casuta + 2 * rama), 255)
        draw = ImageDraw.Draw(image)
        box = (rama, rama, rama + casuta, rama + casuta)
        deseneaza_exemplu(draw, eticheta, box)

        a = np.array(image)
        cerneala = a < 250
        assert cerneala.any(), f"{eticheta}: n-a desenat nimic"

        inauntru = np.zeros(a.shape, dtype=bool)
        inauntru[box[1]:box[3], box[0]:box[2]] = True
        assert not (cerneala & ~inauntru).any(), (
            f"{eticheta}: desenul iese din casuta de {casuta}"
        )

    @pytest.mark.parametrize("eticheta", sorted(EXEMPLE))
    def test_desenul_e_gri_nu_negru(self, eticheta):
        # Gri, ca sa nu para parte din chenar si ca sa se distinga de ce scrie
        # omul cu pixul deasupra.
        from PIL import Image, ImageDraw

        image = Image.new("L", (CELL_DEFAULT, CELL_DEFAULT), 255)
        draw = ImageDraw.Draw(image)
        deseneaza_exemplu(draw, eticheta, (0, 0, CELL_DEFAULT, CELL_DEFAULT))
        a = np.array(image)
        assert a.min() > 60


class TestBravura:
    """Desenele-exemplu din fontul de referinta SMuFL (Blocul C, noaptea 04.09).

    Fontul sta in `vendor/`, care nu e in git, deci tot ce-l foloseste trebuie sa
    functioneze si fara el — altfel foaia nu s-ar mai putea genera pe o masina
    proaspata. De aceea se testeaza si calea cu font, si cea fara.
    """

    def test_codepoint_da_caracterul_din_tabelul_smufl(self):
        from scripts.foaie_glife import SMUFL_NUME, codepoint

        if not SMUFL_NUME.exists():
            pytest.skip("tabelul SMuFL nu e descarcat (scripts/ia_bravura.py)")
        assert codepoint("noteheadBlack") == "\ue0a4"
        assert codepoint("gClef") == "\ue050"

    def test_etichetele_proprii_lui_smashcima_nu_au_codepoint(self):
        from scripts.foaie_glife import SMUFL_NUME, codepoint

        if not SMUFL_NUME.exists():
            pytest.skip("tabelul SMuFL nu e descarcat")
        for eticheta in ("beam", "beamHook", "slur"):
            assert codepoint(eticheta) is None, f"{eticheta} nu e nume SMuFL"

    def test_fara_font_glifa_spune_ca_nu_poate(self, monkeypatch, tmp_path):
        from PIL import Image, ImageDraw

        from scripts import foaie_glife

        monkeypatch.setattr(foaie_glife, "BRAVURA", tmp_path / "lipseste.otf")
        foaie_glife._FONTURI.clear()
        draw = ImageDraw.Draw(Image.new("L", (100, 100), 255))
        assert foaie_glife.glifa(draw, "noteheadBlack", 50, 50, 20, 130) is False

    def test_desenul_iese_la_fel_si_fara_font(self, monkeypatch, tmp_path):
        """Revenirea la primitive deseneaza ceva, nu lasa casuta goala."""
        from PIL import Image, ImageDraw

        from scripts import foaie_glife

        monkeypatch.setattr(foaie_glife, "BRAVURA", tmp_path / "lipseste.otf")
        foaie_glife._FONTURI.clear()
        imagine = Image.new("L", (300, 300), 255)
        foaie_glife.deseneaza_exemplu(ImageDraw.Draw(imagine), "noteheadBlack",
                                      (25, 25, 275, 275))
        assert np.array(imagine).min() < 255, "casuta de exemplu a ramas goala"


def test_harta_nu_s_a_mutat_fata_de_foaia_deja_tiparita():
    """Casutele trebuie sa ramana unde erau cand s-a tiparit foaia (§15.22).

    Foile completate se decupeaza cu harta, deci o casuta mutata cu un pixel
    face nefolosibile toate foile deja scrise de oameni. Desenele dinauntru pot
    sa se schimbe oricat; **asezarea** nu.
    """
    import json
    from pathlib import Path

    from scripts.foaie_glife import build

    harta = Path(__file__).resolve().parents[1] / "data" / "glife" / "harta.json"
    if not harta.exists():
        pytest.skip("harta comisa lipseste")
    asteptat = json.loads(harta.read_text(encoding="utf-8"))

    pagini = [plan for _, plan in build()]
    assert len(pagini) == len(asteptat["pages"])
    for plan, referinta in zip(pagini, asteptat["pages"]):
        assert plan["markers"] == referinta["markers"]
        assert [(c["label"], c["index"], c["box"]) for c in plan["cells"]] == [
            (c["label"], c["index"], c["box"]) for c in referinta["cells"]
        ]
