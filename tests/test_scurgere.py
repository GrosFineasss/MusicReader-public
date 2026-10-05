"""Scurgerea: cat din corpusul de antrenare vine din piesa pe care se masoara.

Cazul care a dat nastere modulului e in PROJECT.md §15.41 — 69 din 299 de
perechi erau chiar pagina 3 — si e reprodus mai jos ca test, ca sa nu se poata
strecura a doua oara fara sa se vada.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.scurgere import (  # noqa: E402
    citeste_corpus,
    citeste_set,
    masoara,
    piesa_din_imagine,
    piesa_din_referinta,
    raport,
)


class TestNumelePiesei:
    def test_sistem_scriitor_si_varianta_se_taie(self):
        assert piesa_din_imagine("pagina3-s01-fineas-0.png") == "pagina3"

    def test_transpozitia_nu_face_alta_piesa(self):
        assert piesa_din_imagine("pagina3-t+2-s01-fineas-0.png") == "pagina3"
        assert piesa_din_imagine("pagina3-t-3-s02-radu-1.png") == "pagina3"

    def test_numele_cu_cratime_ramane_intreg(self):
        assert piesa_din_imagine("doamne-bun-asculta-s01-fineas-0.png") == "doamne-bun-asculta"
        assert piesa_din_imagine("doamne-bun-asculta-t-1-s03-ana-2.png") == "doamne-bun-asculta"

    def test_piesa_din_referinta(self):
        assert piesa_din_referinta("data/validation/pagina3.musicxml") == "pagina3"
        assert piesa_din_referinta(Path("x/transpuse/pagina3-t-1.musicxml")) == "pagina3"


class TestMasoara:
    def test_corpus_curat(self):
        s = masoara(["alta-s01-ana-0.png"] * 5, ["pagina3"])
        assert s.curat and s.scurse == 0 and s.procent == 0.0

    def test_numara_direct_si_transpus_separat(self):
        imagini = (
            ["pagina3-s01-fineas-0.png"] * 24
            + ["pagina3-t+2-s01-fineas-0.png"] * 45
            + ["doamne-bun-asculta-s01-fineas-0.png"] * 230
        )
        s = masoara(imagini, ["pagina3"])
        assert s.total == 299
        assert s.pe_piesa == {"pagina3": 24}
        assert s.pe_piesa_transpuse == {"pagina3": 45}
        assert s.scurse == 69
        assert round(s.procent) == 23

    def test_o_piesa_care_e_prefixul_alteia_nu_se_confunda(self):
        s = masoara(["pagina30-s01-ana-0.png"] * 3, ["pagina3"])
        assert s.curat, "`pagina30` nu e `pagina3`"

    def test_mai_multe_piese_masurate(self):
        imagini = ["pagina1-s01-ana-0.png", "pagina2-s01-ana-0.png", "alta-s01-ana-0.png"]
        s = masoara(imagini, ["pagina1", "pagina2"])
        assert s.scurse == 2


class TestRaport:
    def test_raportul_curat_spune_pe_ce_s_a_uitat(self):
        text = raport(masoara(["alta-s01-ana-0.png"], ["pagina3"]), ["pagina3"])
        assert "curat" in text and "pagina3" in text

    def test_raportul_cu_scurgere_da_cifra_si_defalcarea(self):
        s = masoara(["pagina3-s01-ana-0.png", "pagina3-t+2-s01-ana-0.png"], ["pagina3"])
        text = raport(s, ["pagina3"])
        assert "SCURGERE" in text
        assert "1 netranspuse + 1 transpuse = 2" in text

    def test_fara_set_nu_se_pretinde_nimic(self):
        assert "nu se poate spune" in raport(masoara([], []), [])


class TestCitire:
    def test_corpus_si_set(self, tmp_path):
        date = tmp_path / "date.jsonl"
        date.write_text(
            "\n".join(
                json.dumps({"image": nume, "transcription": "x", "staves": 2, "writer": "a"})
                for nume in ("pagina3-s01-a-0.png", "alta-s01-a-0.png")
            ),
            encoding="utf-8",
        )
        assert citeste_corpus(date) == ["pagina3-s01-a-0.png", "alta-s01-a-0.png"]

        set_ = tmp_path / "set.json"
        set_.write_text(
            json.dumps([{"imagine": "x.png", "referinta": "d/pagina3.musicxml"}]), encoding="utf-8"
        )
        assert citeste_set(set_) == ["pagina3"]

    def test_setul_poate_fi_o_singura_intrare(self, tmp_path):
        set_ = tmp_path / "set.json"
        set_.write_text(json.dumps({"referinta": "d/pagina1.musicxml"}), encoding="utf-8")
        assert citeste_set(set_) == ["pagina1"]


def test_cazul_din_project_md_se_reproduce_pe_corpusul_real():
    """§15.41 exact: 69 din 299. Daca cifra se schimba, s-a schimbat corpusul."""
    corpus = ROOT / "data" / "synthetic" / "corpus" / "date.jsonl"
    if not corpus.exists():
        import pytest

        pytest.skip("corpusul sintetic nu e in git; se regenereaza cu sintetic.py")
    s = masoara(citeste_corpus(corpus), ["pagina3"])
    assert (s.scurse, s.total) == (69, 299)
