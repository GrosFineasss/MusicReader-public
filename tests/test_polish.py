"""Adaptorul `polish-scores` (`app/training/polish.py`).

Doua din capcanele de aici sunt tacute, si de aceea sunt testate:

- **exponentul.** `music21` nu cunoaste `**ekern`. Daca ramane asa, intoarce o
  partitura **goala fara sa arunce** — masuratoarea ar iesi „0 note gasite" si ar
  arata ca un esec al modelului, nu al conversiei.
- **separatorii.** `@` si `·` sunt doua straturi diferite. Scos doar unul, tokenii
  raman nevalizi, iar `music21` sare peste ei tot in tacere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.polish import (  # noqa: E402
    PolishError,
    ekern_la_kern,
    pagina_din_rand,
)

CAP = "**ekern\t**ekern"


def doc(*linii: str) -> str:
    return "\n".join((CAP, *linii)) + "\n"


class TestSeparatorii:
    def test_campurile_se_lipesc(self):
        assert "4cc-" in ekern_la_kern(doc("4@cc@-\t."))

    def test_caracterele_se_lipesc(self):
        assert "32qqgg(/L" in ekern_la_kern(doc("32@qq@gg\u00b7(\u00b7/\u00b7L\t."))

    def test_amandoua_straturile_deodata(self):
        """Tokenul real din pagina 72 a setului."""
        assert "8.aa-()L\\" in ekern_la_kern(doc(".\t8@.@aa@-\u00b7(\u00b7)\u00b7L\u00b7\\"))

    def test_tab_urile_raman(self):
        """Coloanele sunt structura documentului; lipite, se pierde o voce."""
        assert ekern_la_kern(doc("4@c\t4@e")).split("\n")[1] == "4c\t4e"


class TestExponentul:
    @pytest.mark.parametrize("scris", ["**ekern", "**bekern"])
    def test_se_redenumeste_in_kern(self, scris):
        iesire = ekern_la_kern(f"{scris}\t{scris}\n4@c\t.\n")
        assert iesire.startswith("**kern\t**kern")

    def test_kern_curat_trece_neatins(self):
        """Idempotenta: adaptorul nu strica un document deja bun."""
        intrare = "**kern\t**kern\n4c\t4e\n*-\t*-\n"
        assert ekern_la_kern(intrare) == intrare

    def test_document_fara_exponent_se_refuza(self):
        with pytest.raises(PolishError):
            ekern_la_kern("4@c\t.\n")


class TestTerminatorul:
    def test_se_adauga_daca_lipseste(self):
        assert ekern_la_kern(doc("4@c\t.")).rstrip().endswith("*-\t*-")

    def test_cate_unul_de_fiecare_coloana(self):
        ultima = ekern_la_kern("**ekern\t**ekern\t**ekern\n4@c\t.\t.\n").rstrip().split("\n")[-1]
        assert ultima == "*-\t*-\t*-"

    def test_nu_se_dubleaza(self):
        iesire = ekern_la_kern("**ekern\t**ekern\n4@c\t.\n*-\t*-\n")
        assert iesire.count("*-\t*-") == 1


class TestRandulDeParquet:
    def rand(self, **peste):
        baza = {"image": {"bytes": b"\xff\xd8jpeg", "path": "72.jpg"}, "transcription": doc("4@c\t.")}
        return {**baza, **peste}

    def test_numele_vine_din_cale_fara_extensie(self):
        assert pagina_din_rand(self.rand()).nume == "72"

    def test_octetii_trec_neatinsi(self):
        assert pagina_din_rand(self.rand()).imagine == b"\xff\xd8jpeg"

    def test_kern_ul_e_convertit(self):
        assert pagina_din_rand(self.rand()).kern.startswith("**kern")

    def test_rand_fara_imagine_se_refuza(self):
        """O pagina fara poza n-are ce cauta intr-o masuratoare."""
        with pytest.raises(PolishError):
            pagina_din_rand(self.rand(image={"bytes": b"", "path": "x.jpg"}))

    def test_pagina_e_inghetata(self):
        with pytest.raises(Exception):
            pagina_din_rand(self.rand()).nume = "altul"  # type: ignore[misc]
