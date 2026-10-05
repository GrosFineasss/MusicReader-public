"""Plafonul de decodare, raportat acolo unde se citeste rezultatul (§15.36).

Decodarea se opreste dupa `maxlen - 1` tokeni. Un model care o ia razna atinge
plafonul si e taiat la mijlocul muzicii — dar iesirea lui arata exact ca a unui
portativ scurt citit corect. §15.36 a gasit tiparul doar fiindca **aceeasi
cifra a aparut de trei ori**; `distanta_domeniu.py` il marcheaza de atunci, insa
transcrierea, lotul si fereastra nu stiau nimic despre el.

Intr-un lot de 500 de pagini, o pagina rupta care nu se plange e mai rea decat
una care esueaza: nimeni nu se uita la ea.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.recognition.engine import PageImage, PageResult, StaffCrop  # noqa: E402


def staff(index: int, taiat: bool = False, tokens: int = 10) -> StaffCrop:
    return StaffCrop(
        index=index,
        box=(0, 0, 10, 10),
        image=np.zeros((2, 2), dtype=np.uint8),
        kern="4c",
        tokens=tokens,
        taiat=taiat,
    )


def pagina() -> PageImage:
    return PageImage(path=Path("x.png"), kind="handwritten")


class TestSteagulPePortativ:
    def test_implicit_nu_e_taiat(self):
        assert staff(1).taiat is False

    def test_pagina_fara_taieri_da_lista_goala(self):
        rezultat = PageResult(page=pagina(), staves=[staff(1), staff(2)])
        assert rezultat.taiate == []

    def test_pagina_spune_care_portative_sunt_taiate(self):
        rezultat = PageResult(
            page=pagina(),
            staves=[staff(1), staff(2, taiat=True), staff(3), staff(4, taiat=True)],
        )
        assert rezultat.taiate == [2, 4]

    def test_o_pagina_taiata_nu_mai_arata_ca_una_citita(self):
        """Exact confuzia din §15.36, acum deosebita de doua obiecte diferite."""
        scurta = PageResult(page=pagina(), staves=[staff(1, tokens=80)])
        rupta = PageResult(page=pagina(), staves=[staff(1, tokens=80, taiat=True)])
        assert scurta.staves[0].tokens == rupta.staves[0].tokens
        assert scurta.taiate != rupta.taiate


class TestPlafonulDecoderului:
    """Aritmetica plafonului, fara sa se incarce modelul.

    `plafon()` exista tocmai ca sa nu fie doua expresii: cine compara o lungime
    cu plafonul trebuie sa obtina acelasi numar ca bucla care taie.
    """

    def decoder(self, maxlen: int):
        from app.recognition.incremental import IncrementalSMT

        obiect = IncrementalSMT.__new__(IncrementalSMT)
        obiect.maxlen = maxlen
        return obiect

    def test_fara_limita_ceruta_plafonul_e_al_modelului(self):
        assert self.decoder(1277).plafon(None) == 1276

    def test_o_limita_mai_mica_castiga(self):
        assert self.decoder(1277).plafon(100) == 99

    def test_o_limita_mai_mare_decat_modelul_nu_ridica_plafonul(self):
        assert self.decoder(1277).plafon(5000) == 1276

    def test_cifra_din_project_md(self):
        """§15.36: 1276 de tokeni de model pe `smt-grandstaff`."""
        assert self.decoder(1277).plafon(None) == 1276
