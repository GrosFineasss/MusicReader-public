"""Taierea unei referinte de mai multe pagini (`evaluate.py --pagina`).

O referinta de doua pagini masurata intreaga fata de o singura poza da o cifra
fara sens: tot ce e pe pagina cealalta iese `lipsa`, iar `note%` se imparte la
un numitor de doua ori prea mare. `doamne-bun-asculta.musicxml` e chiar asa —
41 de masuri pe doua pagini — si de aceea n-a putut fi masurata pana la
§15.44.

Aceeasi taiere o face si `dataset.py` de mai demult; ce se testeaza aici e ca
marginile ies la fel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.music_model.model import Duration, Measure, Note, NoteType, Part, Pitch, Score  # noqa: E402
from scripts.evaluate import taie_pagina  # noqa: E402


def partitura(masuri: int) -> Score:
    """O partitura cu `masuri` masuri, fiecare cu o singura nota, numerotate 1..N."""
    score = Score(title="proba")
    part = Part(id="P1", name="v")
    part.measures = [
        Measure(
            number=n,
            events=[Note(pitch=Pitch(step="C", octave=5), duration=Duration(NoteType.WHOLE))],
        )
        for n in range(1, masuri + 1)
    ]
    score.parts.append(part)
    return score


class TestMarginile:
    def test_prima_pagina_incepe_la_1_si_se_opreste_inainte_de_a_doua(self):
        _, first, last = taie_pagina(partitura(41), [1, 23], 1)
        assert (first, last) == (1, 22)

    def test_ultima_pagina_merge_pana_la_capat(self):
        _, first, last = taie_pagina(partitura(41), [1, 23], 2)
        assert (first, last) == (23, 41)

    def test_pagina_din_mijloc(self):
        _, first, last = taie_pagina(partitura(30), [1, 11, 21], 2)
        assert (first, last) == (11, 20)

    def test_o_singura_pagina_e_toata_referinta(self):
        _, first, last = taie_pagina(partitura(14), [1], 1)
        assert (first, last) == (1, 14)


class TestBucataTaiata:
    def test_are_exact_masurile_paginii(self):
        bucata, _, _ = taie_pagina(partitura(41), [1, 23], 1)
        assert bucata.measure_count == 22

    def test_a_doua_pagina_are_restul(self):
        bucata, _, _ = taie_pagina(partitura(41), [1, 23], 2)
        assert bucata.measure_count == 19

    def test_bucatile_acopera_totul_fara_suprapunere(self):
        """Suma paginilor trebuie sa fie referinta, altfel se pierde muzica."""
        intreaga = partitura(41)
        bucati = [taie_pagina(intreaga, [1, 23], p)[0].measure_count for p in (1, 2)]
        assert sum(bucati) == intreaga.measure_count


class TestRefuzuri:
    @pytest.mark.parametrize("ceruta", [0, 3, -1, 99])
    def test_pagina_care_nu_exista_se_refuza(self, ceruta):
        with pytest.raises(SystemExit) as eroare:
            taie_pagina(partitura(41), [1, 23], ceruta)
        assert "2 pagini" in str(eroare.value)
