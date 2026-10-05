"""Paginile generate in stilul imnarelor: S+A si T+B ca acorduri unde merg in acelasi ritm."""

import random
from fractions import Fraction

import pytest

from app.music_model.model import Duration, Note, NoteType, Pitch, Rest
from app.training.bach import musicxml_strans, se_scrie_acord

music21 = pytest.importorskip("music21")


def _nota(step, octave, tip=NoteType.QUARTER, **legato):
    return Note(Pitch(step, octave), Duration(tip), **legato)


class TestCandSeScrieAcord:
    def test_acelasi_ritm_voce_de_sus_deasupra(self):
        assert se_scrie_acord([_nota("E", 5), _nota("D", 5)], [_nota("C", 5), _nota("B", 4)])

    def test_unisonul_ramane_doua_voci(self):
        assert not se_scrie_acord([_nota("C", 5)], [_nota("C", 5)])

    def test_incrucisarea_ramane_doua_voci(self):
        assert not se_scrie_acord([_nota("A", 4)], [_nota("C", 5)])

    def test_aceeasi_treapta_cu_alta_alteratie_ramane_doua_voci(self):
        assert not se_scrie_acord([Note(Pitch("F", 4, 1), Duration(NoteType.QUARTER))],
                                  [_nota("F", 4)])

    def test_alt_ritm_ramane_doua_voci(self):
        assert not se_scrie_acord(
            [_nota("E", 5, NoteType.HALF)], [_nota("C", 5), _nota("B", 4)]
        )

    def test_pauza_ramane_doua_voci(self):
        assert not se_scrie_acord(
            [_nota("E", 5)], [Rest(Duration(NoteType.QUARTER))]
        )

    def test_octava_se_leaga_decima_nu(self):
        assert se_scrie_acord([_nota("C", 4)], [_nota("C", 3)])
        assert not se_scrie_acord([_nota("E", 4)], [_nota("C", 3)])

    def test_legatura_doar_intr_o_voce_ramane_doua_voci(self):
        assert not se_scrie_acord(
            [_nota("E", 5, tie_start=True)], [_nota("C", 5)]
        )


@pytest.fixture(scope="module")
def doua_stiluri(tmp_path_factory):
    """Prima pagina din bwv66.6, scrisa in ambele stiluri si citita inapoi de pe disc."""
    from app.evaluation.reference import load_with_layout
    from app.training.bach import coral_satb, pagini
    from scripts.dataset import system_bounds, targets_for

    score = coral_satb(music21.corpus.parse("bach/bwv66.6"))
    pagina, limite = pagini(score, random.Random(0))[0]
    folder = tmp_path_factory.mktemp("acorduri")
    citite = {}
    for nume, acorduri in (("voci", False), ("acorduri", True)):
        cale = folder / f"{nume}.musicxml"
        cale.write_text(musicxml_strans(pagina, limite, True, acorduri), encoding="utf-8")
        inapoi, starts, _ = load_with_layout(cale, min_voices=2)
        citite[nume] = (
            cale.read_text(encoding="utf-8"),
            inapoi,
            targets_for(inapoi, system_bounds(starts, inapoi.measure_count)),
        )
    return citite


class TestDusIntorsCuAcorduri:
    def test_pagina_chiar_are_acorduri(self, doua_stiluri):
        assert "<chord" in doua_stiluri["acorduri"][0]
        assert "<chord" not in doua_stiluri["voci"][0]

    def test_tintele_ies_identice_in_ambele_stiluri(self, doua_stiluri):
        assert doua_stiluri["acorduri"][2] == doua_stiluri["voci"][2]

    def test_vocile_raman_satb_cu_aceleasi_durate(self, doua_stiluri):
        inapoi = doua_stiluri["acorduri"][1]
        assert [part.id for part in inapoi.parts] == ["S", "A", "T", "B"]
        for part in inapoi.parts:
            for masura in part.measures:
                total = sum((event.quarter_length for event in masura.events), Fraction(0))
                assert total == masura.quarter_length, f"{part.id} m{masura.number}"
