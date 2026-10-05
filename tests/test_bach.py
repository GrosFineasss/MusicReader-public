"""Coralele Bach -> pagini SATB tiparite (`app/training/bach.py`).

Imaginea o deseneaza MuseScore din MusicXML-ul scris aici, iar tinta o scrie
`kern_writer` din aceeasi partitura. Deci proba care conteaza e ca **fisierul,
recitit de `music21`, da aceeasi muzica si aceleasi rupturi** — altfel decupajul
si tinta ar descrie lucruri diferite si nimic n-ar observa.

Alteratiile au teste proprii fiindca sunt singurul loc unde partitura stransa se
poarta altfel decat cea pe patru portative: doua voci pe acelasi portativ isi
impart semnele din masura.
"""

import random
from fractions import Fraction

import pytest

from app.evaluation.reference import UnsupportedNotation, load_with_layout
from app.music_model.model import (
    Accidental,
    Duration,
    KeySignature,
    Note,
    NoteType,
    Pitch,
    TimeSignature,
    empty_satb,
    measures_slice,
)
from app.recognition.kern_writer import to_kern_text
from app.training.bach import (
    COLOANE_PE_SISTEM,
    CU_SOLO,
    MASURI_PE_SISTEM,
    SISTEME_PE_PAGINA,
    alteratii_afisate,
    coloane,
    coral_satb,
    cu_solo,
    musicxml_strans,
    normalizeaza_alteratiile,
    pagini,
    sisteme,
)

PATRIME = Duration(NoteType.QUARTER)


def nota(step, octave, alter=0, **legato):
    return Note(Pitch(step, octave, alter), PATRIME, **legato)


def partitura_sol_major(sopran, alto, tenor=None, bas=None):
    """O masura de 4/4 in Sol major; vocile nedate cant Sol-uri."""
    score = empty_satb(1, TimeSignature(4, 4), KeySignature(1))
    implicit = {"S": nota("G", 4), "A": nota("G", 4), "T": nota("G", 3), "B": nota("G", 2)}
    for part_id, note in zip("SATB", (sopran, alto, tenor, bas)):
        events = note or [implicit[part_id]] * 4
        score.part(part_id).measures[0].events = list(events)
    return score


class TestAlteratiile:
    def test_portativul_comun_isi_imparte_semnele(self):
        """Fa diez din armura, becar la alto, apoi diezul sopranului trebuie redesenat.

        Pe patru portative, sopranul n-ar avea nevoie de niciun semn: becarul
        altistului ar fi pe alt portativ.
        """
        score = partitura_sol_major(
            sopran=[nota("F", 4, 1), nota("G", 4), nota("F", 4, 1), nota("G", 4)],
            alto=[nota("D", 4), nota("F", 4, 0), nota("D", 4), nota("D", 4)],
        )
        semne = alteratii_afisate(score)
        assert semne[("S", 1, 0)] is None  # diezul vine din armura
        assert semne[("A", 1, 1)] == Accidental.NATURAL
        assert semne[("S", 1, 2)] == Accidental.SHARP

    def test_portativul_de_jos_nu_vede_semnele_de_sus(self):
        score = partitura_sol_major(
            sopran=[nota("F", 4, 0)] + [nota("G", 4)] * 3,
            alto=None,
            tenor=[nota("F", 4, 1)] + [nota("G", 3)] * 3,
        )
        semne = alteratii_afisate(score)
        assert semne[("S", 1, 0)] == Accidental.NATURAL
        assert semne[("T", 1, 0)] is None

    def test_nota_legata_dinspre_stanga_nu_primeste_semn(self):
        score = partitura_sol_major(
            sopran=[nota("F", 4, 0, tie_stop=True), nota("G", 4), nota("G", 4), nota("G", 4)],
            alto=None,
        )
        assert alteratii_afisate(score)[("S", 1, 0)] is None

    def test_normalizarea_scrie_inaltimea_si_doar_becarul_vazut(self):
        """Ca o referinta MuseScore citita de `music21`: `f#` si in Sol major."""
        score = partitura_sol_major(
            sopran=[nota("F", 4, 1), nota("F", 4, 0), nota("G", 4), nota("C", 5)],
            alto=None,
        )
        normalizeaza_alteratiile(score)
        events = score.part("S").measures[0].events
        assert events[0].pitch.accidental == Accidental.SHARP
        assert events[1].pitch.accidental == Accidental.NATURAL
        assert events[3].pitch.accidental is None


@pytest.fixture(scope="module")
def coral():
    corpus = pytest.importorskip("music21").corpus
    return coral_satb(corpus.parse("bach/bwv269"))


class TestCoralul:
    def test_asezarea_e_partitura_stransa(self, coral):
        assert coral.staves == [["S", "A"], ["T", "B"]]
        assert [part.id for part in coral.parts] == ["S", "A", "T", "B"]

    def test_tenorul_trece_in_cheia_fa(self, coral):
        """Un portativ are o singura cheie; altfel `kern_writer` refuza sistemul."""
        cheie = coral.part("T").measures[0].clef
        assert (cheie.sign, cheie.line, cheie.octave_change) == ("F", 4, 0)

    def test_refuza_ce_nu_e_in_patru_voci(self):
        corpus = pytest.importorskip("music21").corpus

        with pytest.raises(UnsupportedNotation, match="5 portative"):
            coral_satb(corpus.parse("bach/bwv1.6"))


class TestAsezarea:
    @pytest.mark.parametrize("numar", [1, 2, 7, 17, 40])
    def test_sistemele_acopera_toate_masurile(self, numar):
        limite = sisteme(numar, random.Random(numar))
        assert limite[0][0] == 1 and limite[-1][1] == numar
        assert all(b[0] == a[1] + 1 for a, b in zip(limite, limite[1:]))

    def test_niciun_sistem_de_o_masura_la_coada(self):
        for samanta in range(200):
            limite = sisteme(13, random.Random(samanta))
            assert limite[-1][1] > limite[-1][0]

    def test_lungimile_raman_in_plaja(self):
        limite = sisteme(60, random.Random(0))
        assert {b - a + 1 for a, b in limite[:-1]} <= set(MASURI_PE_SISTEM)

    def test_un_sistem_dens_se_scurteaza(self):
        """Peste `COLOANE_PE_SISTEM`, MuseScore rupe singur randul — deci nu se scrie asa."""
        for samanta in range(50):
            limite = sisteme(30, random.Random(samanta), latime=lambda numar: 12)
            assert all(12 * (b - a + 1) <= COLOANE_PE_SISTEM for a, b in limite)

    def test_o_masura_mai_lata_decat_plafonul_ramane_singura(self):
        """Nu se poate scurta sub o masura; macar nu se lipeste de vecina."""
        limite = sisteme(4, random.Random(0), latime=lambda numar: 40)
        assert limite == [(1, 1), (2, 2), (3, 3), (4, 4)]

    def test_coloanele_numara_momentele_din_toate_vocile(self):
        score = partitura_sol_major(
            sopran=[Note(Pitch("G", 4), Duration(NoteType.HALF))] * 2,
            alto=[nota("D", 4)] * 4,
        )
        assert coloane(score, 1) == 4

    def test_paginile_sunt_echilibrate_si_numerotate_de_la_unu(self):
        score = empty_satb(37)
        rezultat = pagini(score, random.Random(3))
        marimi = [len(limite) for _, limite in rezultat]
        assert max(marimi) <= SISTEME_PE_PAGINA
        assert max(marimi) - min(marimi) <= 1
        for pagina, limite in rezultat:
            assert pagina.part("S").measures[0].number == 1
            assert limite[0][0] == 1 and limite[-1][1] == pagina.measure_count
        assert sum(pagina.measure_count for pagina, _ in rezultat) == 37


@pytest.fixture(scope="module")
def pagina(tmp_path_factory):
    corpus = pytest.importorskip("music21").corpus
    coral = coral_satb(corpus.parse("bach/bwv10.7"))
    pagina, limite = pagini(coral, random.Random("bwv10.7"))[0]
    cale = tmp_path_factory.mktemp("bach") / "pagina.musicxml"
    cale.write_text(musicxml_strans(pagina, limite), encoding="utf-8")
    return pagina, limite, cale


class TestDusIntors:
    """Fisierul scris, recitit cu `music21`, trebuie sa dea exact pagina."""

    def test_rupturile_se_citesc_inapoi(self, pagina):
        _, limite, cale = pagina
        _, starts, pages = load_with_layout(cale)
        assert starts == [inceput for inceput, _ in limite]
        assert pages == [1]

    def test_tintele_ies_identice(self, pagina):
        """Tinta din memorie si cea din fisierul recitit — caracter cu caracter."""
        original, limite, cale = pagina
        recitit, _, _ = load_with_layout(cale)
        assert recitit.staves == [["S", "A"], ["T", "B"]]
        for prima, ultima in limite:
            assert to_kern_text(measures_slice(recitit, prima, ultima), ties=False) == (
                to_kern_text(measures_slice(original, prima, ultima), ties=False)
            )

    def test_duratele_fiecarei_voci_se_pastreaza(self, pagina):
        original, _, cale = pagina
        recitit, _, _ = load_with_layout(cale)
        for part_id in "SATB":
            lungimi = [m.quarter_length for m in original.part(part_id).measures]
            assert [m.quarter_length for m in recitit.part(part_id).measures] == lungimi
            assert sum(lungimi, Fraction(0)) > 0

    def test_codite_opuse_si_fara_nume_de_parti(self, pagina):
        _, _, cale = pagina
        text = cale.read_text(encoding="utf-8")
        assert "<stem>up</stem>" in text and "<stem>down</stem>" in text
        assert 'print-object="no"' in text
        assert "Soprano" not in text

    def test_bara_finala_doar_la_sfarsitul_piesei(self, pagina):
        original, limite, _ = pagina
        assert "light-heavy" in musicxml_strans(original, limite, ultima_din_piesa=True)
        assert "light-heavy" not in musicxml_strans(original, limite, ultima_din_piesa=False)


@pytest.fixture(scope="module")
def pagina_cu_solo(tmp_path_factory):
    """Gazda si donorul cu aceeasi metrica (4/4), ca la generare; armura difera, nu conteaza aici."""
    corpus = pytest.importorskip("music21").corpus
    gazda = coral_satb(corpus.parse("bach/bwv10.7"))
    donor = coral_satb(corpus.parse("bach/bwv26.6"))
    cu = cu_solo(gazda, donor, random.Random("solo"))
    pagina, limite = pagini(cu, random.Random("bwv10.7"), sisteme_pe_pagina=3)[0]
    cale = tmp_path_factory.mktemp("solo") / "pagina.musicxml"
    cale.write_text(musicxml_strans(pagina, limite), encoding="utf-8")
    return gazda, cu, pagina, limite, cale


class TestCuSolo:
    """Un portativ de solo deasupra corului: sisteme de trei portative (2026-10-04)."""

    def test_solo_are_aceleasi_masuri_ca_gazda(self, pagina_cu_solo):
        gazda, cu, _, _, _ = pagina_cu_solo
        assert cu.staves == CU_SOLO
        solo, sopran = cu.part("X1").measures, gazda.part("S").measures
        assert [m.quarter_length for m in solo] == [m.quarter_length for m in sopran]
        assert solo[0].clef is not None

    def test_corul_ramane_neatins(self, pagina_cu_solo):
        gazda, cu, _, _, _ = pagina_cu_solo
        for vocea in "SATB":
            assert cu.part(vocea).measures == gazda.part(vocea).measures

    def test_fara_legaturi_de_durata_in_solo(self, pagina_cu_solo):
        """Masurile donorului nu mai urmeaza una dupa alta: o legatura ar lega alte note."""
        _, cu, _, _, _ = pagina_cu_solo
        note = [e for m in cu.part("X1").measures for e in m.events if isinstance(e, Note)]
        assert note
        assert not any(n.tie_start or n.tie_stop for n in note)

    def test_tintele_ies_identice_dus_intors(self, pagina_cu_solo):
        """Recitit cu o singura voce pe portativul de solo: aceeasi tinta, caracter cu caracter."""
        _, _, pagina, limite, cale = pagina_cu_solo
        recitit, starts, _ = load_with_layout(cale, min_voices=[1, 2, 2])
        assert starts == [inceput for inceput, _ in limite]
        assert [len(grup) for grup in recitit.staves] == [1, 2, 2]
        for prima, ultima in limite:
            assert to_kern_text(measures_slice(recitit, prima, ultima), ties=False) == (
                to_kern_text(measures_slice(pagina, prima, ultima), ties=False)
            )

    def test_doua_voci_impuse_pe_solo_ar_adauga_o_voce_goala(self, pagina_cu_solo):
        """De ce `min_voices` e pe portativ: cu 2 peste tot, solo-ul ar primi o voce de pauze."""
        _, _, _, _, cale = pagina_cu_solo
        recitit, _, _ = load_with_layout(cale, min_voices=2)
        assert [len(grup) for grup in recitit.staves] == [2, 2, 2]

    def test_codita_solo_o_hotaraste_musescore(self, pagina_cu_solo):
        """Pe un portativ cu o singura voce nu se impune codita in sus (code review, 2026-10-04)."""
        _, _, _, _, cale = pagina_cu_solo
        text = cale.read_text(encoding="utf-8")
        solo = text.split('<part id="P1">')[1].split("</part>")[0]
        cor = text.split('<part id="P2">')[1].split("</part>")[0]
        assert "<stem>" not in solo
        assert "<stem>up</stem>" in cor and "<stem>down</stem>" in cor

    def test_anacruza_fara_pereche_se_umple_cu_pauze(self):
        """O anacruza de o patrime nu are pereche la donor: solo-ul tace, nu se pierde piesa."""
        gazda = empty_satb(3, TimeSignature(4, 4), KeySignature(0))
        for part in gazda.parts:
            part.measures[0].events = [nota("C", 4)]  # anacruza: o patrime
            part.measures[0].partial = True
            for masura in part.measures[1:]:
                masura.events = [nota("C", 4)] * 4
        cu = cu_solo(gazda, gazda, random.Random(0))
        assert [m.quarter_length for m in cu.part("X1").measures] == [1, 4, 4]
        assert all(not isinstance(e, Note) for e in cu.part("X1").measures[0].events)

    def test_min_voices_cu_alt_numar_de_portative_e_refuzat(self, pagina_cu_solo):
        _, _, _, _, cale = pagina_cu_solo
        with pytest.raises(ValueError, match="min_voices"):
            load_with_layout(cale, min_voices=[1, 2])
