"""Orice partitura cu forma SATB -> partitura stransa (`app/training/satb.py`).

Sursele publice (PDMX, imnare) scriu corul in patru feluri. Testele construiesc
fiecare forma cu `music21`, o trec prin `cor_satb` si cer aceeasi muzica in
S, A, T, B. Apoi `probleme`: o piesa care nu e cor in patru voci trebuie
refuzata cu motiv, fiindca altfel ar intra in antrenare cu o tinta falsa.
"""

import pytest

music21 = pytest.importorskip("music21")

from music21 import chord, instrument, meter, note, stream  # noqa: E402

from app.evaluation.reference import UnsupportedNotation  # noqa: E402
from app.music_model.model import Duration, Note, NoteType, Pitch, Rest, empty_satb  # noqa: E402
from app.training.satb import amprenta, cor_satb, probleme  # noqa: E402

# O masura de 4/4 in fiecare voce, inaltimi tipice SATB.
SOPRAN = ["E5", "D5", "C5", "D5"]
ALTO = ["C5", "B4", "A4", "B4"]
TENOR = ["G3", "G3", "E3", "G3"]
BAS = ["C3", "G2", "A2", "G2"]


def masura(inaltimi, numar=1):
    m = stream.Measure(number=numar)
    if numar == 1:
        m.append(meter.TimeSignature("4/4"))
    for inaltime in inaltimi:
        m.append(note.Note(inaltime, quarterLength=1))
    return m


def parte(inaltimi, instr, masuri=4):
    p = stream.Part()
    p.insert(0, instr)
    for numar in range(1, masuri + 1):
        p.append(masura(inaltimi, numar))
    return p


def deschisa(masuri=4, extra=None):
    s = stream.Score()
    for inaltimi, instr in zip(
        (SOPRAN, ALTO, TENOR, BAS),
        (instrument.Soprano(), instrument.Alto(), instrument.Tenor(), instrument.Bass()),
    ):
        s.insert(0, parte(inaltimi, instr, masuri))
    if extra is not None:
        s.insert(0, extra)
    return s


def pian_cu_acorduri(masuri=4):
    """Imnul scris pe pian: S+A si T+B ca acorduri de doua note, fara voci."""
    s = stream.Score()
    for sus, jos in ((SOPRAN, ALTO), (TENOR, BAS)):
        p = stream.Part()
        p.insert(0, instrument.Piano())
        for numar in range(1, masuri + 1):
            m = stream.Measure(number=numar)
            if numar == 1:
                m.append(meter.TimeSignature("4/4"))
            for a, b in zip(sus, jos):
                m.append(chord.Chord([a, b], quarterLength=1))
            p.append(m)
        s.insert(0, p)
    return s


def inaltimi(score, voce):
    return [str(e.pitch) for m in score.part(voce).measures for e in m.events][:4]


class TestFormele:
    def test_patru_portative_vocale(self):
        score = cor_satb(deschisa())
        assert score.staves == [["S", "A"], ["T", "B"]]
        assert inaltimi(score, "S") == SOPRAN
        assert inaltimi(score, "B") == BAS

    def test_acompaniamentul_se_lasa(self):
        """Patru voci plus pian: pianul nu e a cincea voce, e acompaniament."""
        score = cor_satb(deschisa(extra=parte(["C4"] * 4, instrument.Piano())))
        assert [p.id for p in score.parts] == ["S", "A", "T", "B"]
        assert inaltimi(score, "T") == TENOR

    def test_acordurile_de_doua_note_devin_doua_voci(self):
        """Forma cea mai des intalnita la imnurile urcate pe MuseScore."""
        score = cor_satb(pian_cu_acorduri())
        assert inaltimi(score, "S") == SOPRAN
        assert inaltimi(score, "A") == ALTO
        assert inaltimi(score, "T") == TENOR
        assert inaltimi(score, "B") == BAS

    def test_trei_voci_se_refuza(self):
        s = stream.Score()
        for inaltimi_voce, instr in zip((SOPRAN, ALTO, BAS), (instrument.Soprano(), instrument.Alto(), instrument.Bass())):
            s.insert(0, parte(inaltimi_voce, instr))
        with pytest.raises(UnsupportedNotation, match="3 portative"):
            cor_satb(s)

    def test_acordul_de_trei_note_se_refuza(self):
        s = pian_cu_acorduri()
        s.parts[0].getElementsByClass("Measure")[0].notes[0].add("G4")
        with pytest.raises(UnsupportedNotation, match="trei|mai mult de doua note"):
            cor_satb(s)


def satb_plin(masuri=8, s=("E", 5), a=("C", 5), t=("G", 3), b=("C", 3)):
    score = empty_satb(masuri)
    for voce, (pas, octava) in zip("SATB", (s, a, t, b)):
        for m in score.part(voce).measures:
            m.events = [Note(Pitch(pas, octava), Duration(NoteType.WHOLE))]
    return score


class TestProbleme:
    def test_un_cor_bun_nu_are_probleme(self):
        assert probleme(satb_plin()) == []

    def test_basul_prea_sus_e_prins(self):
        assert any("B: ambitus" in p for p in probleme(satb_plin(b=("C", 5))))

    def test_vocile_inversate_sunt_prinse(self):
        """Tenorul deasupra altistului, toata piesa: nu e SATB, e alta asezare."""
        assert any("ordine" in p for p in probleme(satb_plin(a=("A", 3), t=("E", 4))))

    def test_o_voce_care_tace_e_prinsa(self):
        score = satb_plin()
        for m in score.part("T").measures:
            m.events = [Rest(Duration(NoteType.WHOLE))]
        assert any("T: nicio nota" in p for p in probleme(score))

    def test_doua_voci_dublate_nu_sunt_patru(self):
        """Melodie cu bas, fiecare scrisa de doua ori: patru portative, doua voci."""
        assert any("unison" in p for p in probleme(satb_plin(a=("E", 5), t=("C", 3))))


class TestAmprenta:
    def test_transpozitia_are_aceeasi_amprenta(self):
        """Aceeasi melodie in alta tonalitate trebuie recunoscuta ca aceeasi."""
        unu = cor_satb(deschisa())
        doi = cor_satb(deschisa().transpose(2))
        assert amprenta(unu) == amprenta(doi)
