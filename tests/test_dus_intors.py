"""Dus-intorsul prin MusicXML trebuie sa fie neutru (PROJECT.md §15.39).

§13 a verificat dus-intorsul pe o referinta scrisa in MuseScore: 182 din 182.
§15.39 a gasit ca pe **iesirea modelului** nu mai e neutru — aceeasi
configuratie da 17 corecte din 38 masurata in conducta si 16 din 42 masurata pe
fisierul salvat cu `--out`. Doua cifre corecte care nu sunt de acord fac
suspecta orice comparatie intre masuratori luate pe drumuri diferite.

Cauza, gasita numarand: o masura **fara niciun eveniment**. Scrisa, iese un
`<measure>` fara nicio nota; citita inapoi, umplutura din `_events_of` o face o
masura de tacere. Deci obiectul avea 0 evenimente acolo si fisierul are 1.

Fisierul are dreptate — o masura fara nimic in ea *e* o masura de tacere, si asa
o arata orice cititor. Ce s-a reparat e obiectul.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.reference import load  # noqa: E402
from app.music_model.model import (  # noqa: E402
    Duration,
    Measure,
    Note,
    NoteType,
    Part,
    Pitch,
    Score,
)
from app.recognition.page import fill_empty_measures  # noqa: E402


def patrime(step: str, octave: int) -> Note:
    return Note(pitch=Pitch(step=step, octave=octave), duration=Duration(NoteType.QUARTER))


def partitura_cu_masura_goala() -> Score:
    """Doua voci, trei masuri, iar masura a doua goala in amandoua.

    Exact forma pe care o produce modelul cand nu scrie nimic pentru un sistem.
    """
    score = Score(title="proba")
    for index, (step, octave) in enumerate((("C", 5), ("E", 4)), start=1):
        part = Part(id=f"P{index}", name=f"vocea {index}")
        part.measures = [
            Measure(number=1, events=[patrime(step, octave)] * 4),
            Measure(number=2, events=[]),
            Measure(number=3, events=[patrime(step, octave)] * 4),
        ]
        score.parts.append(part)
    return score


def evenimente(score: Score) -> int:
    return sum(len(m.events) for p in score.parts for m in p.measures)


class TestMasuraGoala:
    def test_umplerea_adauga_o_masura_de_tacere_pe_voce(self):
        score = partitura_cu_masura_goala()
        assert evenimente(score) == 16
        assert fill_empty_measures(score) == 2
        assert evenimente(score) == 18

    def test_umplerea_e_idempotenta(self):
        score = partitura_cu_masura_goala()
        fill_empty_measures(score)
        assert fill_empty_measures(score) == 0

    def test_masurile_scrise_nu_se_ating(self):
        score = partitura_cu_masura_goala()
        fill_empty_measures(score)
        for part in score.parts:
            assert len(part.measures[0].events) == 4
            assert len(part.measures[2].events) == 4

    def test_umplutura_are_lungimea_masurii(self):
        score = partitura_cu_masura_goala()
        fill_empty_measures(score)
        for part in score.parts:
            total = sum(e.quarter_length for e in part.measures[1].events)
            assert Fraction(total) == Fraction(4)


class TestDusIntors:
    """Proba care conteaza: acelasi numar de evenimente pe obiect si pe fisier."""

    def scrie_si_citeste(self, score: Score, tmp_path: Path) -> Score:
        from app.musicxml.export import write

        cale = tmp_path / "proba.musicxml"
        write(score, cale)
        return load(cale)

    def test_fara_umplere_fisierul_are_mai_multe_evenimente(self, tmp_path):
        """Defectul din §15.39, pastrat ca test ca sa nu se piarda explicatia."""
        score = partitura_cu_masura_goala()
        intors = self.scrie_si_citeste(score, tmp_path)
        assert evenimente(intors) > evenimente(score)
        assert evenimente(intors) - evenimente(score) == 2

    def test_cu_umplere_cele_doua_drumuri_sunt_de_acord(self, tmp_path):
        score = partitura_cu_masura_goala()
        fill_empty_measures(score)
        intors = self.scrie_si_citeste(score, tmp_path)
        assert evenimente(intors) == evenimente(score)

    def test_o_partitura_fara_masuri_goale_era_deja_neutra(self, tmp_path):
        """§13 masurase asta si avea dreptate — defectul cerea o masura goala."""
        score = partitura_cu_masura_goala()
        for part in score.parts:
            part.measures[1].events = [patrime("G", 4)] * 4
        intors = self.scrie_si_citeste(score, tmp_path)
        assert evenimente(intors) == evenimente(score)


class TestPauzeleNuSuntNoteCitite:
    """`corecte` continea si perechile pauza-cu-pauza, fara sa se vada.

    `pitch_key` da `None` pentru orice pauza, deci doua pauze de aceeasi durata
    ies „identice" si intra in `corecte`. Nu e gresit — pauzele sunt in
    referinta, 26 din cele 182 de evenimente ale paginii 3 — dar o pagina de
    tacere care nimereste tacerea referintei nu e o pagina **citita**.

    Masurat pe cea mai buna cifra a proiectului (§15.38, pasul 25 cu
    `--structura`): din 16 „corecte", **4 sunt note** si 12 sunt pauze.
    """

    def compara(self, evenimente_ref, evenimente_hyp):
        from app.evaluation.metrics import compare

        def partitura(evenimente):
            score = Score(title="proba")
            part = Part(id="P1", name="v")
            part.measures = [Measure(number=1, events=list(evenimente))]
            score.parts.append(part)
            return score

        return compare(partitura(evenimente_ref), partitura(evenimente_hyp)).total

    def pauza(self):
        from app.music_model.model import Rest

        return Rest(Duration(NoteType.QUARTER))

    def test_pauzele_potrivite_intra_in_corecte(self):
        t = self.compara([self.pauza()] * 4, [self.pauza()] * 4)
        assert t.matched == 4, "asa se comporta metrica, si asta se documenteaza"

    def test_dar_se_pot_numara_separat(self):
        t = self.compara([self.pauza()] * 4, [self.pauza()] * 4)
        assert t.matched_rests == 4
        assert t.matched_notes == 0, "nicio nota citita"

    def test_notele_adevarate_se_numara_ca_note(self):
        note = [patrime("C", 5)] * 4
        t = self.compara(note, note)
        assert (t.matched_notes, t.matched_rests) == (4, 0)

    def test_amestecul_se_desparte(self):
        ref = [patrime("C", 5), self.pauza(), patrime("E", 5), self.pauza()]
        t = self.compara(ref, list(ref))
        assert t.matched == 4
        assert (t.matched_notes, t.matched_rests) == (2, 2)


class TestNumitorulPeNote:
    """`note%` se raporteaza la note adevarate, nu la evenimente (§15.42).

    Hotararea e a lui Fineas, luata pe 2026-09-04: cifra devine mai mica si mai
    onesta. Pe pagina 3, din 182 de evenimente ale referintei 26 sunt pauze,
    deci numitorul e 156, iar reperul citeste 5 note — 3,2%, nu 8,8%.
    """

    def partitura(self, evenimente, part_id="P1"):
        score = Score(title="proba")
        part = Part(id=part_id, name="v")
        part.measures = [Measure(number=1, events=list(evenimente))]
        score.parts.append(part)
        return score

    def pauza(self):
        from app.music_model.model import Rest

        return Rest(Duration(NoteType.QUARTER))

    def test_pauzele_ies_din_numitor(self):
        from app.evaluation.metrics import compare

        ref = [patrime("C", 5), patrime("E", 5), self.pauza(), self.pauza()]
        t = compare(self.partitura(ref), self.partitura(list(ref))).total
        assert (t.ref_events, t.ref_rests, t.ref_notes) == (4, 2, 2)
        assert t.note_accuracy == 1, "doua note din doua"

    def test_o_pagina_numai_de_tacere_nu_mai_iese_pe_jumatate_citita(self):
        from app.evaluation.metrics import compare

        ref = [patrime("C", 5), patrime("E", 5), self.pauza(), self.pauza()]
        tacere = [self.pauza()] * 4
        t = compare(self.partitura(ref), self.partitura(tacere)).total
        assert t.matched_rests == 2, "cele doua pauze se potrivesc"
        assert t.matched_notes == 0
        assert t.note_accuracy == 0, "nicio nota citita, deci zero"

    def test_vocile_nepereche_isi_aduc_pauzele_in_numitor(self):
        """Regresie: `_all_extra` uita `hyp_rests`, si precizia iesea gresita.

        Modelul inventeaza voci intregi de tacere — pe reper, patru din opt.
        Daca pauzele lor nu se numara, `hyp_notes` iese prea mare si precizia
        prea mica, exact pe cazul pe care §15.42 il face vizibil.
        """
        from app.evaluation.metrics import compare

        ref = self.partitura([patrime("C", 5)])
        hyp = self.partitura([patrime("C", 5)])
        inventata = Part(id="P2", name="fantoma")
        inventata.measures = [Measure(number=1, events=[self.pauza()] * 4)]
        hyp.parts.append(inventata)

        t = compare(ref, hyp).total
        assert t.hyp_events == 5
        assert t.hyp_rests == 4, "pauzele vocii inventate se numara"
        assert t.hyp_notes == 1
        assert t.precision == 1, "singura nota scrisa e corecta"
