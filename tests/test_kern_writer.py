"""Scriitorul `Score` -> Humdrum, inversul lui `kern.py` (§15.7).

Proba care conteaza e **dus-intorsul**: partitura scrisa de noi, citita inapoi
de cititor, trebuie sa dea aceeasi muzica. Asa scriitorul e verificat de cititor,
iar cititorul e verificat de `music21` (§13.3) — nicio implementare nu se
verifica singura.
"""

from fractions import Fraction
from pathlib import Path

import pytest

from app.evaluation.metrics import compare
from app.music_model.model import (
    Accidental,
    Clef,
    Duration,
    KeySignature,
    Measure,
    Note,
    NoteType,
    Part,
    Pitch,
    Rest,
    Score,
    TimeSignature,
    Tuplet,
)
from app.recognition.bekern import from_kern, to_kern
from app.recognition.kern import parse_open_system
from app.recognition.kern_writer import (
    UnwritableScore,
    to_kern_text,
    write_duration,
    write_event,
    write_key,
    write_pitch,
)
from scripts.make_reference_score import build

REFERINTA = Path(__file__).resolve().parents[1] / "data" / "validation" / "pagina3.musicxml"


class TestTokeni:
    def test_duratele_sunt_reciproce(self):
        assert write_duration(Duration(NoteType.QUARTER)) == "4"
        assert write_duration(Duration(NoteType.WHOLE)) == "1"
        assert write_duration(Duration(NoteType.HALF, dots=1)) == "2."

    def test_trioletul_de_optimi_se_scrie_doisprezece(self):
        """`8 * 3 / 2 = 12` — regula pe care o citeste `parse_duration`."""
        assert write_duration(Duration(NoteType.EIGHTH, tuplet=Tuplet(3, 2))) == "12"

    def test_octavele_urmeaza_conventia_humdrum(self):
        assert write_pitch(Pitch("C", 4)) == "c"
        assert write_pitch(Pitch("C", 5)) == "cc"
        assert write_pitch(Pitch("C", 3)) == "C"
        assert write_pitch(Pitch("C", 2)) == "CC"

    def test_alteratia_scrisa_intra_in_token(self):
        assert write_pitch(Pitch("F", 4, 1, Accidental.SHARP)) == "f#"

    def test_pauzele_si_legaturile(self):
        assert write_event(Rest(Duration(NoteType.QUARTER))) == "4r"
        note = Note(Pitch("C", 4), Duration(NoteType.QUARTER), tie_start=True)
        assert write_event(note) == "[4c"

    def test_armura(self):
        assert write_key(KeySignature(0)) == "*k[]"
        assert write_key(KeySignature(2)) == "*k[f#c#]"
        assert write_key(KeySignature(-1)) == "*k[b-]"


def closed_score() -> Score:
    """Un sistem de partitura stransa, scris in cod: S+A pe Sol, T+B pe Fa.

    `make_reference_score.build()` nu serveste aici: acolo tenorul are cheia Sol
    de tenor si basul cheia Fa, adica patru portative, nu doua.
    """
    score = Score()
    music = {
        "S": [("C", 5), ("D", 5), ("E", 5), ("D", 5)],
        "A": [("E", 4), ("F", 4), ("G", 4), ("F", 4)],
        "T": [("C", 4), ("C", 4), ("B", 3), ("A", 3)],
        "B": [("C", 3), ("A", 2), ("G", 2), ("D", 3)],
    }
    for part_id, clef in (("S", Clef.treble()), ("A", Clef.treble()),
                          ("T", Clef.bass()), ("B", Clef.bass())):
        part = Part(id=part_id, name=part_id)
        for number in (1, 2):
            events = [
                Note(Pitch(step, octave), Duration(NoteType.QUARTER))
                for step, octave in music[part_id]
            ]
            part.measures.append(
                Measure(
                    number=number,
                    events=events,
                    clef=clef if number == 1 else None,
                    key=KeySignature(0) if number == 1 else None,
                    time=TimeSignature(4, 4) if number == 1 else None,
                )
            )
        score.parts.append(part)
    return score


class TestDusIntors:
    def _round_trip(self, score):
        return compare(score, parse_open_system(to_kern_text(score)).score)

    def test_o_partitura_stransa_scrisa_in_cod_trece_neatinsa(self):
        result = self._round_trip(closed_score())
        assert result.exact, result.structure

    def test_patru_portative_sunt_refuzate_explicit(self):
        """`build()` are cheia Sol de tenor si cheia Fa pe acelasi portativ."""
        with pytest.raises(UnwritableScore, match="chei diferite"):
            to_kern_text(build())

    def test_referinta_reala_a_paginii_3(self):
        """Partitura ta din MuseScore, dus si intors prin scriitor."""
        pytest.importorskip("music21")
        from app.evaluation.reference import load

        if not REFERINTA.exists():
            pytest.skip("lipseste referinta paginii 3")
        reference = load(REFERINTA)
        result = self._round_trip(reference)
        assert result.exact, result.structure
        assert result.total.matched == result.total.ref_events == 182

    def test_documentul_are_o_singura_lipire_pe_rand(self):
        """Doua perechi de `*v` alaturate ar deveni o singura coloana."""
        pytest.importorskip("music21")
        from app.evaluation.reference import load

        if not REFERINTA.exists():
            pytest.skip("lipseste referinta paginii 3")
        for line in to_kern_text(load(REFERINTA)).splitlines():
            fields = [f.strip() for f in line.split("\t")]
            if "*v" in fields:
                # un singur sir de `*v`, restul tin locul
                assert fields.count("*v") == 2
                assert set(fields) <= {"*v", "*"}

    def test_vocile_la_unison_impart_o_coloana(self):
        """Asa arata unisonul in Humdrum, si asa il citeste `split_voices`."""
        score = closed_score()
        for part_id in ("T", "B"):
            for measure in score.part(part_id).measures:
                measure.events = [Rest(Duration(NoteType.WHOLE))]
        text = to_kern_text(score)
        assert "*^" in text  # portativul de sus se desparte
        widths = {len(line.split("\t")) for line in text.splitlines() if line.startswith("=")}
        assert widths == {3}  # jos o coloana, sus doua

    def test_o_partitura_cu_masuri_nepotrivite_e_refuzata(self):
        score = closed_score()
        score.part("A").measures.pop()
        with pytest.raises(UnwritableScore, match="aceleasi masuri"):
            to_kern_text(score)


def solo_score(voci_pe_solo: int = 1) -> Score:
    """Trei portative: solo sus, apoi S+A, apoi T+B.

    Forma cantarilor cu voce de solo, despre care stim ca vin (§15.9). Portativul
    de solo poate purta si doua voci, deci se poate cere.
    """
    score = Score()
    solo = [f"X{i}" for i in range(1, voci_pe_solo + 1)]
    score.staves = [solo, ["S", "A"], ["T", "B"]]
    clefs = {part_id: Clef.treble() for part_id in solo + ["S", "A"]}
    clefs.update({"T": Clef.bass(), "B": Clef.bass()})
    # Fiecare voce primeste alta inaltime: doua voci identice sunt la unison, si
    # scriitorul le strange intr-o coloana — corect, dar atunci portativul n-ar
    # mai avea doua coloane de numarat.
    octaves = {"S": 5, "A": 4, "T": 3, "B": 2, "X1": 5, "X2": 4}
    for part_id in solo + ["S", "A", "T", "B"]:
        part = Part(id=part_id, name=part_id)
        for number in (1, 2):
            step = "C" if part_id in ("S", "T", "X1") else "E"
            part.measures.append(
                Measure(
                    number=number,
                    events=[Note(Pitch(step, octaves[part_id]), Duration(NoteType.QUARTER))] * 4,
                    clef=clefs[part_id] if number == 1 else None,
                    key=KeySignature(0) if number == 1 else None,
                    time=TimeSignature(4, 4) if number == 1 else None,
                )
            )
        score.parts.append(part)
    return score


class TestAsezareaCalatorestePartitura:
    """`layout` a fost dintotdeauna general; implicitul era cel care nu era."""

    def test_fara_asezare_se_presupune_partitura_stransa_corala(self):
        # Partiturile construite in cod nu declara nimic si trebuie sa mearga.
        score = closed_score()
        assert score.staves == []
        assert to_kern_text(score) == to_kern_text(score, layout=(("T", "B"), ("S", "A")))

    def test_o_cantare_cu_solo_se_scrie_fara_sa_i_se_spuna(self):
        text = to_kern_text(solo_score())
        widths = {len(line.split("	")) for line in text.splitlines() if line.startswith("=")}
        assert widths == {5}  # T+B, S+A, solo

    def test_portativul_de_solo_poate_avea_doua_voci(self):
        text = to_kern_text(solo_score(voci_pe_solo=2))
        widths = {len(line.split("	")) for line in text.splitlines() if line.startswith("=")}
        assert widths == {6}

    def test_ordinea_e_inversa_fata_de_pagina(self):
        # Humdrum pune portativul de jos in stanga; `staves` merge de sus in jos.
        text = to_kern_text(solo_score())
        clefs = [line for line in text.splitlines() if "clef" in line]
        assert "clefF4" in clefs[0]  # prima coloana e portativul de jos

    def test_asezarea_data_explicit_bate_partitura(self):
        score = solo_score()
        text = to_kern_text(score, layout=(("T", "B"), ("S", "A")))
        widths = {len(line.split("	")) for line in text.splitlines() if line.startswith("=")}
        assert widths == {4}  # solo ignorat, fiindca asa s-a cerut

    def test_taierea_in_sisteme_pastreaza_asezarea(self):
        from app.music_model.model import measures_slice

        assert measures_slice(solo_score(), 1, 1).staves == [["X1"], ["S", "A"], ["T", "B"]]


class TestDusIntorsulCantarilorCuSolo:
    """Proba care lipsea la §15.9: tintele cu solo se scriau, dar nu se verificau."""

    def _round_trip(self, score):
        back = parse_open_system(to_kern_text(score), layout=score.staves).score
        return compare(score, back)

    def test_solo_cu_o_voce_se_inchide_exact(self):
        assert self._round_trip(solo_score()).exact

    def test_solo_cu_doua_voci_se_inchide_exact(self):
        assert self._round_trip(solo_score(voci_pe_solo=2)).exact

    def test_partile_ies_de_sus_in_jos(self):
        score = solo_score()
        back = parse_open_system(to_kern_text(score), layout=score.staves).score
        assert [part.id for part in back.parts] == ["X1", "S", "A", "T", "B"]
        assert back.staves == [["X1"], ["S", "A"], ["T", "B"]]

    def test_partitura_stransa_se_inchide_si_declarata_si_dupa_cheie(self):
        score = closed_score()
        score.staves = [["S", "A"], ["T", "B"]]
        text = to_kern_text(score)
        assert compare(score, parse_open_system(text, layout=score.staves).score).exact
        assert compare(score, parse_open_system(text).score).exact


class TestTokenizare:
    def test_dus_intorsul_prin_tokeni(self):
        document = "**kern\t**kern\n*clefF4\t*clefG2\n=\t=\n4c 4e\t4g\n"
        assert to_kern(from_kern(document)).kern == document.rstrip("\n")

    def test_exponentul_e_cel_pe_care_il_scrie_modelul(self):
        assert from_kern("**kern\n")[0] == "**ekern_1.0"

    def test_lantul_intreg_score_tokeni_score(self):
        """Exact drumul pe care il va parcurge o tinta de antrenare."""
        score = closed_score()
        tokens = from_kern(to_kern_text(score))
        back = parse_open_system(to_kern(tokens).kern).score
        assert compare(score, back).exact


class TestLegatourile:
    """Modelul nu le are in vocabular, deci tintele de antrenare nu le poarta."""

    def _cu_legato(self):
        score = closed_score()
        score.staves = [["S", "A"], ["T", "B"]]
        nota = score.part("S").measures[0].events[0]
        nota.tie_start = True
        score.part("S").measures[0].events[1].tie_stop = True
        return score

    def test_implicit_se_scriu(self):
        # Pentru un om care se uita la partitura, legatoul e informatie.
        assert "[" in to_kern_text(self._cu_legato())

    def test_cu_ties_false_dispar(self):
        text = to_kern_text(self._cu_legato(), ties=False)
        # Armura are si ea paranteze (`*k[]`), deci se cauta in randurile de note.
        note = [line for line in text.splitlines() if line and line[0].isdigit()]
        assert note and not any("[" in line or "]" in line for line in note)

    def test_muzica_ramane_aceeasi(self):
        # Doua note legate sunt tot doua note: se pierde semnul, nu sunetul.
        score = self._cu_legato()
        cu = parse_open_system(to_kern_text(score), layout=score.staves).score
        fara = parse_open_system(to_kern_text(score, ties=False), layout=score.staves).score
        assert compare(cu, fara).exact

    def test_dus_intorsul_se_inchide_si_fara(self):
        score = self._cu_legato()
        text = to_kern_text(score, ties=False)
        assert compare(score, parse_open_system(text, layout=score.staves).score).exact
