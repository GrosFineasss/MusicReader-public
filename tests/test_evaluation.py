"""Masurarea acuratetei — verificata prin greseli injectate deliberat.

Acelasi principiu ca la validare: nu ne multumim ca harnessul „da niste cifre",
ci stricam de fiecare data **exact un lucru** si verificam ca se misca **exact**
metrica potrivita, si nicio alta. Un instrument de masura care raporteaza
diferente reale, dar in coloana gresita, e mai rau decat niciunul: ne-ar trimite
sa reparam ritmul cand de fapt e citita gresit armura.

Cazul cel mai important e `test_masura_stearsa`. Fara alinierea pe doua
niveluri, o masura lipsa ar decala tot ce urmeaza si ar raporta o pagina
intreaga gresita in loc de o singura eroare.
"""

from fractions import Fraction
from pathlib import Path

import pytest

from app.evaluation import OpKind, align
from app.evaluation.metrics import (
    Category,
    classify,
    clef_name,
    compare,
    event_cost,
    pitch_key,
)
from app.music_model.model import (
    Accidental,
    Duration,
    Note,
    NoteType,
    Part,
    Pitch,
    Rest,
    Score,
)
from scripts.make_reference_score import build

Q = Duration(NoteType.QUARTER)
H = Duration(NoteType.HALF)


# --------------------------------------------------------------------------
# Alinierea, in sine
# --------------------------------------------------------------------------


def unit_cost(a, b) -> Fraction:
    return Fraction(0) if a == b else Fraction(1)


def referinta_evenimente(score: Score) -> int:
    return sum(len(m.events) for p in score.parts for m in p.measures)


def test_siruri_identice_dau_numai_potriviri():
    ops = align("abc", "abc", unit_cost)
    assert [op.kind for op in ops] == [OpKind.MATCH] * 3


def test_substituirea_e_preferata_perechii_stergere_inserare():
    """Un element schimbat inseamna *una* diferita, nu una lipsa si una in plus.

    E regula pe care se sprijina despartirea pitch/ritm: daca alinierea ar
    raporta stergere+inserare, n-am mai putea spune *cu ce* difera nota.
    """
    ops = align("a", "b", unit_cost)
    assert [op.kind for op in ops] == [OpKind.SUBSTITUTE]


def test_lipsa_si_in_plus():
    assert [op.kind for op in align("ab", "b", unit_cost)] == [OpKind.MISSING, OpKind.MATCH]
    assert [op.kind for op in align("b", "ab", unit_cost)] == [OpKind.EXTRA, OpKind.MATCH]


def test_siruri_goale():
    assert align("", "", unit_cost) == []
    assert [op.kind for op in align("", "ab", unit_cost)] == [OpKind.EXTRA] * 2
    assert [op.kind for op in align("ab", "", unit_cost)] == [OpKind.MISSING] * 2


def test_alinierea_e_determinista():
    """Aceleasi date, acelasi raport — altfel doua masuratori nu se pot compara."""
    first = align("abcd", "axcd", unit_cost)
    second = align("abcd", "axcd", unit_cost)
    assert first == second


# --------------------------------------------------------------------------
# Clasificarea unei perechi de evenimente
# --------------------------------------------------------------------------


def test_categoriile():
    do = Note(Pitch("C", 4), Q)
    assert classify(do, Note(Pitch("C", 4), Q)) is Category.MATCHED
    assert classify(do, Note(Pitch("D", 4), Q)) is Category.PITCH_WRONG
    assert classify(do, Note(Pitch("C", 4), H)) is Category.DURATION_WRONG
    assert classify(do, Note(Pitch("D", 4), H)) is Category.BOTH_WRONG


def test_pauza_pusa_in_locul_unei_note_e_greseala_de_inaltime():
    """O pauza nu are inaltime, deci nu se potriveste niciodata cu o nota."""
    assert classify(Note(Pitch("C", 4), Q), Rest(Q)) is Category.PITCH_WRONG
    assert classify(Rest(Q), Rest(Q)) is Category.MATCHED


def test_alteratia_scrisa_nu_intra_in_comparatie():
    """Aceeasi nota, notata cu si fara alteratie de curtoazie, e aceeasi nota."""
    scris = Pitch("F", 4, 1, Accidental.SHARP)
    nescris = Pitch("F", 4, 1)
    assert pitch_key(Note(scris, Q)) == pitch_key(Note(nescris, Q))
    # `alter` insa conteaza: fa-diez citit ca fa e o greseala adevarata.
    assert pitch_key(Note(scris, Q)) != pitch_key(Note(Pitch("F", 4), Q))


def test_costul_e_intre_zero_si_unu():
    do = Note(Pitch("C", 4), Q)
    assert event_cost(do, Note(Pitch("C", 4), Q)) == 0
    assert event_cost(do, Note(Pitch("D", 4), Q)) == Fraction(1, 2)
    assert event_cost(do, Note(Pitch("D", 4), H)) == 1


# --------------------------------------------------------------------------
# Partitura fata de ea insasi
# --------------------------------------------------------------------------


def test_identitatea_da_totul_corect():
    result = compare(build(), build())
    total = result.total
    assert result.exact
    assert not result.structure
    assert total.matched == total.ref_events == 22
    assert (total.missing, total.extra, total.substitutions) == (0, 0, 0)
    assert total.measures_exact == 8
    assert total.note_accuracy == 1
    assert total.ser == 0


def test_referinta_de_pe_disc_nu_are_masuri_neinchise():
    """Daca referinta insasi ar fi gresita, toate cifrele de dupa ar minti."""
    assert compare(build(), build()).ref_findings == []


# --------------------------------------------------------------------------
# Cate o singura greseala, pe rand
# --------------------------------------------------------------------------


def test_o_inaltime_gresita_misca_doar_coloana_de_inaltime():
    hypothesis = build()
    hypothesis.part("S").measures[0].events[0].pitch = Pitch("B", 4)

    total = compare(build(), hypothesis).total
    assert total.pitch_wrong == 1
    assert total.matched == 21
    assert (total.duration_wrong, total.both_wrong, total.missing, total.extra) == (0, 0, 0, 0)
    # Ritmul a ramas neatins — exact separarea pentru care exista cele doua cifre.
    assert total.rhythm_accuracy == 1
    assert total.pitch_accuracy == Fraction(21, 22)


def test_o_durata_gresita_misca_doar_coloana_de_ritm():
    hypothesis = build()
    hypothesis.part("S").measures[0].events[0].duration = H

    result = compare(build(), hypothesis)
    total = result.total
    assert total.duration_wrong == 1
    assert total.matched == 21
    assert (total.pitch_wrong, total.both_wrong, total.missing, total.extra) == (0, 0, 0, 0)
    assert total.pitch_accuracy == 1
    # Masura nu se mai inchide la 4/4 — si asta trebuie sa se vada, separat.
    assert len(result.hyp_findings) == 1
    assert result.ref_findings == []


def test_o_nota_stearsa_se_numara_o_singura_data():
    hypothesis = build()
    del hypothesis.part("A").measures[0].events[1]

    total = compare(build(), hypothesis).total
    assert total.missing == 1
    assert total.matched == 21
    assert (total.substitutions, total.extra) == (0, 0)


def test_o_nota_in_plus_se_numara_o_singura_data():
    hypothesis = build()
    hypothesis.part("B").measures[1].events.insert(1, Note(Pitch("D", 3), Q))

    total = compare(build(), hypothesis).total
    assert total.extra == 1
    assert total.matched == 22
    assert (total.substitutions, total.missing) == (0, 0)


def test_masura_stearsa_nu_decaleaza_restul():
    """Testul care justifica alinierea pe doua niveluri.

    Vocea T are 3 note in masura 2. Daca masura dispare, se pierd fix acele 3
    note; masurile ramase trebuie sa se potriveasca in continuare, iar celelalte
    trei voci nu au voie sa fie atinse deloc.
    """
    hypothesis = build()
    del hypothesis.part("T").measures[1]

    result = compare(build(), hypothesis)
    tenor = next(v for v in result.voices if v.part_id == "T")
    assert tenor.measures_missing == 1
    assert tenor.missing == 3
    assert tenor.measures_exact == 1
    assert (tenor.substitutions, tenor.extra) == (0, 0)

    for voice in result.voices:
        if voice.part_id != "T":
            assert voice.exact, f"vocea {voice.part_id} nu trebuia atinsa"

    total = result.total
    assert total.matched == 19
    assert total.measures_exact == 7


def test_prima_nota_stearsa_nu_propaga_erori_in_masura():
    """Fara aliniere, o nota lipsa la inceput ar face toata masura gresita."""
    hypothesis = build()
    del hypothesis.part("T").measures[0].events[0]

    tenor = next(v for v in compare(build(), hypothesis).voices if v.part_id == "T")
    assert tenor.missing == 1
    assert tenor.substitutions == 0
    assert tenor.matched == 6


# --------------------------------------------------------------------------
# Greseli de structura — se raporteaza, nu se ghicesc
# --------------------------------------------------------------------------


def test_o_voce_lipsa_e_pierdere_totala_nu_o_voce_ignorata():
    """A sari peste vocea neprodusa ar umfla acuratetea exact in cazul grav."""
    hypothesis = build()
    hypothesis.parts = [p for p in hypothesis.parts if p.id != "B"]

    result = compare(build(), hypothesis)
    assert any("numar de voci diferit" in item for item in result.structure)
    bass = next(v for v in result.voices if v.part_id == "B")
    assert bass.missing == bass.ref_events == 3
    assert bass.matched == 0
    assert bass.measures_missing == 2
    assert result.total.matched == 19


def test_o_voce_inventata_se_numara_ca_in_plus():
    hypothesis = build()
    extra = Part(id="X", name="inventata")
    extra.measures.append(build().part("S").measures[0])
    hypothesis.parts.append(extra)

    result = compare(build(), hypothesis)
    assert any("numar de voci diferit" in item for item in result.structure)
    assert result.total.extra == 3
    assert result.total.matched == 22


def test_ordinea_partilor_nu_conteaza_daca_id_urile_se_potrivesc():
    """Imperecherea dupa id e un fapt, nu o presupunere — deci e sigura."""
    hypothesis = build()
    hypothesis.parts.reverse()

    result = compare(build(), hypothesis)
    assert result.exact
    assert not result.structure


def test_id_uri_diferite_duc_la_imperechere_pozitionala_anuntata():
    hypothesis = build()
    for part in hypothesis.parts:
        part.id = part.id.lower()

    result = compare(build(), hypothesis)
    assert any("imperechere pozitionala" in item for item in result.structure)
    # Muzica e aceeasi, deci notele se potrivesc; doar botezul difera.
    assert result.total.matched == 22
    assert not result.exact


def test_armura_gresita_se_raporteaza_separat_de_note():
    """Un singur rand de context explica de obicei zeci de note gresite."""
    from app.music_model.model import KeySignature

    hypothesis = build()
    hypothesis.part("S").measures[0].key = KeySignature(3)

    result = compare(build(), hypothesis)
    soprano = next(v for v in result.voices if v.part_id == "S")
    assert any("armura" in item for item in soprano.context)
    assert soprano.matched == 6  # notele nu s-au schimbat
    assert not soprano.exact
    assert not result.exact


def test_cheia_de_tenor_nu_se_confunda_cu_cheia_sol():
    """Gasit pe o rulare reala: raportul spunea „referinta G2, transcriere G2".

    Cheia Sol de tenor *este* G2, doar cu octava schimbata, deci fara sufix
    mesajul arata o nepotrivire intre doua lucruri scrise identic — si te
    trimite sa cauti exact unde nu e problema.
    """
    from app.music_model.model import Clef

    assert clef_name(Clef.treble()) == "G2"
    assert clef_name(Clef.tenor_vocal()) == "G2/8vb"
    assert clef_name(Clef.bass()) == "F4"

    hypothesis = build()
    hypothesis.part("T").measures[0].clef = Clef.treble()
    tenor = next(v for v in compare(build(), hypothesis).voices if v.part_id == "T")
    assert tenor.context == ["cheie: referinta G2/8vb, transcriere G2"]


def test_partitura_goala_fata_de_una_plina():
    result = compare(build(), Score())
    total = result.total
    assert total.matched == 0
    assert total.missing == 22
    assert total.note_accuracy == 0
    assert total.ser == 1


def test_ser_singur_pune_muta_inaintea_vorbaretei():
    """SER-ul nu deosebeste „citeste mai bine" de „a amutit" (§15.28).

    Gasit pe o curba de antrenare adevarata: la pasul 50, SER-ul cadea de la
    1,64 la 0,94 si parea cel mai bun punct al rularii — dar modelul scria 14
    note din 182, iar cifra buna venea din faptul ca nu mai inventa nimic.

    Testul fixeaza ordinea rea: o transcriere aproape muta, cu **o** nota
    corecta din 22, iese inaintea uneia care le nimereste pe **toate** dar
    inventeaza in plus. Cine citeste SER-ul fara `note_transcriere` alege muta.
    """
    referinta = build()

    muta = build()
    for part in muta.parts:
        for measure in part.measures:
            measure.events.clear()
    muta.part("S").measures[0].events.append(referinta.part("S").measures[0].events[0])

    vorbareata = build()
    for _ in range(30):
        vorbareata.part("B").measures[1].events.append(Note(Pitch("D", 3), Q))

    muta_total = compare(referinta, muta).total
    vorbareata_total = compare(referinta, vorbareata).total

    assert (muta_total.matched, muta_total.hyp_events) == (1, 1)
    assert (vorbareata_total.matched, vorbareata_total.extra) == (22, 30)

    # Ordinea pe care o da SER-ul — si care e pe dos fata de utilitate.
    assert muta_total.ser < 1 < vorbareata_total.ser

    # Cifra care le desparte corect: cat scrie modelul, nu doar cat nimereste.
    assert muta_total.hyp_events < referinta_evenimente(referinta) / 2


def test_nici_precizia_singura_nu_ajunge():
    """Perechea desparte cazurile; niciuna din cifre nu o face singura.

    Continuarea testului de deasupra, si corectia unei concluzii comode: dupa
    ce SER-ul s-a dovedit pacalit de un model amutit, precizia pare raspunsul
    — dar e pacalita de acelasi model, in aceeasi directie. Muta scrie o nota
    si o nimereste: precizie 100%. Vorbareata le nimereste pe toate cele 22:
    precizie 42%. Clasamentul pe precizie e tot pe dos.

    Ce le desparte e ca **amandoua** cifrele trebuie sa fie mari deodata:
    `note_accuracy` cade la muta (1 din 22), precizia cade la vorbareata. De
    aceea raportul le pune una langa alta si nu alege una drept „cifra" (§15.28).
    """
    referinta = build()

    muta = build()
    for part in muta.parts:
        for measure in part.measures:
            measure.events.clear()
    muta.part("S").measures[0].events.append(referinta.part("S").measures[0].events[0])

    vorbareata = build()
    for _ in range(30):
        vorbareata.part("B").measures[1].events.append(Note(Pitch("D", 3), Q))

    muta_total = compare(referinta, muta).total
    vorbareata_total = compare(referinta, vorbareata).total

    # Precizia, singura, pune tot muta prima — la fel ca SER-ul.
    assert muta_total.precision == 1
    # 21 din 51, nu 22 din 52: de la §15.42 incoace precizia numara **note**,
    # nu evenimente, iar referinta asta are o pauza de fiecare parte. Cifra s-a
    # schimbat, concluzia testului nu.
    assert vorbareata_total.precision == Fraction(21, 51)
    assert muta_total.precision > vorbareata_total.precision

    # Perechea insa nu se lasa pacalita: fiecare model cade pe cealalta cifra.
    # 1 din 21 de **note**, nu din 22 de evenimente — aceeasi schimbare ca mai sus.
    assert muta_total.note_accuracy == Fraction(1, 21)
    assert vorbareata_total.note_accuracy == 1


def test_transcrierea_goala_are_precizia_zero():
    """Golul nu e „nu s-a inselat niciodata", e „n-a nimerit nimic".

    Marginea care ar face precizia inutila exact in cazul pentru care a fost
    adaugata: 0 din 0 e 0 aici, nu 1. Singurul gol care se citeste ca reusita
    e cel fata de o referinta goala.
    """
    assert compare(build(), Score()).total.precision == 0
    assert compare(Score(), Score()).total.precision == 1


def test_raportul_spune_cand_modelul_tace(capsys):
    """Avertismentul care apara cifra de o citire grabita (§15.28).

    Coloana `scrise` e in tabel pentru cine se uita la tabel. Randul asta e
    pentru cine se uita doar la SER — cazul real, fiindca SER-ul e cifra pe
    care o raporteaza literatura si pe care o cauta ochiul.
    """
    from scripts.evaluate import report

    referinta = build()
    muta = build()
    for part in muta.parts:
        for measure in part.measures:
            measure.events.clear()
    muta.part("S").measures[0].events.append(referinta.part("S").measures[0].events[0])

    report(compare(referinta, muta))
    iesire = capsys.readouterr().out
    assert "din pagina lipseste muzica" in iesire
    assert "scrise" in iesire

    # Controlul: o transcriere completa nu primeste avertismentul, altfel
    # randul ar deveni zgomot pe care nimeni nu-l mai citeste.
    report(compare(referinta, build()))
    assert "din pagina lipseste muzica" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# Citirea referintei — cere `music21`, implementarea independenta
# --------------------------------------------------------------------------


def test_dus_intors_prin_musicxml_e_perfect(tmp_path):
    """Export, import si metrica, verificate dintr-o singura miscare.

    Daca oricare dintre cele trei greseste, comparatia nu mai iese exacta.
    """
    pytest.importorskip("music21")
    from app.evaluation.reference import load
    from app.musicxml import export

    original = build()
    path = export.write(original, tmp_path / "referinta.musicxml")

    result = compare(original, load(path))
    assert result.exact
    assert result.total.matched == 22
    assert [p.id for p in load(path).parts] == ["S", "A", "T", "B"]


def test_partitura_stransa_se_desface_in_patru_voci(tmp_path):
    """Forma pe care o exporta MuseScore: 2 portative, cate doua voci.

    Masura fara voci explicite se scrie in **amandoua** vocile portativului —
    regula unisonului, aceeasi ca la `*v` in `kern.py`. Fara ea, vocea a doua
    ar aparea cu masuri lipsa care nu lipsesc.
    """
    m21 = pytest.importorskip("music21")
    from app.evaluation.reference import load

    def staff(clef, split_measure, unison_measure):
        part = m21.stream.Part()
        first = m21.stream.Measure(number=1)
        first.insert(0, clef)
        first.insert(0, m21.meter.TimeSignature("4/4"))
        first.insert(0, m21.key.KeySignature(0))
        for voice_id, names in zip(("1", "2"), split_measure):
            voice = m21.stream.Voice(id=voice_id)
            for name in names:
                voice.append(m21.note.Note(name, type="half"))
            first.insert(0, voice)
        second = m21.stream.Measure(number=2)
        second.append(m21.note.Note(unison_measure, type="whole"))
        part.append(first)
        part.append(second)
        return part

    score = m21.stream.Score()
    score.insert(0, staff(m21.clef.TrebleClef(), (["C5", "D5"], ["E4", "F4"]), "G4"))
    score.insert(0, staff(m21.clef.BassClef(), (["C4", "B3"], ["C3", "G2"]), "C3"))
    path = tmp_path / "stransa.musicxml"
    score.write("musicxml", fp=str(path))

    loaded = load(path)
    assert [p.id for p in loaded.parts] == ["S", "A", "T", "B"]
    assert [str(e.pitch) for e in loaded.part("S").measures[0].events] == ["C5", "D5"]
    assert [str(e.pitch) for e in loaded.part("A").measures[0].events] == ["E4", "F4"]
    # Masura la unison ajunge in amandoua vocile portativului.
    assert [str(e.pitch) for e in loaded.part("S").measures[1].events] == ["G4"]
    assert [str(e.pitch) for e in loaded.part("A").measures[1].events] == ["G4"]
    assert [str(e.pitch) for e in loaded.part("B").measures[1].events] == ["C3"]
    assert loaded.part("B").measures[0].clef.sign == "F"


def test_trioletele_se_citesc_din_referinta(tmp_path):
    """Trioletele sunt obisnuite in partiturile reale, deci referinta le contine.

    Cat timp cititorul le refuza, pagina scrisa in MuseScore nu poate fi nici
    macar citita — iar fara ea nu exista nicio masuratoare de acuratete.
    """
    m21 = pytest.importorskip("music21")
    from app.evaluation.reference import load

    part = m21.stream.Part()
    measure = m21.stream.Measure(number=1)
    measure.insert(0, m21.meter.TimeSignature("4/4"))
    for name in ("C4", "D4", "E4"):
        triplet = m21.duration.Duration(type="quarter")
        triplet.appendTuplet(m21.duration.Tuplet(3, 2, "quarter"))
        measure.append(m21.note.Note(name, duration=triplet))
    measure.append(m21.note.Note("F4", type="half"))
    part.append(measure)
    score = m21.stream.Score()
    score.insert(0, part)
    path = tmp_path / "triolet.musicxml"
    score.write("musicxml", fp=str(path))

    loaded = load(path)
    events = loaded.parts[0].measures[0].events
    assert [e.duration.quarter_length for e in events[:3]] == [Fraction(2, 3)] * 3
    # Trei patrimi de triolet plus o doime umplu exact masura de 4/4 — deci
    # validatorul nu trebuie sa raporteze nimic.
    assert loaded.parts[0].measures[0].quarter_length == 4
    assert compare(loaded, loaded).exact


def test_trioletele_supravietuiesc_drumului_complet(tmp_path):
    """Model intern -> MusicXML -> `music21` -> model intern, cu triolete.

    Verifica dintr-o singura miscare `<time-modification>`, `divisions` (care
    trebuie sa devina multiplu de 3 ca duratele sa ramana intregi) si citirea
    grupului inapoi. Daca oricare greseste, comparatia nu mai iese exacta.
    """
    pytest.importorskip("music21")
    from app.evaluation.reference import load
    from app.music_model.model import Tuplet
    from app.musicxml import export

    triplet = Duration(NoteType.QUARTER, 0, Tuplet(3, 2))
    original = build()
    # Trei patrimi de triolet (2/3 fiecare) plus o doime umplu exact 4/4.
    original.part("S").measures[0].events = [
        Note(Pitch("C", 5), triplet),
        Note(Pitch("D", 5), triplet),
        Note(Pitch("E", 5), triplet),
        Note(Pitch("F", 5), H),
    ]
    assert original.part("S").measures[0].quarter_length == 4

    path = export.write(original, tmp_path / "cu-triolete.musicxml")
    result = compare(original, load(path))
    assert result.exact, result.structure
    assert result.ref_findings == []


def test_notatia_din_afara_scopului_e_refuzata_explicit(tmp_path):
    """Mai bine o eroare clara decat o referinta citita aproximativ.

    O referinta aproximata ar produce diferente raportate in dreptul modelului,
    care n-are nicio vina — adica exact genul de cifra in care nu poti avea
    incredere.
    """
    m21 = pytest.importorskip("music21")
    from app.evaluation.reference import UnsupportedNotation, load

    part = m21.stream.Part()
    measure = m21.stream.Measure(number=1)
    measure.insert(0, m21.meter.TimeSignature("4/4"))
    measure.append(m21.note.Note("C4", type="64th"))
    measure.append(m21.note.Note("D4", type="whole"))
    part.append(measure)
    score = m21.stream.Score()
    score.insert(0, part)
    path = tmp_path / "prea-scurt.musicxml"
    score.write("musicxml", fp=str(path))

    with pytest.raises(UnsupportedNotation, match="nu e reprezentabila"):
        load(path)


# --------------------------------------------------------------------------
# Referinta reala: pagina 3 din manuscrisul din §14, scrisa in MuseScore
# --------------------------------------------------------------------------

PAGINA3 = Path(__file__).resolve().parents[1] / "data" / "validation" / "pagina3.musicxml"


@pytest.fixture(scope="module")
def pagina3():
    pytest.importorskip("music21")
    from app.evaluation.reference import load

    if not PAGINA3.exists():
        pytest.skip(f"lipseste {PAGINA3}")
    return load(PAGINA3)


def test_pagina3_are_patru_voci_de_cate_paisprezece_masuri(pagina3):
    assert [part.id for part in pagina3.parts] == ["S", "A", "T", "B"]
    assert {len(part.measures) for part in pagina3.parts} == {14}


def test_toate_vocile_paginii_au_aceleasi_lungimi_de_masura(pagina3):
    """Golurile scrise ca `<forward>` faceau alto sa iasa cu 2 patrimi in m5.

    Vocile unei partituri stranse canta aceleasi masuri; daca lungimile
    difera intre ele, undeva s-a pierdut timp la citire — si comparatia ar
    raporta note lipsa care nu lipsesc.
    """
    lengths = {
        part.id: [measure.quarter_length for measure in part.measures]
        for part in pagina3.parts
    }
    assert len(set(map(tuple, lengths.values()))) == 1, lengths
    assert lengths["A"] == [Fraction(n) for n in (1, 4, 4, 4, 4, 4, 4, 4, 3, 1, 4, 4, 4, 4)]


def test_masurile_scurte_ale_paginii_sunt_marcate_ca_atare(pagina3):
    """m1 are 1 timp, m9 are 3, m10 are 1 — scrise asa dinadins."""
    for part in pagina3.parts:
        assert [m.number for m in part.measures if m.partial] == [1, 9, 10]


def test_referinta_paginii_nu_are_masuri_care_nu_se_inchid(pagina3):
    """Altfel cifrele de acuratete ar judeca modelul pentru scrierea paginii."""
    assert compare(pagina3, pagina3).ref_findings == []


def test_referinta_paginii_se_compara_perfect_cu_ea_insasi(pagina3):
    result = compare(pagina3, pagina3)
    assert result.exact, result.structure
    assert result.total.note_accuracy == 1


def test_anacruza_a_doua_e_raportata_ca_neimperecheata(pagina3):
    """m10 incepe repetitia a doua, dar masura finala e intreaga.

    Nu e o eroare de citire, e ce scrie pe hartie — deci avertisment, si
    raportat, nu inghitit.
    """
    from app.validation.rules import check_partial_measures_pair_up

    found = check_partial_measures_pair_up(pagina3)
    assert {(f.part_id, f.measure) for f in found} == {
        ("S", 10), ("A", 10), ("T", 10), ("B", 10)
    }


def test_o_masura_scurta_dintr_o_transcriere_nu_trece_drept_anacruza(tmp_path):
    """MusicXML nu pastreaza diferenta dintre „scurta dinadins" si „gresita".

    Deci cine citeste trebuie s-o declare. Fara asta, aceeasi transcriere
    raporta 79 de masuri neinchise citita din memorie si 9 citita de pe disc —
    o cifra care depindea de drumul pe care venise, nu de muzica.
    """
    pytest.importorskip("music21")
    from app.evaluation.reference import load
    from app.musicxml import export
    from app.validation.rules import check_measure_durations

    score = build()
    score.part("S").measures[0].events = score.part("S").measures[0].events[:1]
    path = export.write(score, tmp_path / "scurta.musicxml")

    incredere = load(path)
    assert check_measure_durations(incredere) == []

    neincredere = load(path, trust_short_measures=False)
    assert [f.part_id for f in check_measure_durations(neincredere)] == ["S"]


# --------------------------------------------------------------------------
# Semnatura erorilor si taierea in sisteme (§15.7)
# --------------------------------------------------------------------------


def test_semnatura_numara_confuzia_de_durate():
    """Cifra „11% corecte" nu spune ce sa reparam; distributia spune."""
    hypothesis = build()
    measure = hypothesis.part("S").measures[0]
    measure.events[0] = Note(measure.events[0].pitch, Duration(NoteType.HALF))
    signature = compare(build(), hypothesis).signature
    assert signature.durations[("1", "2")] == 1


def test_semnatura_numara_decalajul_de_inaltime():
    """Un decalaj **constant** ar fi bug determinist, nu greseala de citire."""
    hypothesis = build()
    measure = hypothesis.part("S").measures[0]
    original = measure.events[0]
    measure.events[0] = Note(Pitch("D", 5), original.duration)
    signature = compare(build(), hypothesis).signature
    assert signature.pitches[1] == 1


def test_notele_corecte_nu_intra_in_semnatura():
    signature = compare(build(), build()).signature
    assert not signature.durations and not signature.pitches


def test_rupturile_de_sistem_se_citesc_din_musicxml(pagina3):
    """De ele atarna imperecherea decupaj <-> masuri, deci datele de antrenare."""
    pytest.importorskip("music21")
    from app.evaluation.reference import load_with_layout

    if not PAGINA3.exists():
        pytest.skip(f"lipseste {PAGINA3}")
    score, starts, pages = load_with_layout(PAGINA3)
    assert starts == [1, 6, 12]
    assert pages == [1]
    assert score.measure_count == 14
    # Asezarea pe portative calatoreste cu partitura, ca scriitorul sa nu mai
    # presupuna SATB (§15.9).
    assert score.staves == [["S", "A"], ["T", "B"]]


def test_ruptura_de_pagina_porneste_si_un_sistem():
    """MuseScore scrie doar `new-page="yes"`, fara `new-system` alaturi.

    Regresie: pe referinta de doua pagini a piesei „Doamne bun asculta" ieseau
    9 sisteme in loc de 10, iar primul sistem al paginii a doua se lipea de
    ultimul al paginii intai. Perechea (decupaj, tinta) construita asa ar fi
    invatat modelul alte note decat cele din imagine, si n-ar fi avut cine sa
    observe. Pe pagina 3 — o singura pagina — defectul nu se vedea.
    """
    music21 = pytest.importorskip("music21")
    from app.evaluation.reference import system_starts

    part = music21.stream.Part()
    for number in range(1, 5):
        measure = music21.stream.Measure(number=number)
        if number == 3:
            measure.insert(0, music21.layout.SystemLayout(isNew=True))
        if number == 4:
            # Doar ruptura de pagina, cum o scrie MuseScore.
            measure.insert(0, music21.layout.PageLayout(isNew=True))
        measure.append(music21.note.Note("C4", type="whole"))
        part.append(measure)
    score = music21.stream.Score()
    score.insert(0, part)

    assert system_starts(score) == [1, 3, 4]


def test_paginile_se_citesc_separat_de_sisteme():
    """`dataset.py` imperecheaza o poza de **o pagina** cu masurile ei.

    Referinta poate acoperi mai multe, deci trebuie sa se stie unde se rup —
    altfel o piesa de doua pagini nu poate produce perechi de antrenare deloc.
    """
    music21 = pytest.importorskip("music21")
    from app.evaluation.reference import page_starts, system_starts

    part = music21.stream.Part()
    for number in range(1, 6):
        measure = music21.stream.Measure(number=number)
        if number == 3:
            measure.insert(0, music21.layout.SystemLayout(isNew=True))
        if number == 4:
            measure.insert(0, music21.layout.PageLayout(isNew=True))
        measure.append(music21.note.Note("C4", type="whole"))
        part.append(measure)
    score = music21.stream.Score()
    score.insert(0, part)

    assert page_starts(score) == [1, 4]
    # Ruptura de pagina e si inceput de sistem, deci apare in amandoua listele.
    assert system_starts(score) == [1, 3, 4]


def test_rupturile_se_numara_dupa_pozitie_nu_dupa_numarul_scris():
    """Rupturile sunt in aceleasi unitati ca masurile partiturii: 1..n, la rand.

    Regresie (2026-10-04): MuseScore numeste anacruza 0 si sare peste numar la
    masurile scurte. Referinta „Doamne bun asculta" are numerele 0, 0, 1, 2, ...,
    iar rupturile se intorceau cu numarul scris (5, 10, ..., pagina la 23) pe
    cand `from_music21` numara 1..n (7, 12, ..., pagina la 25). Fiecare sistem
    primea ca tinta masurile altui loc din pagina, decalate cu doua.
    """
    music21 = pytest.importorskip("music21")
    from app.evaluation.reference import page_starts, system_starts

    part = music21.stream.Part()
    # anacruza 0, o masura scurta care nu se numara (tot 0), apoi 1, 2, 3
    for number, layout in ((0, None), (0, None), (1, "sistem"), (2, None), (3, "pagina")):
        measure = music21.stream.Measure(number=number)
        if layout == "sistem":
            measure.insert(0, music21.layout.SystemLayout(isNew=True))
        if layout == "pagina":
            measure.insert(0, music21.layout.PageLayout(isNew=True))
        measure.append(music21.note.Note("C4", type="whole"))
        part.append(measure)
    score = music21.stream.Score()
    score.insert(0, part)

    assert system_starts(score) == [1, 3, 5]
    assert page_starts(score) == [1, 5]
