"""Orice partitura cu forma SATB -> partitura stransa, plus verificarea ei.

`bach.coral_satb` stie un singur fel de intrare: corpusul `music21`, patru
portative curate, cate o voce pe fiecare. Sursele publice mari (PDMX, imnare)
vin in toate formele in care scrie un om in MuseScore:

1. **patru portative vocale** — partitura deschisa, ca la Bach;
2. **doua portative** cu cate doua voci — partitura stransa, forma tinta;
3. **un pian** cu doua portative, pe care s-a scris un imn: doua voci pe
   portativ, sau acorduri de doua note in loc de doua voci;
4. **patru portative vocale plus acompaniament** — acompaniamentul se lasa.

Toate ajung la acelasi `Score` cu vocile S, A, T, B pe portativele
`[["S", "A"], ["T", "B"]]`, gata pentru `bach.musicxml_strans` si `kern_writer`.

**Ce intra nu e crezut pe cuvant.** Metadatele PDMX spun doar ce instrumente
are piesa; un „Choir Aahs" poate canta orice. `probleme` verifica muzica
insasi — ambitusul fiecarei voci, ordinea lor, cat canta fiecare, unisonul
perpetuu — si o piesa cu vreo problema nu intra in antrenare. E aceeasi regula
ca peste tot in proiect: o tinta gresita e mai rea decat una lipsa.
"""

from __future__ import annotations

from fractions import Fraction
from statistics import median
from typing import Any

from app.evaluation.reference import UnsupportedNotation, from_music21
from app.music_model.model import Clef, Note, Score
from app.training.bach import STRANSA, normalizeaza_alteratiile

# Programele MIDI ale vocilor (General MIDI: Choir Aahs, Voice Oohs, Synth
# Voice). Asa le exporta MuseScore pe Soprano, Alto, Tenor, Bass, Choir.
PROGRAME_VOCALE = {52, 53, 54}

# Ambitusul acceptat al fiecarei voci, ca note MIDI. Generos dinadins — un
# sopran care urca la Do6 sau un bas care coboara la La1 exista — dar destul de
# strans ca sa prinda o voce de pian scrisa pe un portativ vocal, sau vocile
# puse in alta ordine decat SATB.
AMBITUS = {
    "S": (55, 84),  # Sol3 - Do6
    "A": (50, 79),  # Re3 - Sol5
    "T": (43, 72),  # Sol2 - Do5
    "B": (33, 67),  # La1 - Sol4
}

# Cat din durata piesei poate tacea o voce. Peste, nu e un cor in patru voci, e
# o piesa cu o voce care intra rar — sau un portativ gol.
TACERE_MAXIMA = Fraction(1, 2)

# Cat din masuri pot avea ambele perechi la unison. Peste, „patru voci" sunt de
# fapt doua, dublate — o melodie cu bas scrisa pe doua portative.
UNISON_MAXIM = Fraction(1, 2)

MASURI = (4, 300)


def _e_vocala(part: Any) -> bool:
    """Portativul e al unei voci? Dupa instrument, apoi dupa programul MIDI."""
    import music21

    instrument = part.getInstrument(returnDefault=False)
    if instrument is None:
        return False
    if isinstance(instrument, (music21.instrument.Vocalist, music21.instrument.Choir)):
        return True
    program = getattr(instrument, "midiProgram", None)
    return program is not None and int(program) in PROGRAME_VOCALE


def cor_satb(m21_score: Any) -> Score:
    """O partitura `music21` in oricare din cele patru forme -> SATB strans.

    Refuza ce nu iese **exact** patru voci: trei (SAB), cinci (SSATB), divisi.
    """
    # Notele ornamentale n-au durata, iar modelul intern nu le reprezinta: ar
    # lungi masura cu o optime care nu exista. Sunt podoaba, nu voce — se scot.
    for nota in list(m21_score.recurse().notes):
        if nota.duration.isGrace:
            nota.activeSite.remove(nota)

    parts = list(m21_score.parts)
    vocale = [part for part in parts if _e_vocala(part)]
    if len(vocale) == 4 and len(parts) > 4:
        # Forma 4: se pastreaza doar vocile, in ordinea de pe pagina.
        for part in parts:
            if part not in vocale:
                m21_score.remove(part)
        parts = vocale

    if len(parts) == 4:
        for part in parts:
            if any(len(m.voices) > 1 for m in part.getElementsByClass("Measure")):
                raise UnsupportedNotation("un portativ din patru are mai multe voci — divisi")
    elif len(parts) == 2:
        for part in parts:
            for measure in part.getElementsByClass("Measure"):
                acorduri = list(measure.recurse().getElementsByClass("Chord"))
                if len(measure.voices) > 2:
                    raise UnsupportedNotation("un portativ are mai mult de doua voci")
                if any(len(chord.pitches) > 2 for chord in acorduri):
                    raise UnsupportedNotation("acord de mai mult de doua note — nu sunt doua voci")
                if len(measure.voices) == 2 and acorduri:
                    raise UnsupportedNotation("acord in interiorul unei voci — nu se stie a cui e nota")
    else:
        raise UnsupportedNotation(f"{len(parts)} portative — nici 4 voci, nici partitura stransa")

    # Pe doua portative, fiecare poarta doua voci chiar cand fisierul n-o spune:
    # acordul de doua note e sopran + alto (tenor + bas), unisonul e o nota.
    score = from_music21(m21_score, min_voices=2 if len(parts) == 2 else 1)
    if [part.id for part in score.parts] != ["S", "A", "T", "B"]:
        raise UnsupportedNotation(f"au iesit vocile {[part.id for part in score.parts]}, nu SATB")

    for part in score.parts:
        cheie = Clef.treble() if part.id in ("S", "A") else Clef.bass()
        for measure in part.measures:
            for event in measure.events:
                if isinstance(event, Note) and abs(event.pitch.alter) > 1:
                    raise UnsupportedNotation(
                        f"{part.id} m{measure.number}: alteratie dubla ({event.pitch})"
                    )
            if measure.clef is not None:
                measure.clef = cheie
        if part.measures and part.measures[0].clef is None:
            part.measures[0].clef = cheie

    score.staves = [list(group) for group in STRANSA]
    normalizeaza_alteratiile(score)
    return score


def probleme(score: Score) -> list[str]:
    """De ce piesa **nu** e un cor SATB folosibil. Lista goala = buna."""
    gasite: list[str] = []
    n = score.measure_count
    if not MASURI[0] <= n <= MASURI[1]:
        gasite.append(f"{n} masuri, in afara plajei {MASURI}")

    mediane = {}
    for part in score.parts:
        inaltimi = [e.pitch.midi for m in part.measures for e in m.events if isinstance(e, Note)]
        total = sum((m.quarter_length for m in part.measures), Fraction(0))
        sunat = sum(
            (e.quarter_length for m in part.measures for e in m.events if isinstance(e, Note)),
            Fraction(0),
        )
        if not inaltimi:
            gasite.append(f"{part.id}: nicio nota")
            continue
        if total and 1 - sunat / total > TACERE_MAXIMA:
            gasite.append(f"{part.id}: tace {float(1 - sunat / total):.0%} din piesa")
        jos, sus = AMBITUS[part.id]
        if min(inaltimi) < jos or max(inaltimi) > sus:
            gasite.append(f"{part.id}: ambitus {min(inaltimi)}-{max(inaltimi)} in afara {jos}-{sus}")
        mediane[part.id] = median(inaltimi)

    if len(mediane) == 4 and not (mediane["S"] >= mediane["A"] >= mediane["T"] >= mediane["B"]):
        gasite.append(f"vocile nu sunt in ordine SATB (mediane {mediane})")

    unison = 0
    for number in range(1, n + 1):
        sus = [score.part(v).measure(number).events for v in ("S", "A")]
        jos = [score.part(v).measure(number).events for v in ("T", "B")]
        if _aceleasi(*sus) and _aceleasi(*jos):
            unison += 1
    if n and Fraction(unison, n) > UNISON_MAXIM:
        gasite.append(f"{unison} din {n} masuri au ambele portative la unison — nu sunt patru voci")
    return gasite


def _aceleasi(a: list, b: list) -> bool:
    """Doua voci canta exact acelasi lucru in masura (inaltimi si durate)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if type(x) is not type(y) or x.quarter_length != y.quarter_length:
            return False
        if isinstance(x, Note) and x.pitch.midi != y.pitch.midi:
            return False
    return True


def amprenta(score: Score, lungime: int = 24) -> tuple[int, ...]:
    """Primele intervale ale sopranului — aceeasi melodie, oricare ar fi tonalitatea.

    Folosita ca sa nu intre in antrenare o piesa care e deja in validare (o
    corala Bach urcata pe MuseScore de cineva) sau chiar pagina pe care se
    masoara. Intervalele, nu inaltimile: o transpozitie e aceeasi melodie.
    """
    inaltimi = [
        e.pitch.midi for m in score.part("S").measures for e in m.events if isinstance(e, Note)
    ]
    return tuple(b - a for a, b in zip(inaltimi, inaltimi[1:]))[:lungime]
