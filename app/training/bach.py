"""Coralele lui Bach -> pagini SATB tiparite, cu tinta stiuta prin constructie.

De ce exista. §15.53-§15.57 au aratat ca **tiparitul citeste, scrisul de mana
nu**: 29,5% pe `pagina3` randata, 37,7% pe scanuri reale. Dar modelul,
`smt-grandstaff`, a fost antrenat pe **pian** (GrandStaff). Partitura stransa
corala are alta forma: doua voci pe fiecare portativ, cu codite opuse, si cele
doua portative cantand patru linii independente. Coralele lui Bach sunt exact
forma asta, si sunt cel mai mare corpus SATB curat care exista.

Sursa e corpusul `music21`, deja instalat: 364 de corale in 4 voci, 5.739 de
masuri. **Licenta editiei e neclara pentru uz comercial** (editia Greentree,
distribuita cu permisiune impreuna cu `music21`), deci regimul e cel din §3.6:
diagnostic, greutatile nu intra in produs.

Lantul, fiecare pas o functie:

1. `coral_satb` — patru portative separate (S, A, T, B) -> partitura stransa:
   S si A pe portativul de sus, T si B pe cel de jos, in cheia Fa.
2. `alteratii_afisate` — ce alteratii **se vad** pe portativul comun. Doua voci
   pe acelasi portativ isi impart alteratiile din masura: un Fa diez in sopran
   cere becar la Fa-ul altistului de dupa el, ceea ce pe patru portative nu se
   intampla. Imaginea si tinta iau aceeasi decizie din aceeasi functie.
3. `pagini` — impartirea in sisteme si pagini.
4. `musicxml_strans` — MusicXML pe doua portative, cu rupturile scrise, gata de
   randat in MuseScore.

Imaginea si tinta vin **din acelasi `Score`**: MuseScore deseneaza ce scrie
`musicxml_strans`, iar tinta o scrie `kern_writer` din aceeasi partitura. Nimic
nu se potriveste dupa aceea, deci nu are ce sa nu se potriveasca.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import replace
from fractions import Fraction
from typing import Any

from app.evaluation.reference import UnsupportedNotation, from_music21
from app.music_model.model import (
    Accidental,
    Clef,
    Duration,
    KeySignature,
    Measure,
    Note,
    NoteType,
    Part,
    Rest,
    Score,
    measures_slice,
    rests_for,
)
from app.musicxml.export import (
    DOCTYPE,
    _duration_ticks,
    _sub,
    _write_attributes,
    _write_time_modification,
    divisions_for,
    tuplet_marks,
)

# Asezarea partiturii stranse corale, de sus in jos.
STRANSA = [["S", "A"], ["T", "B"]]

# Cu un portativ de solo deasupra (`cu_solo`), ca pe cantarile cu solo si pian.
CU_SOLO = [["X1"], ["S", "A"], ["T", "B"]]

# Cu trei portative, sistemele sunt mai inalte; patru nu mai incap mereu pe o pagina.
SISTEME_PE_PAGINA_CU_SOLO = 3

# De unde vine linia de solo (vocea donorului) si in ce cheie se scrie. Pe „Doamne bun asculta"
# solo-ul e o data in cheia Sol (sus) si o data in cheia Fa (bas si tenor).
SOLO_DIN = (("S", "G"), ("S", "G"), ("T", "F"), ("B", "F"))

# Cat de des tace solo-ul o masura intreaga: pe paginile reale solo-ul intra si iese.
SOLO_TACE = 0.12

# Pauza de o masura intreaga, pe lungimea masurii (in patrimi).
_PAUZA_INTREAGA = {
    Fraction(4): Duration(NoteType.WHOLE),
    Fraction(3): Duration(NoteType.HALF, dots=1),
    Fraction(2): Duration(NoteType.HALF),
    Fraction(6): Duration(NoteType.WHOLE, dots=1),
}

# Cate masuri pe sistem. Paginile reale au intre 3 si 6 (pagina 3: 5, 6, 3);
# Bach e mai dens decat ele — optimi in toate vocile — deci plafonul e mai jos,
# altfel MuseScore rupe singur randul si decupajele nu se mai potrivesc.
MASURI_PE_SISTEM = (3, 4, 5)

# Cate „coloane" — momente distincte in care incepe ceva, in oricare voce —
# incap pe un sistem inainte ca MuseScore sa rupa singur randul. Masurat pe
# 2026-10-02 pe primele 458 de pagini: 31 s-au rupt, toate cu un sistem de
# **peste 32** de coloane; dintre cele care au incaput, unele aveau si 42, deci
# masura e grosiera (o saisprezecime ia mai putin loc decat o patrime). Plafonul
# e pus sub toate cele rupte: costa cateva sisteme mai scurte decat ar fi
# incaput, si nu mai pierde pagini intregi.
COLOANE_PE_SISTEM = 30

# Cel mult cate sisteme pe pagina. O pagina = un fisier = o referinta, cum cere
# `masoara.py`, deci pagina **nu are voie** sa curga pe a doua. Masurat pe
# 2026-10-02: din sase sisteme Bach, MuseScore asaza cinci pe o pagina Letter si
# il muta pe al saselea; cu note pe linii suplimentare intra si mai putine.
# Patru lasa loc, si e aproape de paginile reale (pagina 3 are trei).
SISTEME_PE_PAGINA = 4

_ALTERATIE_SCRISA = {1: Accidental.SHARP, -1: Accidental.FLAT, 0: Accidental.NATURAL}

# Cel mai larg interval, in trepte, pe care doua voci il impart pe o singura
# codita in stilul `acorduri` (vezi `se_scrie_acord`).
OCTAVA_PE_CODITA = 7


def coral_satb(m21_score: Any) -> Score:
    """Un coral din `music21` (patru portative) -> partitura stransa SATB.

    Refuza ce nu e coral in patru voci — cateva fisiere din corpus au si parti
    instrumentale — si ce nu se poate scrie exact: dublul diez n-are semn in
    modelul intern, iar o tinta fara el ar descrie alta nota decat imaginea.
    """
    parts = list(m21_score.parts)
    if len(parts) != 4:
        raise UnsupportedNotation(f"{len(parts)} portative, nu 4 — nu e coral SATB")
    for part in parts:
        if any(len(measure.voices) > 1 for measure in part.getElementsByClass("Measure")):
            raise UnsupportedNotation("un portativ are mai multe voci — divisi")

    score = from_music21(m21_score)
    if [part.id for part in score.parts] != ["S", "A", "T", "B"]:
        raise UnsupportedNotation(f"voci neasteptate: {[part.id for part in score.parts]}")

    for part in score.parts:
        cheie = Clef.treble() if part.id in ("S", "A") else Clef.bass()
        for measure in part.measures:
            for event in measure.events:
                if isinstance(event, Note) and abs(event.pitch.alter) > 1:
                    raise UnsupportedNotation(
                        f"{part.id} m{measure.number}: alteratie dubla ({event.pitch})"
                    )
            # Cheia o hotaraste portativul, nu vocea: tenorul, scris in Sol cu 8
            # dedesubt pe portativul lui, ajunge in Fa langa bas.
            if measure.clef is not None:
                measure.clef = cheie
        if part.measures and part.measures[0].clef is None:
            part.measures[0].clef = cheie

    score.staves = [list(group) for group in STRANSA]
    normalizeaza_alteratiile(score)
    return score


def alteratii_afisate(score: Score) -> dict[tuple[str, int, int], Accidental | None]:
    """Ce semn se deseneaza langa fiecare nota, pe portativul ei.

    Cheia e `(voce, masura, pozitia evenimentului)`. Regula e cea obisnuita de
    gravura: un semn apare cand nota nu are alteratia in vigoare pe treapta si
    octava ei — din armura sau de la o nota anterioara **de pe acelasi
    portativ**, din oricare voce. Nota legata dinspre stanga nu primeste semn,
    si nici nu schimba starea.
    """
    afisate: dict[tuple[str, int, int], Accidental | None] = {}
    for group in score.staves:
        voci = [score.part(part_id) for part_id in group]
        for measure in voci[0].measures:
            number = measure.number
            armura: KeySignature = voci[0].context_at(number)[1]
            din_armura = armura.altered_steps
            stare: dict[tuple[str, int], int] = {}
            note = []
            for voce in voci:
                pozitie = Fraction(0)
                for index, event in enumerate(voce.measure(number).events):
                    if isinstance(event, Note):
                        note.append((pozitie, voce.id, index, event))
                    pozitie += event.quarter_length
            # La acelasi moment, vocea de sus intai: e ordinea in care citeste
            # ochiul un acord, si cea in care MuseScore asaza semnele.
            ordine = {voce.id: rang for rang, voce in enumerate(voci)}
            note.sort(key=lambda item: (item[0], ordine[item[1]]))
            for _, part_id, index, nota in note:
                treapta = (nota.pitch.step, nota.pitch.octave)
                in_vigoare = stare.get(treapta, din_armura.get(nota.pitch.step, 0))
                semn = None
                if not nota.tie_stop and nota.pitch.alter != in_vigoare:
                    semn = _ALTERATIE_SCRISA[nota.pitch.alter]
                    stare[treapta] = nota.pitch.alter
                afisate[(part_id, number, index)] = semn
    return afisate


def normalizeaza_alteratiile(score: Score) -> None:
    """`Pitch.accidental` in sensul pe care il scrie `kern_writer`.

    Humdrum scrie **inaltimea**, nu desenul: Fa diez e `f#` si in Sol major, unde
    semnul vine din armura. Singurul semn care tine de desen e becarul, `n`,
    scris doar unde se vede. Deci: diez si bemol mereu, becar doar unde
    `alteratii_afisate` il deseneaza. E exact ce iese dintr-o referinta scrisa in
    MuseScore si citita de `music21`, deci tintele de aici vorbesc aceeasi limba
    cu cele din `dataset.py`.
    """
    afisate = alteratii_afisate(score)
    for part in score.parts:
        for measure in part.measures:
            for index, event in enumerate(measure.events):
                if not isinstance(event, Note):
                    continue
                alter = event.pitch.alter
                if alter:
                    semn = _ALTERATIE_SCRISA[alter]
                else:
                    semn = (
                        Accidental.NATURAL
                        if afisate.get((part.id, measure.number, index)) == Accidental.NATURAL
                        else None
                    )
                event.pitch = replace(event.pitch, accidental=semn)


def renumeroteaza(score: Score) -> Score:
    """Masurile de la 1, ca o pagina sa fie un document de sine statator.

    `measures_slice` pastreaza numerele originale, iar o pagina care incepe la
    masura 13 ar intra in MusicXML cu `number="13"`. La recitire `music21` ia
    rupturile dupa numarul scris, pe cand `from_music21` numara de la 1 — si
    sistemele ar cadea pe alte masuri decat cele desenate.
    """
    for part in score.parts:
        for index, measure in enumerate(part.measures, 1):
            measure.number = index
    return score


def coloane(score: Score, numar: int) -> int:
    """Cate momente distincte de atac are masura, peste toate vocile."""
    momente = set()
    for part in score.parts:
        pozitie = Fraction(0)
        for event in part.measure(numar).events:
            momente.add(pozitie)
            pozitie += event.quarter_length
    return len(momente)


def sisteme(
    numar_masuri: int, rng: random.Random, latime=None
) -> list[tuple[int, int]]:
    """Impartirea masurilor in sisteme, `(prima, ultima)`, cu lungimi variate.

    Variate dinadins: o lungime fixa ar da imagini de aceeasi latime si acelasi
    numar de masuri, iar modelul ar putea invata „patru masuri pe sistem" in loc
    sa citeasca barele. Un rest de o singura masura se lipeste de sistemul
    dinainte — un sistem de o masura e intins de MuseScore pe toata latimea si nu
    seamana cu nimic de pe paginile reale.

    `latime(numar)` da coloanele unei masuri; cand e data, un sistem se scurteaza
    pana incape in `COLOANE_PE_SISTEM`, iar restul de o masura se lipeste doar
    daca incape. Fara ea, impartirea e cea de dinainte de 2026-10-02.
    """
    def incape(prima: int, ultima: int) -> bool:
        if latime is None or prima == ultima:
            return True
        return sum(latime(n) for n in range(prima, ultima + 1)) <= COLOANE_PE_SISTEM

    limite: list[tuple[int, int]] = []
    prima = 1
    while prima <= numar_masuri:
        ultima = min(prima + rng.choice(MASURI_PE_SISTEM) - 1, numar_masuri)
        while not incape(prima, ultima):
            ultima -= 1
        limite.append((prima, ultima))
        prima = ultima + 1
    if (
        len(limite) > 1
        and limite[-1][1] == limite[-1][0]
        and incape(limite[-2][0], limite[-1][1])
    ):
        ultima = limite.pop()[1]
        limite[-1] = (limite[-1][0], ultima)
    return limite


def cu_solo(score: Score, donor: Score, rng: random.Random) -> Score:
    """Coralul cu un portativ de solo deasupra (`CU_SOLO`): o linie dintr-o **alta** piesa.

    De ce: pe cantarile cu solo si pian („Doamne bun asculta", 2026-10-04) jumatate din note stau
    pe sisteme de trei portative, iar datele tiparite n-aveau niciunul — modelul nu scria al
    treilea portativ separat. De ce dintr-o alta piesa: in kern toate vocile unui moment stau pe
    acelasi rand, iar un solo copiat din sopranul aceleiasi piese l-ar invata sa scrie sopranul
    de doua ori, nu sa citeasca portativul de sus.

    Donorul are aceeasi armura si aceeasi metrica (le alege apelantul). Linia lui — sopranul in
    cheia Sol, ori tenorul sau basul in cheia Fa, ca in `SOLO_DIN` — se pune masura cu masura,
    fiecare masura a gazdei luand urmatoarea masura a donorului de aceeasi lungime; unde nu e
    niciuna (o anacruza) sau, din cand in cand (`SOLO_TACE`), solo-ul tace o masura. Legaturile
    de durata se scot: masurile donorului nu mai urmeaza una dupa alta. Armura, metrica, barele
    si repetitiile se iau de la gazda, ca toate portativele sa le aiba la fel.
    """
    vocea, cheia = rng.choice(SOLO_DIN)
    sursa = donor.part(vocea).measures
    gazda = score.part("S").measures
    clef = Clef.treble() if cheia == "G" else Clef.bass()

    solo = Part(id="X1", name="Solo")
    urmatoarea = 0
    for pozitie, masura in enumerate(gazda):
        lungime = masura.quarter_length
        evenimente = None
        if not (rng.random() < SOLO_TACE and lungime in _PAUZA_INTREAGA):
            for pas in range(len(sursa)):
                candidata = sursa[(urmatoarea + pas) % len(sursa)]
                if candidata.quarter_length == lungime and not candidata.partial:
                    evenimente = [
                        replace(e, tie_start=False, tie_stop=False) if isinstance(e, Note) else e
                        for e in candidata.events
                    ]
                    urmatoarea = (urmatoarea + pas + 1) % len(sursa)
                    break
        if evenimente is None:
            evenimente = (
                [Rest(_PAUZA_INTREAGA[lungime])] if lungime in _PAUZA_INTREAGA else rests_for(lungime)
            )
        solo.measures.append(Measure(
            number=masura.number,
            events=evenimente,
            clef=clef if pozitie == 0 else None,
            key=masura.key,
            time=masura.time,
            barline=masura.barline,
            partial=masura.partial,
            repeat_start=masura.repeat_start,
        ))

    rezultat = Score(
        parts=[solo, *score.parts],
        title=score.title,
        composer=score.composer,
        staves=[list(grup) for grup in CU_SOLO],
    )
    # Becarele solo-ului se socotesc in contextul gazdei: masurile donorului au fost mutate, iar
    # tinta trebuie sa spuna ce se vede (`normalizeaza_alteratiile`). Corul nu se schimba: solo-ul
    # e pe portativul lui, iar semnele se impart doar pe acelasi portativ.
    normalizeaza_alteratiile(rezultat)
    return rezultat


def pagini(
    score: Score,
    rng: random.Random,
    dupa_densitate: bool = True,
    sisteme_pe_pagina: int = SISTEME_PE_PAGINA,
) -> list[tuple[Score, list[tuple[int, int]]]]:
    """Coralul taiat in pagini: fiecare cu partitura ei si sistemele ei.

    O pagina e un fisier separat, cu masurile numerotate de la 1 si sistemele
    socotite in numerotarea aceea — deci exact forma unei referinte scrise in
    MuseScore pentru o singura poza.

    `dupa_densitate=False` reface impartirea primului lot (`data/bach`, generat
    inainte de `COLOANE_PE_SISTEM`), ca setul acela sa ramana reproductibil.
    """
    latime = (lambda numar: coloane(score, numar)) if dupa_densitate else None
    toate = sisteme(score.measure_count, rng, latime)
    # Impartite egal, nu umplute pe rand: sapte sisteme dau 4 + 3, nu 4 + 4 + ...
    # sau 6 + 1 — o pagina cu un singur sistem nu seamana cu nimic real.
    numar_pagini = -(-len(toate) // sisteme_pe_pagina)
    marimi = [len(toate) // numar_pagini + (i < len(toate) % numar_pagini)
              for i in range(numar_pagini)]
    rezultat = []
    start = 0
    for marime in marimi:
        bucata = toate[start : start + marime]
        start += marime
        prima, ultima = bucata[0][0], bucata[-1][1]
        pagina = renumeroteaza(measures_slice(score, prima, ultima))
        decalaj = prima - 1
        rezultat.append(
            (pagina, [(inceput - decalaj, sfarsit - decalaj) for inceput, sfarsit in bucata])
        )
    return rezultat


def se_scrie_acord(sus: list[Note | Rest], jos: list[Note | Rest]) -> bool:
    """Pot cele doua voci ale portativului, in masura asta, sa fie scrise ca acorduri?

    Asa le scriu imnarele — si `pagina3`: sopranul si altistul pe aceeasi codita,
    cand merg in acelasi ritm. Se poate doar daca acordul se desface inapoi in
    aceleasi doua voci: aceleasi durate si legaturi, numai note (o pauza a unei
    singure voci nu are loc intr-un acord) si vocea de sus **strict** deasupra pe
    portativ — la unison sau la incrucisare, desfacerea de sus in jos (vezi
    `reference._events_of`) ar da alte voci decat cele scrise.
    """
    if not sus or len(sus) != len(jos):
        return False
    for a, b in zip(sus, jos):
        if not (isinstance(a, Note) and isinstance(b, Note)):
            return False
        if a.duration != b.duration or (a.tie_start, a.tie_stop) != (b.tie_start, b.tie_stop):
            return False
        if a.pitch.diatonic_index <= b.pitch.diatonic_index or a.pitch.midi <= b.pitch.midi:
            return False
        # Peste o octava, gravorii despart vocile: tenorul si basul la o decima pe
        # aceeasi codita nu seamana cu nicio pagina reala.
        if a.pitch.diatonic_index - b.pitch.diatonic_index > OCTAVA_PE_CODITA:
            return False
    return True


def masuri_cu_acorduri(score: Score) -> int:
    """Cate (portativ, masura) s-ar scrie ca acorduri — 0 = pagina arata la fel in ambele stiluri."""
    total = 0
    for group in score.staves:
        if len(group) != 2:
            continue
        sus, jos = (score.part(part_id) for part_id in group)
        for masura in sus.measures:
            if se_scrie_acord(masura.events, jos.measure(masura.number).events):
                total += 1
    return total


def _scrie_nota(
    parinte: ET.Element,
    event: Note | Rest,
    divisions: int,
    voce: int,
    semn: Accidental | None,
    grup: tuple[bool, bool],
    acord: bool = False,
    codita: bool = True,
) -> None:
    """O nota sau o pauza, cu elementele in ordinea ceruta de DTD.

    `acord` = a doua nota a unui acord (`<chord/>` vine primul); `codita=False`
    lasa directia coditei in seama MuseScore, ca la un acord de imnar.
    """
    nod = _sub(parinte, "note")
    if acord:
        _sub(nod, "chord")
    if isinstance(event, Rest):
        _sub(nod, "rest")
    else:
        pitch = _sub(nod, "pitch")
        _sub(pitch, "step", event.pitch.step)
        if event.pitch.alter:
            _sub(pitch, "alter", event.pitch.alter)
        _sub(pitch, "octave", event.pitch.octave)
    _sub(nod, "duration", _duration_ticks(event.quarter_length, divisions))
    if isinstance(event, Note):
        if event.tie_stop:
            ET.SubElement(nod, "tie", {"type": "stop"})
        if event.tie_start:
            ET.SubElement(nod, "tie", {"type": "start"})
    _sub(nod, "voice", voce)
    _sub(nod, "type", event.duration.note_type.value)
    for _ in range(event.duration.dots):
        _sub(nod, "dot")
    if semn is not None:
        _sub(nod, "accidental", semn.value)
    _write_time_modification(nod, event.duration)
    if isinstance(event, Note) and codita:
        # Vocea de sus cu codita in sus, cea de jos in jos: asta e partitura
        # stransa. Fara directie, MuseScore ar alege dupa pozitia pe portativ.
        _sub(nod, "stem", "up" if voce == 1 else "down")
    starts, stops = grup
    tie_start = isinstance(event, Note) and event.tie_start
    tie_stop = isinstance(event, Note) and event.tie_stop
    if tie_start or tie_stop or starts or stops:
        notations = _sub(nod, "notations")
        if tie_stop:
            ET.SubElement(notations, "tied", {"type": "stop"})
        if tie_start:
            ET.SubElement(notations, "tied", {"type": "start"})
        if starts:
            ET.SubElement(notations, "tuplet", {"type": "start"})
        if stops:
            ET.SubElement(notations, "tuplet", {"type": "stop"})


def repetitii(score: Score) -> tuple[set[int], set[int]]:
    """Masurile care deschid (`|:`) si cele care inchid (`:|`) o repetitie.

    Ale partiturii, nu ale unei voci: daca o singura voce din sursa poarta
    semnul, il primesc toate portativele — altfel MuseScore l-ar desena doar pe
    unul, iar tinta (scrisa dintr-o singura voce, `kern_writer._barline_before`)
    ar putea spune altceva decat imaginea.
    """
    deschid = {m.number for part in score.parts for m in part.measures if m.repeat_start}
    inchid = {
        m.number
        for part in score.parts
        for m in part.measures
        if m.barline is not None and m.barline.repeat
    }
    return deschid, inchid


def _bara_din_dreapta(nod: ET.Element, inchide: bool, finala: bool) -> None:
    """Bara de la capatul masurii, cand nu e cea obisnuita: finala sau `:|`.

    Repetitia de la capatul piesei are aceeasi bara subtire-groasa ca finalul,
    deci un singur element le spune pe amandoua.
    """
    if not (finala or inchide):
        return
    bara = ET.SubElement(nod, "barline", {"location": "right"})
    _sub(bara, "bar-style", "light-heavy")
    if inchide:
        ET.SubElement(bara, "repeat", {"direction": "backward"})


def musicxml_strans(
    score: Score,
    limite: list[tuple[int, int]],
    ultima_din_piesa: bool = True,
    acorduri: bool = False,
) -> str:
    """Partitura stransa in MusicXML: un `<part>` pe portativ, doua voci in el.

    `limite` sunt sistemele paginii; fiecare incepe cu `<print new-system>`,
    exact ce scrie MuseScore cand rupi randul cu Ctrl+Enter. `music21` il
    citeste inapoi ca `SystemLayout`, deci fisierul e si referinta paginii.

    Numele partilor nu se tiparesc: paginile reale n-au „Soprano/Alto" in
    margine, iar textul in plus ar fi o diferenta fata de ele fara niciun folos.

    `acorduri`: masurile in care cele doua voci ale unui portativ merg in acelasi
    ritm (`se_scrie_acord`) se scriu ca acorduri pe o singura codita, ca in
    imnare; restul raman doua voci cu codite opuse. Gasit pe 2026-10-03: antrenat
    numai pe voci separate, modelul citea tot mai prost `pagina3`, scrisa cu
    acorduri — 91 -> 69 de note — dar 120-125 din 156 pe aceeasi pagina rescrisa
    cu voci separate. Citit inapoi cu `min_voices=2`, acordul se desface in
    aceleasi voci, deci tinta e aceeasi in ambele stiluri.
    """
    if not score.staves:
        raise ValueError("partitura nu spune pe ce portative stau vocile")
    divisions = divisions_for(score)
    semne = alteratii_afisate(score)
    inceputuri = {inceput for inceput, _ in limite}
    deschid, inchid = repetitii(score)

    root = ET.Element("score-partwise", {"version": "4.0"})
    part_list = _sub(root, "part-list")
    for index, _ in enumerate(score.staves, 1):
        score_part = ET.SubElement(part_list, "score-part", {"id": f"P{index}"})
        ET.SubElement(score_part, "part-name", {"print-object": "no"})

    for index, group in enumerate(score.staves, 1):
        voci: list[Part] = [score.part(part_id) for part_id in group]
        nod_parte = ET.SubElement(root, "part", {"id": f"P{index}"})
        for pozitie, masura in enumerate(voci[0].measures):
            number = masura.number
            atribute = {"number": str(number)}
            if masura.partial:
                atribute["implicit"] = "yes"
            nod = ET.SubElement(nod_parte, "measure", atribute)
            if pozitie and number in inceputuri:
                ET.SubElement(nod, "print", {"new-system": "yes"})
            finala = ultima_din_piesa and pozitie == len(voci[0].measures) - 1
            if number in deschid:
                # In ordinea in care il scrie MuseScore: dupa `print`, inaintea
                # atributelor. Desenat dupa cheie si armura, ca in `pagina3`.
                bara = ET.SubElement(nod, "barline", {"location": "left"})
                _sub(bara, "bar-style", "heavy-light")
                ET.SubElement(bara, "repeat", {"direction": "forward"})
            _write_attributes(
                nod, divisions if pozitie == 0 else None, masura.key, masura.time, masura.clef
            )
            lungime = masura.quarter_length
            if acorduri and len(voci) == 2:
                sus = voci[0].measure(number).events
                jos = voci[1].measure(number).events
                if se_scrie_acord(sus, jos):
                    for i, (a, b, grup) in enumerate(zip(sus, jos, tuplet_marks(sus))):
                        _scrie_nota(
                            nod, a, divisions, 1, semne.get((voci[0].id, number, i)), grup,
                            codita=False,
                        )
                        _scrie_nota(
                            nod, b, divisions, 1, semne.get((voci[1].id, number, i)),
                            (False, False), acord=True, codita=False,
                        )
                    _bara_din_dreapta(nod, number in inchid, finala)
                    continue
            for numar_voce, voce in enumerate(voci, 1):
                if numar_voce > 1:
                    inapoi = _sub(nod, "backup")
                    _sub(inapoi, "duration", _duration_ticks(lungime, divisions))
                evenimente = voce.measure(number).events
                for i, (event, grup) in enumerate(zip(evenimente, tuplet_marks(evenimente))):
                    semn = semne.get((voce.id, number, i)) if isinstance(event, Note) else None
                    # Codita impusa doar unde sunt doua voci; pe un portativ cu o singura voce
                    # (solo-ul) o hotaraste MuseScore dupa inaltime, ca pe o pagina reala.
                    _scrie_nota(
                        nod, event, divisions, numar_voce, semn, grup, codita=len(voci) > 1
                    )
            _bara_din_dreapta(nod, number in inchid, finala)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{DOCTYPE}\n{body}\n'
