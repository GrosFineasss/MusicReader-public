"""Asamblarea unei pagini: portative decupate -> o partitura coerenta.

Detectorul de portative da o lista plata, de sus in jos. O pagina de cor in
partitura stransa e insa structurata: perechi de portative (cheia Sol + cheia
Fa) formeaza un **sistem**, iar sistemele se citesc unul dupa altul, continuand
aceleasi patru voci.

Pasul asta e determinist si separat de model. Daca modelul citeste gresit un
portativ, vrem ca greseala sa iasa la iveala ca o voce desincronizata, pe care
validatorul o semnaleaza — nu topita in restul partiturii.
"""

from __future__ import annotations

from collections.abc import Sequence

from dataclasses import dataclass

from app.music_model.model import Measure, Part, Score, rests_for
from app.recognition.kern import parse_closed_score, parse_document, parse_open_system

STAVES_PER_SYSTEM = 2  # partitura stransa: Sol (S+A) si Fa (T+B)

_FALLBACK_NOTE = (
    "partitura a fost construita cu o parte per portativ, nu ca sistem coral — verific-o inainte de a te baza pe ea"
)


@dataclass
class PageResult:
    score: Score
    systems: int
    problems: list[str]


def group_into_systems(staff_kerns: list[str]) -> list[list[str]]:
    """Imparte portativele in sisteme, in ordinea de pe pagina."""
    if len(staff_kerns) % STAVES_PER_SYSTEM:
        raise ValueError(
            f"{len(staff_kerns)} portative nu se impart in sisteme de "
            f"{STAVES_PER_SYSTEM} — probabil detectia a ratat sau a inventat unul"
        )
    return [
        staff_kerns[i : i + STAVES_PER_SYSTEM]
        for i in range(0, len(staff_kerns), STAVES_PER_SYSTEM)
    ]


def _joinable(target: Score, addition: Score) -> bool:
    """Se continua ultima masura a paginii in prima masura a sistemului nou?

    Pe manuscris, o masura incepe des la capatul unui portativ si se termina la
    inceputul urmatorului. Modelul citeste **un sistem odata**, deci masura
    aceea ii iese in doua bucati, cate una in fiecare document. Lipite inapoi,
    dau masura scrisa; lasate asa, dau doua masuri scurte si o masura in plus
    fata de partitura adevarata — la fiecare hotar de sistem.

    Hotarul nu se ghiceste, se **socoteste**, si numai cand raspunsul e
    neambiguu:

    - toate vocile trebuie sa fie de acord, in aceeasi masura: fiecare pereche
      (ultima, prima) insumeaza **exact** o masura intreaga, si fiecare bucata e
      strict mai scurta decat una. O singura voce care nu inchide opreste
      lipirea pentru toate — desincronizarea vocilor e greseala pe care §1.1 o
      numeste cea mai costisitoare;
    - vocile trebuie sa fie aceleasi de-o parte si de alta. Un sistem in care
      intra o voce noua nu continua o masura, ci incepe una;
    - **niciodata peste o bara de repetitie.** Masura de dinaintea unei
      repetitii e scurta dinadins (anacruza o completeaza, §14), si tocmai de
      aceea aritmetica ar da acolo un raspuns fals: 3 + 1 face tot 4. Bara
      *finala* nu opreste nimic, desi ar parea ca ar trebui: fiecare document
      Humdrum se incheie cu `==`, fiindca e sfarsitul **decupajului**, nu al
      piesei. Folosita ca semnal, ar bloca orice lipire;
    - prima masura a sistemului nou nu are voie sa declare o cheie, o armura sau
      o metrica **diferita** de cea in vigoare. Fiecare sistem si le rescrie pe
      ale lui la inceput, si o repetare identica nu inseamna nimic; una
      *schimbata* la mijloc de masura e insa in afara scopului v1, si e mai bine
      vazuta decat inghitita.
    """
    ids = {part.id for part in target.parts}
    if not ids or ids != {part.id for part in addition.parts}:
        return False

    for part in target.parts:
        source = addition.part(part.id)
        if not part.measures or not source.measures:
            return False
        last, first = part.measures[-1], source.measures[0]
        if not last.events or not first.events:
            return False
        if last.barline is not None and last.barline.repeat:
            return False
        # Semnul `|:` de la inceputul sistemului nou e aceeasi bara, vazuta din
        # partea cealalta: MuseScore si imnarele il deseneaza dupa cheie si armura.
        if first.repeat_start:
            return False
        clef, key, time = part.context_at(last.number)
        if (
            (first.clef is not None and first.clef != clef)
            or (first.key is not None and first.key != key)
            or (first.time is not None and first.time != time)
        ):
            return False
        full = time.measure_quarter_length
        if last.quarter_length >= full or first.quarter_length >= full:
            return False
        if last.quarter_length + first.quarter_length != full:
            return False
    return True


def _append_system(target: Score, addition: Score) -> list[str]:
    """Leaga masurile unui sistem in continuarea celor de dinainte.

    Numerele de masura se refac de la zero pentru fiecare sistem, fiindca
    fiecare portativ a fost transcris separat si nu are de unde sti al catelea
    e pe pagina.

    Vocile nu sunt aceleasi in toate sistemele: pe „Iata-L pe Hristos", primul
    sistem are doar S&A, iar T&B intra abia in al doilea. O voce care apare mai
    tarziu primeste intai masuri de pauza pentru tot ce a pierdut, ca sa inceapa
    sa cante la locul potrivit; una care lipseste dintr-un sistem e completata
    de `pad_parts_to_equal_length`. Decalajul se ia o singura data, pentru toate
    vocile, tocmai ca ele sa ramana aliniate intre ele.

    Cand ultima masura de pana acum se continua in prima a sistemului nou —
    conditiile sunt in `_joinable` — cele doua bucati se lipesc in loc sa
    devina doua masuri. Intoarce ce s-a facut, ca sa ajunga in raport: o
    lipire gresita trebuie sa fie citibila, nu invizibila.
    """
    pad_parts_to_equal_length(target)

    notes: list[str] = []
    joined = _joinable(target, addition)
    if joined:
        for part in target.parts:
            source = addition.part(part.id)
            last = part.measures[-1]
            last.events = last.events + source.measures[0].events
            last.barline = source.measures[0].barline
        notes.append(
            f"masura {target.measure_count} era taiata de hotarul dintre sisteme "
            "si a fost lipita la loc"
        )

    # Masurile sistemului nou isi pastreaza numerele lor (1..k) si se decaleaza
    # cu cate masuri sunt deja pe pagina. Cand prima masura a fost lipita, ea nu
    # mai vine incoace, deci decalajul scade cu unu ca numerotarea sa ramana
    # neintrerupta.
    offset = target.measure_count - (1 if joined else 0)

    known = {part.id for part in target.parts}
    for part in addition.parts:
        if part.id not in known:
            target.parts.append(Part(id=part.id, name=part.name))
    if len(target.parts) > len(known):
        pad_parts_to_equal_length(target)
        for part in target.parts:
            if part.id in known or not part.measures:
                continue
            # Cheia si armura vocii noi sunt cunoscute abia din sistemul in
            # care intra. Fara ele, pauzele de dinainte s-ar scrie in cheia
            # Sol implicita, iar basul ar aparea cu doua chei diferite.
            venite = addition.part(part.id).measures
            if not venite:
                # Modelul poate emite o voce **fara nicio masura** — un spine
                # deschis si inchis cu nimic intre. N-avem de unde lua cheia, iar
                # pana la §15.26 asta oprea asamblarea intregii pagini cu
                # `IndexError`: o singura voce goala si nu mai iesea nimic din
                # poza. `_joinable` verifica deja cazul; aici lipsea.
                continue
            first = venite[0]
            part.measures[0].clef = first.clef
            part.measures[0].key = first.key
            part.measures[0].time = first.time

    for part in target.parts:
        try:
            source = addition.part(part.id)
        except KeyError:
            continue
        for measure in source.measures[1:] if joined else source.measures:
            renumbered = Measure(
                number=offset + measure.number,
                events=measure.events,
                clef=measure.clef,
                key=measure.key,
                time=measure.time,
                barline=measure.barline,
                partial=measure.partial,
                repeat_start=measure.repeat_start,
            )
            part.measures.append(renumbered)

    return notes


def pad_parts_to_equal_length(score: Score) -> int:
    """Aduce toate partile la acelasi numar de masuri, umplandu-le cu pauze.

    MusicXML nu are un model in care o parte are mai putine masuri decat alta:
    o partitura are N masuri, iar fiecare parte trebuie sa le contina pe toate.
    MuseScore **refuza sa deschida** un fisier care incalca asta — verificat, cu
    cele sapte portative ale unei pagini reale, care se deschideau separat dar
    nu si impreuna.

    Intoarce cate masuri au fost adaugate.
    """
    target = score.measure_count
    added = 0
    for part in score.parts:
        while len(part.measures) < target:
            number = len(part.measures) + 1
            _, _, time = part.context_at(number)
            part.measures.append(
                Measure(number=number, events=rests_for(time.measure_quarter_length))
            )
            added += 1
    return added


def fill_empty_measures(score: Score) -> int:
    """Umple cu pauze masurile care exista dar n-au niciun eveniment.

    O masura fara evenimente nu supravietuieste unui dus-intors prin MusicXML.
    Scrisa, iese `<measure>` fara nicio nota; citita inapoi, umplutura din
    `reference._events_of` o transforma intr-o masura de tacere. Deci acelasi
    rezultat masurat pe obiect si pe fisier da doua cifre diferite.

    Masurat pe transcrierea din §15.38 (pasul 25 + `--structura`): 38 de
    evenimente in memorie, 42 dupa dus-intors — masura 2, goala in toate cele
    patru voci. Exact nepotrivirea lasata deschisa in §15.39.

    Fisierul e cel care are dreptate: o masura fara nimic in ea **e** o masura
    de tacere, si asa o arata orice cititor, MuseScore inclusiv. Ce se repara
    aici e obiectul, ca sa spuna acelasi lucru cu fisierul in care se scrie.

    `pad_parts_to_equal_length` face treaba vecina — masuri care **lipsesc** de
    la coada unei parti — si nu le acopera pe astea, fiindca masura goala exista
    si e numarata.

    Intoarce cate masuri au fost umplute.
    """
    umplute = 0
    for part in score.parts:
        for measure in part.measures:
            if measure.events:
                continue
            _, _, time = part.context_at(measure.number)
            measure.events = rests_for(time.measure_quarter_length)
            umplute += 1
    return umplute


def one_part_per_staff(staff_kerns: list[str], title: str = "") -> Score:
    """Varianta de rezerva: fiecare portativ devine o parte separata.

    Nu incearca sa recunoasca sistemele si nu presupune nimic despre cate voci
    sunt pe portativ. Rezultatul nu e o partitura corala corecta, dar e un
    MusicXML valid, care **se deschide in MuseScore** — iar o transcriere pe
    care o poti privi si corecta bate un esec din care nu iese nimic.
    """
    score = Score(title=title)
    for index, text in enumerate(staff_kerns, 1):
        voices = parse_document(text)
        for voice_index, result in enumerate(voices, 1):
            part = result.part
            part.id = f"P{index}" if len(voices) == 1 else f"P{index}v{voice_index}"
            part.name = f"Portativ {index}" + ("" if len(voices) == 1 else f", vocea {voice_index}")
            score.parts.append(part)

    pad_parts_to_equal_length(score)
    fill_empty_measures(score)
    return score


def assemble_page(staff_kerns: list[str], title: str = "") -> PageResult:
    """Construieste partitura unei pagini din transcrierile portativelor.

    Daca portativele nu se pot organiza in sisteme corale, **nu esuam**: cadem
    pe o parte per portativ. Utilizatorul primeste ceva de deschis in MuseScore,
    plus explicatia a ce nu a mers.
    """
    problems: list[str] = []

    try:
        systems = group_into_systems(staff_kerns)
    except ValueError as error:
        problems.append(str(error))
        return PageResult(
            score=one_part_per_staff(staff_kerns, title),
            systems=0,
            problems=problems + [_FALLBACK_NOTE],
        )

    score: Score | None = None
    used = 0
    for index, staves in enumerate(systems, 1):
        try:
            system_score = parse_closed_score(staves)
        except ValueError as error:
            problems.append(f"sistemul {index}: {error}")
            continue

        if score is None:
            score = system_score
            score.title = title
        else:
            problems.extend(_append_system(score, system_score))
        used += 1

    if score is None:
        return PageResult(
            score=one_part_per_staff(staff_kerns, title),
            systems=0,
            problems=problems + [_FALLBACK_NOTE],
        )

    # Un sistem sarit lasa o gaura in mijlocul piesei. Merita spus explicit:
    # partitura va parea completa, dar ii vor lipsi masuri.
    if used != len(systems):
        problems.append(f"doar {used} din {len(systems)} sisteme au intrat in partitura")

    # Si aici vocile pot ramane inegale, daca modelul a citit gresit un
    # portativ. Fara egalizare, fisierul nu se deschide in MuseScore.
    padded = pad_parts_to_equal_length(score)
    if padded:
        problems.append(
            f"{padded} masuri au fost completate cu pauze ca vocile sa aiba "
            "aceeasi lungime — undeva lipseste muzica"
        )

    # Masurile goale se umplu **inainte** sa se masoare ceva pe partitura asta:
    # altfel obiectul si fisierul in care se scrie spun lucruri diferite (§15.39).
    golite = fill_empty_measures(score)
    if golite:
        problems.append(
            f"{golite} masuri n-aveau niciun eveniment si s-au umplut cu pauze — "
            "modelul n-a scris nimic acolo"
        )

    return PageResult(score=score, systems=used, problems=problems)


def part_measure_counts(score: Score) -> dict[str, int]:
    """Cate masuri are fiecare voce — prima verificare cand ceva pare gresit."""
    return {part.id: len(part.measures) for part in score.parts}



def assemble_systems(
    system_kerns: list[str],
    title: str = "",
    staves: list[int] | None = None,
    layout: Sequence[Sequence[Sequence[str]]] | None = None,
) -> PageResult:
    """Ca `assemble_page`, dar fiecare text e un **sistem intreg**.

    `smt-grandstaff` citeste doua portative deodata si scoate toate vocile
    sistemului intr-un singur document Humdrum. Nu mai avem deci de grupat
    portative in sisteme — gruparea s-a facut pe imagine, inainte de model — ci
    doar de legat sistemele intre ele.

    Politica de esec e aceeasi ca la `assemble_page`: daca niciun sistem nu se
    asambleaza, cadem pe o parte per document si spunem de ce. Un MusicXML pe
    care il poti privi si corecta bate un esec din care nu iese nimic.
    """
    problems: list[str] = []
    score: Score | None = None
    used = 0

    # O singura asezare inseamna „la fel pe toate sistemele"; mai multe se dau
    # cate una pe sistem, fiindca asezarea variaza in interiorul paginii — vocea
    # de solo poate intra abia la al patrulea sistem.
    if layout is not None and 1 < len(layout) != len(system_kerns):
        problems.append(
            f"s-au declarat {len(layout)} asezari pentru {len(system_kerns)} "
            "sisteme — se foloseste prima pentru toate"
        )
        layout = [layout[0]]

    for index, text in enumerate(system_kerns, 1):
        per_system = None
        if layout is not None:
            per_system = layout[0] if len(layout) == 1 else layout[index - 1]
        try:
            parsed = parse_open_system(
                text, staves[index - 1] if staves else None, per_system
            )
        except ValueError as error:
            problems.append(f"sistemul {index}: {error}")
            continue
        if not parsed.score.parts:
            problems.append(f"sistemul {index}: nicio voce citita")
            continue

        problems.extend(f"sistemul {index}: {problem}" for problem in parsed.problems)
        if score is None:
            score = parsed.score
            score.title = title
        else:
            problems.extend(_append_system(score, parsed.score))
        used += 1

    if score is None:
        return PageResult(
            score=one_part_per_staff(system_kerns, title),
            systems=0,
            problems=problems + [_FALLBACK_NOTE],
        )

    if used != len(system_kerns):
        problems.append(f"doar {used} din {len(system_kerns)} sisteme au intrat in partitura")

    padded = pad_parts_to_equal_length(score)
    if padded:
        problems.append(
            f"{padded} masuri au fost completate cu pauze ca vocile sa aiba "
            "aceeasi lungime — undeva lipseste muzica"
        )

    # Masurile goale se umplu **inainte** sa se masoare ceva pe partitura asta:
    # altfel obiectul si fisierul in care se scrie spun lucruri diferite (§15.39).
    golite = fill_empty_measures(score)
    if golite:
        problems.append(
            f"{golite} masuri n-aveau niciun eveniment si s-au umplut cu pauze — "
            "modelul n-a scris nimic acolo"
        )

    # Asezarea pe portative a **primului** sistem nu descrie pagina intreaga:
    # sistemele pot avea asezari diferite, iar vocile in plus scoase de model
    # se adauga pe drum. Cand nu mai corespunde, se sterge — campul inseamna
    # „nu se stie" cand e gol, si o asezare gresita ar fi mai rea decat niciuna.
    if {part.id for part in score.parts} != {
        voice for group in score.staves for voice in group
    }:
        score.staves = []

    return PageResult(score=score, systems=used, problems=problems)
