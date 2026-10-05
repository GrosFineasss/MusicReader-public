"""Corectarea unei transcrieri: adrese, modificari, si drumul spre antrenare.

Etapa 12 din roadmap. Ce lipsea pana acum nu era fereastra, ci **unde se duc
corectiile**: o corectie care doar repara un fisier e munca pierduta, fiindca
modelul greseste la fel pe pagina urmatoare.

Aici e partea care se poate strica tacut, deci partea care se testeaza. Widgetul
de deasupra (`app/ui/corectare.py`) doar arata ce e aici.

**Nu se scrie un al doilea format de perechi de antrenare.** `scripts/dataset.py`
face deja (imagine, MusicXML) -> perechi, cu decupare pe sisteme si poarta de
dus-intors. Deci corectarea produce **MusicXML corectat**, si atat; materialul de
antrenare iese pe drumul care exista. Doua drumuri care ar putea sa nu fie de
acord ar insemna ca nu stim care are dreptate.

**Fiecare modificare intoarce o partitura noua**, nu o modifica pe cea primita.
Asa istoricul de „inapoi" e o lista de partituri, nu un jurnal de operatii
inversabile — si nu exista starea in care jumatate dintr-o modificare s-a aplicat.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from app.music_model.model import Duration, Event, Note, Pitch, Rest, Score
from app.validation.rules import Finding, validate


@dataclass(frozen=True)
class Adresa:
    """Unde se afla un eveniment. Aceleasi campuri ca `Finding`, dinadins.

    `Finding` localizeaza deja o constatare prin (part_id, measure, event_index).
    Folosind exact aceleasi campuri, constatarile validatorului se leaga de
    evenimentele editabile fara nicio traducere — iar fereastra poate arata
    „problema e la nota asta" fara sa ghiceasca.

    **Fara `order=True`**, desi ar fi fost gratis: ar fi sortat alfabetic dupa
    `part_id`, adica A, B, S, T — nu ordinea in care stau vocile in partitura.
    O ordonare care arata plauzibil si e gresita e mai rea decat niciuna. Ordinea
    de citire o da `evenimente()`, care merge pe `score.parts`.
    """

    part_id: str
    measure: int
    index: int


def evenimente(score: Score) -> list[tuple[Adresa, Event]]:
    """Toate evenimentele, in ordinea in care se citesc: voce, masura, pozitie."""
    return [
        (Adresa(part.id, masura.number, i), event)
        for part in score.parts
        for masura in part.measures
        for i, event in enumerate(masura.events)
    ]


def eveniment_la(score: Score, adresa: Adresa) -> Event:
    """Evenimentul de la adresa data, sau `KeyError` cu adresa in mesaj."""
    masura = score.part(adresa.part_id).measure(adresa.measure)
    if not 0 <= adresa.index < len(masura.events):
        raise KeyError(f"{adresa.part_id}.m{adresa.measure} n-are evenimentul #{adresa.index}")
    return masura.events[adresa.index]


def _cu_masura(score: Score, adresa: Adresa, la_coada: bool = False):
    """Copie a partiturii, plus masura din ea de la adresa. Copia se modifica.

    `la_coada` doar pentru inserare, singura operatie careia pozitia de dupa
    ultimul eveniment ii e buna. Fara distinctia asta, o adresa cu unul peste
    capat ar trece de verificare si ar cadea mai incolo cu `IndexError`, adica
    fara sa spuna nici ce adresa, nici ce masura.
    """
    noua = copy.deepcopy(score)
    masura = noua.part(adresa.part_id).measure(adresa.measure)
    capat = len(masura.events) if la_coada else len(masura.events) - 1
    if not 0 <= adresa.index <= capat:
        raise KeyError(f"{adresa.part_id}.m{adresa.measure} n-are pozitia #{adresa.index}")
    return noua, masura


def schimba_inaltimea(score: Score, adresa: Adresa, pitch: Pitch) -> Score:
    """Alta inaltime pentru nota de la adresa. Pauzele n-au inaltime."""
    noua, masura = _cu_masura(score, adresa)
    event = masura.events[adresa.index]
    if not isinstance(event, Note):
        raise TypeError(f"{adresa} e o pauza, nu are inaltime")
    masura.events[adresa.index] = Note(
        pitch=pitch, duration=event.duration,
        tie_start=event.tie_start, tie_stop=event.tie_stop,
    )
    return noua


def schimba_durata(score: Score, adresa: Adresa, duration: Duration) -> Score:
    """Alta durata, si pentru note si pentru pauze."""
    noua, masura = _cu_masura(score, adresa)
    event = masura.events[adresa.index]
    if isinstance(event, Note):
        masura.events[adresa.index] = Note(
            pitch=event.pitch, duration=duration,
            tie_start=event.tie_start, tie_stop=event.tie_stop,
        )
    else:
        masura.events[adresa.index] = Rest(duration=duration)
    return noua


def catre_pauza(score: Score, adresa: Adresa) -> Score:
    """Nota devine pauza, pastrandu-si durata.

    Cea mai frecventa corectie de asteptat: modelul vede o nota unde e o pauza.
    Se face dintr-o apasare, nu prin stergere plus inserare.
    """
    noua, masura = _cu_masura(score, adresa)
    event = masura.events[adresa.index]
    masura.events[adresa.index] = Rest(duration=event.duration)
    return noua


def catre_nota(score: Score, adresa: Adresa, pitch: Pitch) -> Score:
    """Pauza devine nota, pastrandu-si durata."""
    noua, masura = _cu_masura(score, adresa)
    event = masura.events[adresa.index]
    masura.events[adresa.index] = Note(pitch=pitch, duration=event.duration)
    return noua


def sterge(score: Score, adresa: Adresa) -> Score:
    """Scoate evenimentul. Masura ramane scurta — validatorul o va semnala."""
    noua, masura = _cu_masura(score, adresa)
    del masura.events[adresa.index]
    return noua


def insereaza(score: Score, adresa: Adresa, event: Event) -> Score:
    """Baga un eveniment **inaintea** pozitiei date; la lungimea listei, la coada."""
    noua, masura = _cu_masura(score, adresa, la_coada=True)
    masura.events.insert(adresa.index, event)
    return noua


def probleme(score: Score) -> dict[Adresa | None, list[Finding]]:
    """Constatarile validatorului, grupate pe adresa.

    Cele care nu spun un eveniment anume (o masura intreaga prea scurta, o voce
    lipsa) cad sub cheia `None`: sunt adevarate, dar nu se pot arata pe o nota.
    """
    grupate: dict[Adresa | None, list[Finding]] = {}
    for finding in validate(score):
        cheie = (
            Adresa(finding.part_id, finding.measure, finding.event_index)
            if finding.part_id and finding.measure is not None
            and finding.event_index is not None
            else None
        )
        grupate.setdefault(cheie, []).append(finding)
    return grupate


class Istoric:
    """Sirul de versiuni ale partiturii, pentru „inapoi" si „inainte".

    Fiindca fiecare modificare intoarce o partitura noua, istoricul e o lista de
    partituri intregi si nu are cum sa ajunga in dezacord cu ce se vede. La zeci
    de corectii pe pagina, costul de memorie e neglijabil fata de o poza.
    """

    def __init__(self, score: Score) -> None:
        self._versiuni: list[Score] = [score]
        self._acum = 0

    @property
    def curenta(self) -> Score:
        return self._versiuni[self._acum]

    @property
    def se_poate_inapoi(self) -> bool:
        return self._acum > 0

    @property
    def se_poate_inainte(self) -> bool:
        return self._acum < len(self._versiuni) - 1

    @property
    def modificari(self) -> int:
        """Cate schimbari s-au facut fata de transcrierea primita."""
        return self._acum

    def aplica(self, noua: Score) -> Score:
        """Versiune noua. Ce era „inainte" se pierde — ca in orice editor."""
        del self._versiuni[self._acum + 1:]
        self._versiuni.append(noua)
        self._acum += 1
        return self.curenta

    def inapoi(self) -> Score:
        if self.se_poate_inapoi:
            self._acum -= 1
        return self.curenta

    def inainte(self) -> Score:
        if self.se_poate_inainte:
            self._acum += 1
        return self.curenta
