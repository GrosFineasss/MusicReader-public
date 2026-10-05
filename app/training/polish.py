"""Adaptor pentru `PRAIG/polish-scores`: parquet HuggingFace -> poza + referinta.

De ce exista. Pana acum proiectul n-a avut **nicio** masuratoare pe muzica
tiparita cu referinta sigura (§15.51): singurele cifre erau pe scris de mana,
iar iesirea tiparita din §12.4 n-avea cu ce fi comparata. `polish-scores` sunt
117 pagini **scanate** de partituri tiparite, cu ground truth `**ekern` — adica
exact scanuri, cu hartie ingalbenita si pete, nu randari curate. Sunt materialul
pentru intrebarea „merita directia PDF?", si sunt libere (celelalte seturi scrise
de mana ale PRAIG cer aprobare manuala).

Ce nu sunt: scris de mana, si nici muzica corala — sunt pian, portativ dublu.
Cifra care iese de aici e o **margine de sus** pentru tiparit, nu un raspuns
despre manuscrise.

Conversia. `**ekern` e kern cu doua straturi de separatori peste el:

    32@qq@gg·(·/·L   ->   32qqgg(/L

`@` desparte campurile semantice ale unui token, `·` (U+00B7) caracterele din
fiecare camp; codul lor de referinta le pune ca sa dea modelului tokeni mai mici.
Scoase amandoua, ramane Humdrum obisnuit. Mai trebuie doar exponentul redenumit
in `**kern`, altfel `music21` nu recunoaste coloana si intoarce o partitura goala
**fara sa se planga** — esecul arata ca o pagina fara note.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEPARATOR_CAMP = "@"
SEPARATOR_CARACTER = "\u00b7"


class PolishError(ValueError):
    """Un rand din parquet care nu se poate converti."""


def ekern_la_kern(text: str) -> str:
    """Document `**ekern` -> document `**kern` pe care il citeste `music21`.

    Idempotenta: un document deja in `**kern` trece neatins.
    """
    kern = text.replace(SEPARATOR_CAMP, "").replace(SEPARATOR_CARACTER, "")
    kern = kern.replace("**ekern", "**kern").replace("**bekern", "**kern")
    linii = kern.rstrip("\n").split("\n")
    if not linii or not linii[0].startswith("**"):
        raise PolishError("documentul nu incepe cu un exponent `**`")
    if not linii[-1].startswith("*-"):
        linii.append("\t".join(["*-"] * len(linii[0].split("\t"))))
    return "\n".join(linii) + "\n"


@dataclass(frozen=True)
class Pagina:
    """O pagina extrasa: numele, pozele si referinta ei."""

    nume: str
    imagine: bytes
    kern: str


def pagina_din_rand(rand: dict) -> Pagina:
    """Un rand de parquet (`image`, `transcription`) -> `Pagina`."""
    imagine = rand.get("image") or {}
    octeti = imagine.get("bytes")
    if not octeti:
        raise PolishError("randul n-are octeti de imagine")
    nume = Path(imagine.get("path") or "pagina").stem
    return Pagina(nume=nume, imagine=octeti, kern=ekern_la_kern(rand["transcription"]))


def musicxml_din_kern(kern: str) -> bytes:
    """Kern -> MusicXML, prin `music21`, ca sa intre in `scripts/evaluate.py`.

    Trece prin MusicXML si nu prin parserul nostru (`app/recognition/kern.py`)
    dinadins: referinta ajunge sa fie citita de exact acelasi cod care citeste
    referintele scrise in MuseScore, deci cifra e comparabila cu toate celelalte
    din proiect.
    """
    import music21

    partitura = music21.converter.parseData(kern, format="humdrum")
    if not partitura.parts:
        raise PolishError("`music21` n-a gasit nicio parte — exponent gresit?")
    return music21.musicxml.m21ToXml.GeneralObjectExporter().parse(partitura)
