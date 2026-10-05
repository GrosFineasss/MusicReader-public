"""Cat din corpusul de antrenare vine din piesele pe care se masoara.

PROJECT.md §15.41: din 299 de perechi, 69 erau pagina 3 — chiar pagina pe care
se ia fiecare cifra a proiectului. Nimeni n-a hotarat asta. `sintetic.py`
randeaza tot ce i se da, iar optiunile care ar fi cerut separarea
(`--scriitor-validare`, `--procent-validare`) despart **maini**, nu **piese**.
Cu o singura mana n-aveau ce desparti.

De aceea scurgerea se **numara**, nu se tine minte. Modulul nu stie nimic despre
torch si nu deschide nicio imagine: primeste un corpus si o lista de piese
masurate si spune ce procent din perechi deriva din ele.

Numele piesei se citeste din numele fisierului, fiindca acolo l-a scris
generatorul: `pagina3-t+2-s01-fineas-0.png` -> piesa `pagina3`, transpusa cu
doua semitonuri, sistemul 1, scrisul lui fineas, varianta 0. Transpozitia
**nu** face o piesa noua: aceeasi muzica urcata cu un ton e tot muzica pe care
se masoara (§15.23).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# `<piesa>[-t±N]-s<NN>-<scriitor>-<varianta>.png`. Se taie de la dreapta, fiindca
# numele piesei poate contine si el cratime — `doamne-bun-asculta`.
SUFIX = re.compile(r"-s\d+-.+$")
TRANSPOZITIE = re.compile(r"-t[+-]\d+$")


def piesa_din_imagine(nume: str) -> str:
    """Piesa din care s-a randat imaginea, fara transpozitie si fara sistem."""
    baza = SUFIX.sub("", Path(nume).stem)
    return TRANSPOZITIE.sub("", baza)


def piesa_din_referinta(cale: str | Path) -> str:
    """Piesa masurata, din numele referintei ei."""
    return TRANSPOZITIE.sub("", Path(cale).stem)


@dataclass(frozen=True)
class Scurgere:
    """Cate perechi din corpus vin din piesele masurate."""

    total: int
    pe_piesa: dict[str, int]
    pe_piesa_transpuse: dict[str, int]

    @property
    def scurse(self) -> int:
        return sum(self.pe_piesa.values()) + sum(self.pe_piesa_transpuse.values())

    @property
    def procent(self) -> float:
        return 100.0 * self.scurse / self.total if self.total else 0.0

    @property
    def curat(self) -> bool:
        return self.scurse == 0


def citeste_corpus(cale: Path) -> list[str]:
    """Numele imaginilor din corpus, in ordinea din fisier."""
    imagini = []
    for rand in cale.read_text(encoding="utf-8").splitlines():
        rand = rand.strip()
        if rand:
            imagini.append(json.loads(rand)["image"])
    return imagini


def citeste_set(cale: Path) -> list[str]:
    """Piesele pe care se masoara, din setul de validare."""
    date = json.loads(cale.read_text(encoding="utf-8"))
    if isinstance(date, dict):
        date = [date]
    return [piesa_din_referinta(intrare["referinta"]) for intrare in date]


def masoara(imagini: list[str], masurate: list[str]) -> Scurgere:
    """Numara perechile care deriva din piesele masurate, direct sau transpuse."""
    cautate = set(masurate)
    directe: dict[str, int] = {}
    transpuse: dict[str, int] = {}
    for nume in imagini:
        piesa = piesa_din_imagine(nume)
        if piesa not in cautate:
            continue
        e_transpusa = bool(TRANSPOZITIE.search(SUFIX.sub("", Path(nume).stem)))
        tinta = transpuse if e_transpusa else directe
        tinta[piesa] = tinta.get(piesa, 0) + 1
    return Scurgere(total=len(imagini), pe_piesa=directe, pe_piesa_transpuse=transpuse)


def raport(s: Scurgere, masurate: list[str]) -> str:
    """Raportul, scris ca sa se poata citi fara sa se stie ce e o scurgere."""
    if not masurate:
        return "niciun set de masurare dat; nu se poate spune nimic despre scurgere"
    if s.curat:
        return (
            f"curat: niciuna din cele {s.total} perechi nu vine din "
            f"{', '.join(sorted(masurate))}"
        )
    linii = [
        f"SCURGERE: {s.scurse} din {s.total} de perechi — "
        f"{s.procent:.0f}% — vin din piesa pe care se masoara"
    ]
    for piesa in sorted(set(s.pe_piesa) | set(s.pe_piesa_transpuse)):
        d = s.pe_piesa.get(piesa, 0)
        tr = s.pe_piesa_transpuse.get(piesa, 0)
        linii.append(f"  {piesa}: {d} netranspuse + {tr} transpuse = {d + tr}")
    linii.append(
        "Modelul se antreneaza pe tinta pe care va fi masurat, desenata altfel. "
        "Cifra care iese e optimista cu o cantitate care nu se poate sti."
    )
    return "\n".join(linii)
