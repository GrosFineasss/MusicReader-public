"""Descarca Bravura, fontul de referinta SMuFL, si tabelul lui de nume.

    .venv/Scripts/python.exe scripts/ia_bravura.py

De ce e nevoie de el: desenele-exemplu de pe foaia de glife trebuie sa arate
exact ca simbolul cerut, fiindca §15.22 a aratat ca oamenii copiaza ce vad, nu
ce scrie. Pana acum erau facute cu primitive PIL „fiindca pe masina asta nu
exista niciun font muzical".

De ce **Bravura** si nu altul: e fontul de referinta al specificatiei SMuFL —
aceleasi nume de glife pe care le foloseste si Smashcima — si e sub SIL Open
Font License 1.1, deci poate intra intr-un produs comercial. Spre deosebire de
MUSCIMA++ si JAZZMUS (§3.6), aici nu e nimic de ocolit.

Fisierele stau in `vendor/`, care nu e in git: se descarca, nu se versioneaza.
Tot ce le foloseste are revenire la primitivele vechi, deci lipsa lor nu rupe
nimic — schimba doar cat de bine arata desenele.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATIE = ROOT / "vendor" / "bravura"

BAZA_FONT = "https://raw.githubusercontent.com/steinbergmedia/bravura/master"
BAZA_SPEC = "https://raw.githubusercontent.com/w3c/smufl/gh-pages"

FISIERE = [
    (f"{BAZA_FONT}/redist/otf/Bravura.otf", "Bravura.otf"),
    (f"{BAZA_FONT}/LICENSE.txt", "LICENSE.txt"),
    (f"{BAZA_SPEC}/metadata/glyphnames.json", "glyphnames.json"),
]

# Sub atat, ce s-a descarcat e o pagina de eroare, nu fisierul cerut. GitHub
# raspunde cu 200 si un „404: Not Found" de 14 octeti pe caile gresite, deci
# codul HTTP nu e de ajuns ca verificare.
MINIM = 1024


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    DESTINATIE.mkdir(parents=True, exist_ok=True)
    for url, nume in FISIERE:
        cale = DESTINATIE / nume
        try:
            with urllib.request.urlopen(url, timeout=60) as raspuns:
                continut = raspuns.read()
        except OSError as eroare:
            print(f"{nume}: nu s-a putut descarca — {eroare}")
            return 1
        if len(continut) < MINIM:
            print(f"{nume}: raspuns de {len(continut)} octeti, prea mic ca sa fie fisierul")
            return 1
        cale.write_bytes(continut)
        print(f"{nume}: {len(continut)} octeti")

    print(f"\nscris in {DESTINATIE}")
    print("licenta: SIL Open Font License 1.1 — vezi LICENSE.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
