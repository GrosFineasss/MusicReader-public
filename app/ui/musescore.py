"""Deschiderea partiturilor in MuseScore.

PROJECT.md §1: nu construim un editor de partituri. MuseScore e editorul final,
iar aplicatia noastra doar il cheama. Modulul asta e tot ce trebuie pentru asta.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Locurile obisnuite pe Windows, in ordinea in care merita cautate.
CANDIDATES = (
    Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
    Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.com"),
    Path(r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe"),
    Path(r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"),
)


def find() -> Path | None:
    """Calea catre MuseScore, sau `None` daca nu e instalat."""
    override = os.environ.get("MUSESCORE")
    if override and Path(override).exists():
        return Path(override)

    for candidate in CANDIDATES:
        if candidate.exists():
            return candidate

    found = shutil.which("MuseScore4") or shutil.which("musescore")
    return Path(found) if found else None


def is_configured() -> bool:
    """MuseScore a fost pornit macar o data?

    Pana la prima pornire, expertul initial blocheaza si interfata grafica si
    linia de comanda. Merita spus utilizatorului inainte sa para ca aplicatia
    noastra s-a blocat.
    """
    if sys.platform != "win32":
        return True
    return (Path(os.environ.get("APPDATA", "")) / "MuseScore").exists()


def open_score(path: Path) -> None:
    """Deschide un fisier in MuseScore, fara sa astepte inchiderea lui."""
    executable = find()
    if executable is None:
        raise FileNotFoundError(
            "MuseScore nu a fost gasit. Instaleaza-l, sau pune calea in "
            "variabila de mediu MUSESCORE."
        )
    subprocess.Popen([str(executable), str(path)], close_fds=True)
