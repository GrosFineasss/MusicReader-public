"""Porneste aplicatia.

    OMR-Cor.bat                          (dublu-clic)
    .venv\\Scripts\\pythonw.exe scripts\\app.py

Lansata cu `pythonw.exe`, aplicatia nu are consola — ceea ce e corect pentru o
fereastra, dar inseamna ca o eroare de pornire nu s-ar vedea nicaieri. De asta
tot ce se intampla inainte de bucla de evenimente e prins si aratat intr-o
caseta de mesaj, plus scris intr-un fisier de log.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Pozitia 0 nu e un detaliu: rulat ca script, Python pune `scripts/` in capul
# lui `sys.path`, iar acolo exista `app.py` — care ar eclipsa pachetul `app/`.
# Radacina trebuie sa treaca inaintea lui.
sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "data" / "eroare-pornire.log"
TITLE = "OMR Cor"


def _show_error(message: str) -> None:
    """Arata eroarea cu ce e disponibil.

    Ordinea conteaza: daca Qt a pornit, folosim o caseta Qt. Daca nu — si cazul
    cel mai probabil e chiar ca PySide6 lipseste — cadem pe caseta de mesaj din
    Windows, care nu cere nicio dependenta. Ultima varianta ramane consola,
    pentru cine a lansat din terminal.
    """
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        application = QApplication.instance() or QApplication([])
        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle(TITLE)
        box.setText("Aplicatia nu a putut porni.")
        box.setInformativeText(f"{message}\n\nDetalii complete in:\n{LOG_PATH}")
        box.exec()
        return
    except Exception:  # noqa: BLE001 - suntem deja pe calea de esec
        pass

    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            None, f"Aplicatia nu a putut porni.\n\n{message}\n\nDetalii: {LOG_PATH}", TITLE, 0x10
        )
        return
    except Exception:  # noqa: BLE001
        pass

    print(f"{TITLE}: aplicatia nu a putut porni.\n{message}", file=sys.stderr)


def _log(error: BaseException) -> str:
    """Scrie urma completa si intoarce un rezumat de o linie pentru utilizator."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text(
            "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            encoding="utf-8",
        )
    except OSError:
        pass

    if isinstance(error, ModuleNotFoundError):
        return (
            f"Lipseste pachetul {error.name}.\n\n"
            "Reinstaleaza dependentele:\n"
            "  uv pip install -p .venv/Scripts/python.exe PySide6-Essentials"
        )
    return f"{type(error).__name__}: {error}"


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication

        from app.ui.main_window import MainWindow

        application = QApplication(sys.argv)
        application.setApplicationName(TITLE)

        window = MainWindow(out_dir=ROOT / "data" / "processed")
        window.show()
    except BaseException as error:  # noqa: BLE001 - orice esec trebuie sa fie vizibil
        _show_error(_log(error))
        return 1

    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
