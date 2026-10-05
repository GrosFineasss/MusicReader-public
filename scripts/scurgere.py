"""Spune daca datele de antrenare contin piesa pe care se masoara.

    .venv/Scripts/python.exe scripts/scurgere.py \
        --date data/synthetic/corpus/date.jsonl --set data/validation/set-reper.json

Iese cu 1 daca gaseste scurgere, ca sa se poata pune inaintea unei antrenari
sau intr-o verificare automata. Motivul, in PROJECT.md §15.41.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.scurgere import citeste_corpus, citeste_set, masoara, raport  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cat din corpusul de antrenare vine din piesele masurate."
    )
    parser.add_argument("--date", type=Path, required=True, help="corpusul (date.jsonl)")
    parser.add_argument(
        "--set", type=Path, default=None, dest="set_",
        help="setul de masurare (JSON, ca data/validation/set-reper.json)",
    )
    parser.add_argument(
        "--piese", nargs="+", default=None, metavar="PIESA",
        help="numele pieselor masurate, daca nu ai un set (ex: pagina3)",
    )
    args = parser.parse_args()

    if not args.set_ and not args.piese:
        parser.error("da fie --set, fie --piese")

    masurate = list(args.piese or [])
    if args.set_:
        masurate.extend(citeste_set(args.set_))

    rezultat = masoara(citeste_corpus(args.date), masurate)
    print(raport(rezultat, masurate))
    return 0 if rezultat.curat else 1


if __name__ == "__main__":
    raise SystemExit(main())
