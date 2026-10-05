"""Extrage `PRAIG/polish-scores` ca perechi poza + referinta MusicXML.

    .venv/Scripts/python.exe scripts/polish.py --split val --out data/polish/val

Iese cu poze `.jpg` si referinte `.musicxml` cu acelasi nume, gata de dat lui
`scripts/evaluate.py`. Motivul si limitele setului: `app/training/polish.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.polish import PolishError, musicxml_din_kern, pagina_din_rand  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

DEPOZIT = "PRAIG/polish-scores"


def main() -> int:
    parser = argparse.ArgumentParser(description="polish-scores -> poze + referinte MusicXML")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--out", type=Path, required=True, help="unde se scriu perechile")
    parser.add_argument("--cate", type=int, default=None, help="doar primele N pagini")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    fisier = hf_hub_download(
        DEPOZIT, f"data/{args.split}-00000-of-00001.parquet", repo_type="dataset"
    )
    randuri = pq.read_table(fisier).to_pylist()[: args.cate]
    args.out.mkdir(parents=True, exist_ok=True)

    scrise, sarite = 0, []
    for rand in randuri:
        try:
            pagina = pagina_din_rand(rand)
            xml = musicxml_din_kern(pagina.kern)
        except (PolishError, Exception) as eroare:  # music21 arunca de tot felul
            sarite.append(f"{rand.get('image', {}).get('path', '?')}: {eroare}")
            continue
        (args.out / f"{pagina.nume}.jpg").write_bytes(pagina.imagine)
        (args.out / f"{pagina.nume}.musicxml").write_bytes(xml)
        scrise += 1

    print(f"{scrise}/{len(randuri)} pagini scrise in {args.out}")
    for motiv in sarite:
        print(f"  sarita — {motiv}")
    return 0 if scrise else 1


if __name__ == "__main__":
    raise SystemExit(main())
