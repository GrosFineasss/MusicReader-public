"""Un esantion din GrandStaff, de amestecat printre corale la antrenare.

    python scripts/grandstaff.py --out data/grandstaff --cate 1500

De ce. Antrenat numai pe corale Bach, modelul a castigat pe `pagina3` tiparita
(49 -> 91 de note) si a **pierdut pe pian**: scanurile `polish` au coborat de la
1.023 la 683 de note (§15.59). E uitarea obisnuita a unui model specializat.
Remediul obisnuit e sa vada, printre datele noi, si din cele pe care a invatat
la inceput — GrandStaff, pe care a fost antrenat `smt-grandstaff`. Licenta e
**MIT**, deci merge si in produs.

Se ia impartirea `val` a setului (4.623 de sisteme de pian, 45 MB), nu `train`:
e destul pentru amestec, si e de zece ori mai mica de descarcat.

Transcrierile sunt `**ekern` cu separatori `@` si `·`; se curata cu acelasi cod
care curata `polish-scores` (`ekern_la_kern`) si se traduc in tokenii modelului
cu `from_kern`, ca orice tinta din proiect. Iese `date.jsonl` in forma lui
`dataset.py`, cu `staves = 2` (portativ dublu de pian) si eticheta
`grandstaff-<piesa>`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from app.recognition.bekern import from_kern  # noqa: E402
from app.training.polish import PolishError, ekern_la_kern  # noqa: E402

DEPOZIT = "PRAIG/grandstaff"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cate", type=int, default=1500)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--samanta", type=int, default=0)
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    fisier = hf_hub_download(DEPOZIT, f"data/{args.split}-00000-of-00001.parquet", repo_type="dataset")
    tabel = pq.read_table(fisier)
    alese = sorted(random.Random(args.samanta).sample(range(tabel.num_rows), min(args.cate, tabel.num_rows)))
    args.out.mkdir(parents=True, exist_ok=True)

    randuri, sarite = [], 0
    for index in alese:
        rand = tabel.slice(index, 1).to_pylist()[0]
        try:
            # Exponentul vine cu versiune, `**ekern_1.0`; `ekern_la_kern` l-ar face
            # `**kern_1.0`, pe care `from_kern` nu-l recunoaste si l-ar lasa ca token
            # strain — si fiecare pereche ar cadea la verificarea de vocabular.
            text = rand["transcription"].replace("**ekern_1.0", "**ekern")
            tokeni = from_kern(ekern_la_kern(text))
        except PolishError:
            sarite += 1
            continue
        piesa = Path(rand["image"].get("path") or f"gs{index}").stem
        nume = f"gs-{index:05d}.png"
        Image.open(io.BytesIO(rand["image"]["bytes"])).convert("L").save(args.out / nume)
        randuri.append({
            "image": nume, "transcription": " ".join(tokeni),
            "staves": 2, "writer": f"grandstaff-{piesa.rsplit('-', 2)[0]}",
        })

    with (args.out / "date.jsonl").open("w", encoding="utf-8") as handle:
        for rand in randuri:
            handle.write(json.dumps(rand, ensure_ascii=False) + "\n")
    print(f"{len(randuri)} perechi GrandStaff in {args.out / 'date.jsonl'} ({sarite} sarite)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
