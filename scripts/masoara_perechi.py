"""Cat citeste modelul din perechi gata decupate (decupaj de sistem + tinta), fara detectia paginii.

    python scripts/masoara_perechi.py --perechi data/musicorpus/validare.jsonl --greutati x.pt \
        --limita 300 --json iesire.json

Pentru seturi ca MusiCorpus, unde sistemele vin deja decupate langa MusicXML-ul lor: decupajul
merge direct la model (`_preprocess` + decodare, ca in `GrandstaffEngine.transcribe`), iar
ce scrie se compara cu tinta pe **note pe portativ** si pe tokeni (SER), cu
`masoara_manuscris.compare_system`. Alegerea perechilor e stabila (`--limita` din aceeasi ordine).
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--perechi", type=Path, required=True, help="jsonl: image, transcription, staves")
    parser.add_argument("--greutati", type=Path, default=None, help="implicit: modelul publicat")
    parser.add_argument("--limita", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    from app.recognition.bekern import to_kern
    from app.recognition.grandstaff_engine import GrandstaffEngine
    from scripts.masoara_manuscris import compare_system

    randuri = [json.loads(l) for l in args.perechi.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limita and args.limita < len(randuri):
        randuri = random.Random(0).sample(randuri, args.limita)

    engine = GrandstaffEngine(weights=args.greutati, threads=args.threads)
    engine.load()

    total = {"note_tinta": 0, "scrise": 0, "corecte": 0, "distanta": 0, "tokeni": 0}
    pe_portative: dict[int, dict] = {}
    for k, rand in enumerate(randuri, 1):
        staves = int(rand["staves"])
        imagine = np.array(Image.open(rand["image"]).convert("L"))
        tensor, _, _ = engine._preprocess(imagine, staves)
        tokeni = engine._decoder.generate(tensor, max_tokens=engine.max_tokens)
        scris = to_kern(tokeni).kern
        tinta = to_kern(rand["transcription"].split()).kern
        try:
            r = compare_system(tinta, scris, staves)
        except ValueError:
            continue  # tinta nu se citeste inapoi: nu e o masuratoare a modelului
        for cheie in total:
            total[cheie] += r[cheie]
            pe_portative.setdefault(staves, {c: 0 for c in total})[cheie] += r[cheie]
        if k % 25 == 0:
            print(f"  {k}/{len(randuri)}: {total['corecte']}/{total['note_tinta']} note, "
                  f"SER {total['distanta'] / max(1, total['tokeni']):.3f}")

    def rezumat(t: dict) -> dict:
        return {"corecte": t["corecte"], "note_tinta": t["note_tinta"], "scrise": t["scrise"],
                "acuratete": round(t["corecte"] / max(1, t["note_tinta"]), 4),
                "precizie": round(t["corecte"] / max(1, t["scrise"]), 4),
                "ser": round(t["distanta"] / max(1, t["tokeni"]), 4)}

    rez = {"perechi": len(randuri), "greutati": str(args.greutati), **rezumat(total),
           "pe_portative": {str(n): rezumat(t) for n, t in sorted(pe_portative.items())}}
    print(f"TOTAL {args.perechi.name}: {total['corecte']}/{total['note_tinta']} note pe portativ "
          f"({rez['acuratete']:.1%}), precizie {rez['precizie']:.1%}, SER {rez['ser']}")
    for n, t in rez["pe_portative"].items():
        print(f"  {n} portative: {t['corecte']}/{t['note_tinta']} ({t['acuratete']:.1%}), SER {t['ser']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rez, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
