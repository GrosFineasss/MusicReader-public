"""Mai multe seturi de perechi -> unul singur, de dat lui `finetune.py`.

    python scripts/combina.py --out data/antrenare-3/date.jsonl \
        data/bach2/antrenare/date.jsonl data/pdmx-randat/antrenare/date.jsonl \
        data/grandstaff/date.jsonl:1500

`finetune.py` primeste un singur `--date`. Sursele stau fiecare in folderul ei,
cu imaginile relative la el; aici caile devin **absolute**, ca setul combinat sa
poata sta oriunde (`load_pairs` alatura calea relativa de folderul fisierului,
iar o cale absoluta ramane ce e).

`:N` dupa o sursa ia cel mult N perechi din ea, alese stabil. Asa se tine
GrandStaff la o cota mica: e acolo ca modelul sa nu uite pianul, nu ca sa
invete pian.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("surse", nargs="+", help="`date.jsonl`, optional cu `:N`")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samanta", type=int, default=0)
    args = parser.parse_args()

    toate = []
    for sursa in args.surse:
        # `C:\...` are si el doua puncte; doar un sufix numeric e o limita.
        if ":" in sursa and sursa.rsplit(":", 1)[1].isdigit():
            cale, limita = sursa.rsplit(":", 1)
        else:
            cale, limita = sursa, ""
        fisier = Path(cale).resolve()
        randuri = [json.loads(l) for l in fisier.read_text(encoding="utf-8").splitlines() if l.strip()]
        if limita:
            # Samanta tine si de sursa: doua seturi generate din aceleasi piese in
            # stiluri diferite (`voci`, `acorduri`) au perechile in aceeasi ordine,
            # iar aceeasi samanta ar alege din amandoua aceeasi muzica.
            alegere = random.Random(f"{args.samanta}:{Path(cale).as_posix()}")
            randuri = alegere.sample(randuri, min(int(limita), len(randuri)))
        for rand in randuri:
            rand["image"] = str((fisier.parent / rand["image"]).resolve())
        toate.extend(randuri)
        print(f"{fisier}: {len(randuri)} perechi")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for rand in toate:
            handle.write(json.dumps(rand, ensure_ascii=False) + "\n")
    print(f"total {len(toate)} perechi in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
