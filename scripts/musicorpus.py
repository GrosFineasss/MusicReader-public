"""MusiCorpus v1.0 -> perechi de antrenare (decupaj de sistem scris de mana, tinta bekern).

    python scripts/musicorpus.py --sursa data/raw/musicorpus/musicorpus_v1.0 --out data/musicorpus

MusiCorpus (LINDAT, hdl.handle.net/20.500.12800/1-6147, CC BY-NC-SA 4.0): 1.309 pagini, 1.287
scrise de mana, 79 de manuscrise; fiecare sistem are **decupajul lui si MusicXML-ul lui**
(`<doc>/Systems/<n>/image.jpg` + `transcription.musicxml`), deci perechile nu se mai potrivesc
dupa rupturi, ca la `dataset.py` — vin gata potrivite. Licenta e necomerciala; aplicatia e pentru
uz personal (Fineas, 2026-10-05), deci poate intra in ea (`publica_greutati.py --necomercial`).

Ce se pastreaza: sistemele **scrise de mana** cu 1-3 portative (melodie, pian, voce cu pian, ca
partiturile de cor); orchestrele (pana la 20+ portative) s-ar turti de nerecunoscut la inaltimea
modelului. Imaginea trece prin acelasi alb-negru ca la citire (`binarize`, Otsu), cu cerneala
neagra pe alb ca decupajele din `bach.py`. Tinta trece prin acelasi drum ca a tuturor perechilor:
MusicXML -> `reference.load` -> `to_kern_text(ties=False)` -> `from_kern`.

Ce se tine deoparte: 10% din **manuscrise** (dupa cota, `shelfmark`), ales stabil — un manuscris e
o mana, deci validarea masoara scris nevazut, nu pagini nevazute ale aceleiasi mani.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PORTATIVE_MAXIME = 3


def manuscris(metadata: dict, document: Path) -> str:
    """Al cui e scrisul: cota manuscrisului (o cota = o copie = o mana), altfel documentul."""
    cota = metadata.get("shelfmark") or metadata.get("institution_local_siglum")
    return f"{document.parent.name}:{cota}" if cota else f"{document.parent.name}:{document.name}"


def converteste_sistem(sarcina: tuple[str, str, str]) -> dict:
    """Un sistem -> pereche, sau motivul pentru care nu se poate. Ruleaza in alt proces."""
    sistem, iesire, nume = sarcina
    sistem, iesire = Path(sistem), Path(iesire)
    try:
        from app.evaluation.reference import load
        from app.recognition.bekern import from_kern
        from app.recognition.kern_writer import to_kern_text

        score = load(sistem / "transcription.musicxml")
        portative = len(score.staves) or len(score.parts)
        if not 1 <= portative <= PORTATIVE_MAXIME:
            return {"refuz": f"{portative} portative"}
        text = to_kern_text(score, ties=False)
        tokeni = " ".join(from_kern(text))
    except Exception as problema:  # noqa: BLE001 — orice notatie pe care n-o scriem e un refuz
        return {"refuz": f"{type(problema).__name__}: {str(problema)[:80]}"}

    import cv2
    import numpy as np
    from PIL import Image

    from app.preprocessing import binarize

    gri = np.array(Image.open(sistem / "image.jpg").convert("L"))
    imagine = cv2.bitwise_not(binarize(gri))
    cv2.imwrite(str(iesire / f"{nume}.png"), imagine)
    return {"image": f"{nume}.png", "transcription": tokeni, "staves": portative}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sursa", type=Path, required=True, help="folderul `musicorpus_v1.0`")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--procent-validare", type=int, default=10)
    parser.add_argument("--procese", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--limita", type=int, default=None, help="doar primele N sisteme (proba)")
    args = parser.parse_args()

    perechi_dir = args.out / "perechi"
    perechi_dir.mkdir(parents=True, exist_ok=True)

    sarcini, al_cui = [], {}
    for meta_cale in sorted(args.sursa.glob("*/*/metadata.json")):
        metadata = json.loads(meta_cale.read_text(encoding="utf-8"))
        if metadata.get("production") != "handwritten":
            continue
        document = meta_cale.parent
        mana = manuscris(metadata, document)
        for sistem in sorted(document.glob("Systems/*")):
            if not (sistem / "image.jpg").exists() or not (sistem / "transcription.musicxml").exists():
                continue
            nume = f"{document.parent.name[:4]}_{document.name}_{sistem.name}".replace(" ", "_")
            sarcini.append((str(sistem), str(perechi_dir), nume))
            al_cui[nume] = mana
    sarcini = sarcini[: args.limita]

    maini = sorted(set(al_cui.values()))
    deoparte = set(random.Random(0).sample(maini, max(1, round(len(maini) * args.procent_validare / 100))))
    print(f"{len(sarcini)} sisteme scrise de mana din {len(maini)} manuscrise; "
          f"{len(deoparte)} manuscrise tinute deoparte")

    refuzuri, randuri = Counter(), {"antrenare": [], "validare": []}
    with ProcessPoolExecutor(max_workers=args.procese) as pool:
        for k, (sarcina, rezultat) in enumerate(
            zip(sarcini, pool.map(converteste_sistem, sarcini, chunksize=16)), 1
        ):
            nume = sarcina[2]
            if "refuz" in rezultat:
                refuzuri[rezultat["refuz"].split(":")[0]] += 1
                continue
            rezultat["writer"] = al_cui[nume]
            randuri["validare" if al_cui[nume] in deoparte else "antrenare"].append(rezultat)
            if k % 500 == 0:
                print(f"  {k}/{len(sarcini)}: {sum(len(v) for v in randuri.values())} perechi")

    for parte, lista in randuri.items():
        cale = args.out / ("date.jsonl" if parte == "antrenare" else "validare.jsonl")
        with cale.open("w", encoding="utf-8") as fisier:
            for rand in lista:
                rand = dict(rand, image=str((perechi_dir / rand["image"]).resolve()))
                fisier.write(json.dumps(rand, ensure_ascii=False) + "\n")
    pe_portative = Counter(r["staves"] for lista in randuri.values() for r in lista)
    raport = {
        "perechi_antrenare": len(randuri["antrenare"]), "perechi_validare": len(randuri["validare"]),
        "pe_portative": dict(sorted(pe_portative.items())), "refuzuri": dict(refuzuri.most_common()),
        "manuscrise_deoparte": sorted(deoparte),
    }
    (args.out / "raport.json").write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in raport.items() if k != "manuscrise_deoparte"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
