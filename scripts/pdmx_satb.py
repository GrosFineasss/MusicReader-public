"""Corurile SATB din PDMX: alegere dupa metadate, extragere, verificare.

    python scripts/pdmx_satb.py --pdmx data/pdmx

PDMX (Zenodo 15571083) are 254.000 de partituri MusicXML de pe MuseScore. Se
folosesc doar cele **fara conflict de licenta** (222.856, domeniu public / CC0)
si **deduplicate** — deci, spre deosebire de coralele din `music21`, pot intra
si intr-un produs.

Trei pasi, fiecare reluabil:

1. **alegerea**, din `PDMX.csv`: patru piste vocale (SATB deschis), doua piste
   vocale (partitura stransa), patru vocale plus acompaniament, sau un pian ori
   o orga al carui gen, eticheta sau titlu spune imn / SATB / choir / chorale.
2. **extragerea** din `mxl.tar.gz`, doar a fisierelor alese.
3. **verificarea** fiecarei piese, pe muzica, nu pe metadate:
   `app.training.satb.cor_satb` (iese exact SATB?) si `probleme` (ambitus,
   ordinea vocilor, cat tace fiecare, unison perpetuu). Plus **scurgerea**: o
   piesa a carei melodie e o corala Bach din corpus, sau chiar pagina pe care
   se masoara, nu intra — ar umfla o cifra de validare.

Iese `acceptate.json` (de dat lui `scripts/bach.py --lista`) si
`verificare.jsonl`, cu motivul fiecarei piese refuzate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

VOCE = {"52", "53", "54"}
PIAN_SAU_ORGA = (["0"], ["19"], ["0", "0"], ["19", "19"])
CUVINTE_IMN = ("hymn", "satb", "chorale", "choir")


def forma(rand: dict) -> str | None:
    """Ce forma de cor are piesa, dupa metadate; `None` daca niciuna."""
    programe = [t for t in rand["tracks"].split("-") if t]
    vocale = sum(t in VOCE for t in programe)
    if len(programe) == 4 and vocale == 4:
        return "4-voci"
    if len(programe) == 2 and vocale == 2:
        return "stransa"
    if vocale == 4 and len(programe) in (5, 6):
        return "4-voci+acomp"
    text = f"{rand['genres']}-{rand['tags']}-{rand['title']}".lower()
    if programe in PIAN_SAU_ORGA and any(cuvant in text for cuvant in CUVINTE_IMN):
        return "imn-pe-pian"
    return None


def alege(csv_cale: Path) -> list[dict]:
    csv.field_size_limit(10**9)
    alese = []
    with csv_cale.open(encoding="utf-8", newline="") as handle:
        for rand in csv.DictReader(handle):
            if rand["subset:no_license_conflict"] != "True":
                continue
            if rand["subset:deduplicated"] != "True":
                continue
            f = forma(rand)
            if f is None:
                continue
            alese.append({
                "mxl": rand["mxl"].lstrip("./"),
                "forma": f,
                "titlu": rand["title"],
                "compozitor": rand["composer_name"],
                "licenta": rand["license"],
            })
    return alese


def extrage(arhiva: Path, alese: list[dict], destinatie: Path) -> int:
    """Doar fisierele alese, dintr-o singura trecere prin arhiva."""
    dorite = {a["mxl"] for a in alese if not (destinatie / a["mxl"]).exists()}
    if not dorite:
        return 0
    scoase = 0
    with tarfile.open(arhiva, "r:gz") as tar:
        for membru in tar:
            nume = membru.name.lstrip("./")
            if nume in dorite and membru.isfile():
                tinta = destinatie / nume
                tinta.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(membru) as sursa:
                    tinta.write_bytes(sursa.read())
                scoase += 1
    return scoase


def verifica_una(cale: str) -> dict:
    """O piesa: se converteste, se verifica, se ia amprenta. Ruleaza in alt proces."""
    from music21 import converter

    from app.training.satb import amprenta, cor_satb, probleme

    try:
        score = cor_satb(converter.parse(cale))
        gasite = probleme(score)
        return {
            "cale": cale, "ok": not gasite, "motiv": "; ".join(gasite),
            "masuri": score.measure_count, "amprenta": list(amprenta(score)),
        }
    except Exception as eroare:  # noqa: BLE001 — orice esec refuza piesa, cu motivul scris
        return {"cale": cale, "ok": False, "motiv": f"{type(eroare).__name__}: {eroare}"[:300]}


def amprente_interzise() -> dict[tuple, str]:
    """Melodiile care nu au voie in antrenare: coralele Bach si paginile masurate."""
    from music21 import converter, corpus

    from app.evaluation.reference import load
    from app.training.bach import coral_satb
    from app.training.satb import amprenta

    interzise: dict[tuple, str] = {}
    for cale in corpus.getComposer("bach"):
        if cale.suffix != ".mxl":
            continue
        try:
            interzise[amprenta(coral_satb(converter.parse(str(cale))), 12)] = f"bach/{cale.stem}"
        except Exception:  # noqa: BLE001 — coralele refuzate de `coral_satb` nu sunt in set
            continue
    for referinta in ("pagina3.musicxml", "doamne-bun-asculta.musicxml"):
        score = load(ROOT / "data" / "validation" / referinta)
        if "S" in [p.id for p in score.parts]:
            interzise[amprenta(score, 12)] = referinta
    return interzise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pdmx", type=Path, required=True, help="folderul cu PDMX.csv si mxl.tar.gz")
    parser.add_argument("--procese", type=int, default=3)
    args = parser.parse_args()

    alese = alege(args.pdmx / "PDMX.csv")
    print(f"alese dupa metadate: {len(alese)} {dict(Counter(a['forma'] for a in alese))}")
    (args.pdmx / "candidate.json").write_text(
        json.dumps(alese, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    scoase = extrage(args.pdmx / "mxl.tar.gz", alese, args.pdmx)
    print(f"extrase acum: {scoase}")

    jurnal = args.pdmx / "verificare.jsonl"
    facute = {}
    if jurnal.exists():
        for linie in jurnal.read_text(encoding="utf-8").splitlines():
            rand = json.loads(linie)
            facute[rand["cale"]] = rand
    de_facut = [str(args.pdmx / a["mxl"]) for a in alese if str(args.pdmx / a["mxl"]) not in facute]
    print(f"de verificat: {len(de_facut)} (deja: {len(facute)})")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=args.procese) as pool, jurnal.open("a", encoding="utf-8") as out:
        viitoare = [pool.submit(verifica_una, cale) for cale in de_facut]
        for i, viitor in enumerate(as_completed(viitoare), 1):
            rand = viitor.result()
            facute[rand["cale"]] = rand
            out.write(json.dumps(rand, ensure_ascii=False) + "\n")
            out.flush()
            if i % 200 == 0:
                print(f"  verificate {i}/{len(de_facut)}", flush=True)

    interzise = amprente_interzise()
    pe_forma = {str(args.pdmx / a["mxl"]): a for a in alese}
    acceptate, motive = [], Counter()
    vazute: set[tuple] = set()
    for cale, rand in facute.items():
        if not rand["ok"]:
            motive[rand["motiv"].split(":")[0].split(" (")[0][:60]] += 1
            continue
        cheie = tuple(rand["amprenta"][:12])
        if cheie in interzise:
            motive[f"scurgere: aceeasi melodie ca {interzise[cheie].split('/')[0]}"] += 1
            continue
        if len(cheie) >= 12 and cheie in vazute:
            motive["dublura (aceeasi melodie)"] += 1
            continue
        vazute.add(cheie)
        meta = pe_forma.get(cale, {})
        acceptate.append({
            "cale": cale, "id": "pdmx-" + Path(cale).stem[:12],
            "forma": meta.get("forma"), "titlu": meta.get("titlu"), "masuri": rand["masuri"],
        })

    (args.pdmx / "acceptate.json").write_text(
        json.dumps(acceptate, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nacceptate: {len(acceptate)} din {len(facute)}, "
          f"{sum(a['masuri'] for a in acceptate)} de masuri")
    print(f"pe forma: {dict(Counter(a['forma'] for a in acceptate))}")
    print("motivele refuzului, cele mai dese:")
    for motiv, n in motive.most_common(15):
        print(f"  {n:5}  {motiv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
