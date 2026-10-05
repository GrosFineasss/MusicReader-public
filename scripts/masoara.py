"""Cat citeste modelul pe **mai multe pagini**, defalcat pe mana.

`scripts/evaluate.py` masoara o pagina. Intrebarea produsului e alta: cat
citeste dintr-un scris pe care nu l-a vazut — deci mai multe pagini, agregate si
defalcate pe scriitor (PROJECT.md §15.12).

Setul se da ca fisier, nu ca sir de optiuni, fiindca **fiecare pagina are alte
setari**: numarul de portative pe sistem variaza de la pagina la pagina (§15.8),
iar cele cu voce de solo cer si o asezare.

    python scripts/masoara.py --set data/validation/set.json --greutati models/x.pt

Formatul, o lista de intrari:

    [
      {"imagine": "data/raw/.../pagina3.png",
       "referinta": "data/validation/pagina3.musicxml",
       "scriitor": "manuscris-1994",
       "portative": 2},
      {"imagine": "...", "referinta": "...", "scriitor": "culegerea-verde",
       "portative": [2, 2, 3], "asezare": [["X1"], ["S", "A"], ["T", "B"]]}
    ]

**Agregarea se face pe componente, nu pe medii de SER.** Media a doua rate ar da
alt numar decat rata calculata pe totalul notelor — greseala clasica, si aici ar
face ca o pagina scurta sa cantareasca cat una lunga.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# Campurile care se aduna. Pauzele se numara separat fiindca **doua pauze de
# aceeasi durata ies „identice"** — niciuna n-are inaltime — si umflau cifra:
# §15.42 a gasit ca jumatate din „notele corecte" ale proiectului erau perechi de
# pauze. `evaluate.py` s-a corectat atunci; aici corectia lipsea, deci fiecare
# cifra pe mai multe pagini data pana la §15.57 e la nivel de eveniment.
COMPONENTE = (
    "ref_events", "hyp_events", "matched", "pitch_wrong", "duration_wrong",
    "both_wrong", "missing", "extra", "ref_measures", "hyp_measures",
    "measures_exact", "ref_rests", "hyp_rests", "matched_rests",
)


def aduna(rezultate: list[dict]) -> dict:
    """Componentele insumate, si cifrele derivate din ele.

    Se aduna **numaratori**, nu rate: media a doua SER-uri nu e SER-ul totalului,
    iar diferenta creste cu cat paginile difera mai mult ca lungime.
    """
    total = {camp: sum(r[camp] for r in rezultate) for camp in COMPONENTE}
    referinta = total["ref_events"] or 1
    substituiri = total["pitch_wrong"] + total["duration_wrong"] + total["both_wrong"]

    # Acuratetea se raporteaza pe note; SER ramane pe evenimente, cu pauze cu
    # tot, fiindca asa il da literatura si doar asa e comparabil (§15.44).
    total["ref_note"] = total["ref_events"] - total["ref_rests"]
    total["hyp_note"] = total["hyp_events"] - total["hyp_rests"]
    total["corecte_note"] = total["matched"] - total["matched_rests"]
    total["acuratete_note"] = total["corecte_note"] / (total["ref_note"] or 1)
    total["precizie_note"] = total["corecte_note"] / (total["hyp_note"] or 1)
    total["ser"] = (substituiri + total["missing"] + total["extra"]) / referinta
    total["pagini"] = len(rezultate)
    return total


def asezare_din(intrare: dict) -> list[list[list[str]]] | None:
    """`asezare` din set, adusa la forma motorului: o **lista de asezari** (`parse_layout`).

    Setul o poate da in trei feluri: textul de la `--asezare` (`"X1/S,A/T,B"`), o singura asezare
    (`[["X1"], ["S", "A"], ["T", "B"]]`, ca in exemplul de mai sus) sau cate una pe sistem. Pana pe
    2026-10-05 a doua forma ajungea la motor neschimbata, unde fiecare portativ era luat drept
    asezarea unui sistem — primul sistem primea `["X1"]`, desfacut in voci „X" si „1" — si
    paginile cu solo ieseau la 3,7% cu un model care le citeste.
    """
    asezare = intrare.get("asezare")
    if asezare is None:
        return None
    if isinstance(asezare, str):
        from app.staff_detection.staves import parse_layout

        return parse_layout(asezare)
    if asezare and all(isinstance(voce, str) for portativ in asezare for voce in portativ):
        return [asezare]  # o singura asezare, pentru toate sistemele
    return asezare


def masoara_pagina(intrare: dict, weights: Path | None, kind: str, beam: int = 1) -> dict:
    """O pagina: transcriere + comparatie, pe drumul din `evaluate.py`.

    Se refoloseste `transcribe` de acolo, nu se scrie o a doua conducta: doua
    drumuri care ar putea sa nu fie de acord ar face cifrele de necomparat cu
    reperul inghetat.
    """
    from app.evaluation.metrics import compare
    from app.evaluation.reference import load
    from scripts.evaluate import transcribe

    ipoteza = transcribe(
        Path(intrare["imagine"]),
        kind,
        intrare.get("max_tokens"),
        staves_per_system=intrare.get("portative"),
        layout=asezare_din(intrare),
        weights=weights,
        clean=intrare.get("curata", True),
        beam=beam,
    )
    rezultat = compare(load(Path(intrare["referinta"])), ipoteza)
    total = rezultat.total
    iesire = {camp: getattr(total, camp) for camp in COMPONENTE}
    iesire["ser"] = float(total.ser)
    iesire["scriitor"] = intrare.get("scriitor", Path(intrare["imagine"]).stem)
    iesire["imagine"] = str(intrare["imagine"])
    return iesire


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--set", dest="fisier", type=Path, required=True)
    parser.add_argument("--greutati", type=Path, default=None)
    parser.add_argument("--tip", default="handwritten", choices=("handwritten", "printed"))
    parser.add_argument("--json", type=Path, default=None, help="unde se scriu cifrele")
    parser.add_argument(
        "--beam", type=int, default=1,
        help="ipoteze tinute la decodare; 1 = lacoma, decodarea tuturor reperelor de pana acum",
    )
    args = parser.parse_args()

    intrari = json.loads(args.fisier.read_text(encoding="utf-8"))
    lipsa = [
        i["imagine"] for i in intrari
        if not Path(i["imagine"]).exists() or not Path(i["referinta"]).exists()
    ]
    if lipsa:
        raise SystemExit(f"lipsesc fisiere: {', '.join(str(x) for x in lipsa[:5])}")

    rezultate = []
    for intrare in intrari:
        rezultat = masoara_pagina(intrare, args.greutati, args.tip, args.beam)
        rezultate.append(rezultat)
        print(
            f"  {rezultat['scriitor']:22} "
            f"{rezultat['matched'] - rezultat['matched_rests']:>4}/"
            f"{rezultat['ref_events'] - rezultat['ref_rests']:<4} note"
            f" | SER {rezultat['ser']:.3f}"
            f" | {rezultat['measures_exact']}/{rezultat['ref_measures']} masuri"
        )

    pe_mana: dict[str, list[dict]] = {}
    for rezultat in rezultate:
        pe_mana.setdefault(rezultat["scriitor"], []).append(rezultat)

    print()
    if len(pe_mana) > 1:
        print("pe mana:")
        for nume, ale_lui in sorted(pe_mana.items()):
            t = aduna(ale_lui)
            print(
                f"  {nume:22} {t['corecte_note']:>4}/{t['ref_note']:<4} note"
                f" | {100 * t['acuratete_note']:5.1f}% | SER {t['ser']:.3f}"
                f" | {t['pagini']} pagini"
            )
        print()

    total = aduna(rezultate)
    print(
        f"TOTAL: {total['corecte_note']}/{total['ref_note']} note "
        f"({100 * total['acuratete_note']:.1f}%), precizie "
        f"{100 * total['precizie_note']:.1f}%, SER {total['ser']:.3f}, "
        f"{total['measures_exact']}/{total['ref_measures']} masuri, "
        f"{total['pagini']} pagini, {len(pe_mana)} maini"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"pagini": rezultate, "total": total}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"cifrele in {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
