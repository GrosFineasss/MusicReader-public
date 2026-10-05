"""Poza + referinta -> perechi (decupaj de sistem, tinta bekern), pentru antrenare.

Piesa care lipsea ca sa se poata antrena pe paginile tale (PROJECT.md §15.7).
Modelul invata din perechi (imagine, sir de tokeni); noi aveam imaginile si
aveam referinta in MusicXML, dar nimic care sa le puna fata in fata.

Cum se imperecheaza, fara sa ghicim nimic: **MuseScore scrie in MusicXML unde
rupe randul** (`<print new-system="yes">`), iar tu asezi partitura ca
manuscrisul. Deci al n-lea sistem decupat din poza primeste exact masurile
sistemului al n-lea din referinta.

    python scripts/dataset.py --imagine pagina3.png --ref pagina3.musicxml \
        --out data/antrenare --scriitor ionut

**Fiecare pereche poarta si al cui e scrisul.** Cele ~500 de partituri sunt
fiecare in alt scris, deci un model antrenat si masurat pe aceleasi maini ar
raporta memorare, nu citire. Impartirea antrenare/validare se face **pe
scriitor**, si de asta eticheta se cere aici, la producere: informatia nu se
mai poate recupera din fisier mai tarziu.

Ce se scrie in `--out`: cate un `sistemNN.png`, cate un `sistemNN.krn` de citit
cu ochiul, si `date.jsonl` cu perechile in forma pe care o asteapta SMT —
`image` si `transcription` (vezi sectiunea Data din `vendor/SMT/README.md`).

**Daca numarul de sisteme din poza nu e cel din referinta, nu se scrie nimic.**
O imperechere gresita ar invata modelul sa citeasca alte note decat cele din
imagine, si n-ar avea cine sa observe.
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

from app.music_model.model import Score, drop_silent_staves, measures_slice  # noqa: E402
from app.recognition.bekern import from_kern  # noqa: E402
from app.recognition.kern_writer import to_kern_text  # noqa: E402
from app.staff_detection.staves import parse_grouping  # noqa: E402


def system_bounds(starts: list[int], total: int) -> list[tuple[int, int]]:
    """Perechile (prima masura, ultima masura) ale fiecarui sistem."""
    return list(zip(starts, [start - 1 for start in starts[1:]] + [total]))


def targets_for(
    score: Score, bounds: list[tuple[int, int]], staves: list[int] | None = None
) -> list[tuple[str, str]]:
    """Pentru fiecare sistem: (text Humdrum, tinta ca sir de tokeni).

    Textul se pastreaza pe langa tokeni fiindca e singurul dintre cele doua pe
    care il poti citi cu ochiul cand ceva pare gresit.

    `staves`, cand se da, e numarul de portative din fiecare decupaj: un sistem
    desenat cu mai putine portative decat are referinta pierde portativele care
    tac pe el (`drop_silent_staves`), ca tinta sa aiba ce se vede in imagine.
    """
    pairs = []
    for index, (first, last) in enumerate(bounds):
        piece = measures_slice(score, first, last)
        if staves is not None:
            piece = drop_silent_staves(piece, staves[index])
        # `ties=False`: vocabularul modelului n-are niciun token de legato —
        # 2552 de tokeni cu `[`, dar toti armuri. O tinta care poarta `[2f#` face
        # sistemul intreg sa cada la verificarea de vocabular. Masurat pe
        # 2026-09-03: 6 din 10 sisteme din „Doamne bun asculta" se pierdeau asa,
        # iar fara marcaje niciunul, cu dus-intorsul tot exact.
        text = to_kern_text(piece, ties=False)
        pairs.append((text, " ".join(from_kern(text))))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--imagine", type=Path, required=True, help="poza paginii")
    parser.add_argument("--ref", type=Path, required=True, help="referinta MusicXML a paginii")
    parser.add_argument("--out", type=Path, required=True, help="unde se scriu perechile")
    parser.add_argument(
        "--portative-pe-sistem", type=parse_grouping, default=2,
        metavar="N|N,N,...",
        help="cate portative are un sistem: un numar (`2`) sau cate unul pe "
             "sistem (`2,2,2,3,3`)",
    )
    parser.add_argument("--curata", action="store_true")
    parser.add_argument(
        "--pagina", type=int, default=None, metavar="N",
        help="a cata pagina a referintei e in poza (1 = prima); implicit, toata referinta",
    )
    parser.add_argument(
        "--scriitor", required=True, metavar="ID",
        help="al cui e scrisul de pe pagina (`ionut`, `manuscris-1994`, ...). "
             "Obligatoriu: validarea se face pe scriitor tinut deoparte, iar fara "
             "eticheta asta n-am sti daca modelul citeste sau isi aminteste",
    )
    args = parser.parse_args()

    from app.evaluation.reference import load_with_layout

    score, starts, pages = load_with_layout(args.ref)
    bounds = system_bounds(starts, score.measure_count)
    print(f"referinta: {score.measure_count} masuri, {len(bounds)} sisteme, {len(pages)} pagini")

    if args.pagina is not None:
        if not 1 <= args.pagina <= len(pages):
            raise SystemExit(
                f"referinta are {len(pages)} pagini, ceruta e {args.pagina}"
            )
        first = pages[args.pagina - 1]
        last = pages[args.pagina] - 1 if args.pagina < len(pages) else score.measure_count
        bounds = [pair for pair in bounds if first <= pair[0] <= last]
        print(f"pagina {args.pagina}: masurile {first}-{last}")
    elif len(pages) > 1:
        raise SystemExit(
            f"referinta are {len(pages)} pagini, iar poza e a uneia singure — "
            "spune care, cu `--pagina N`"
        )

    print(f"sisteme de imperecheat: {len(bounds)} {bounds}")

    from app.recognition.grandstaff_engine import GrandstaffEngine

    engine = GrandstaffEngine(
        staves_per_system=args.portative_pe_sistem, clean=args.curata
    )
    engine.load()
    crops = engine.detect_systems(args.imagine)
    print(f"poza: {len(crops)} sisteme decupate")

    if len(crops) != len(bounds):
        raise SystemExit(
            f"poza are {len(crops)} sisteme, referinta are {len(bounds)} — nu se pot "
            "imperechea. Verifica rupturile de rand din MuseScore: trebuie sa fie "
            "acolo unde le are manuscrisul."
        )

    import cv2

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    from app.recognition.kern_writer import UnwritableScore

    try:
        targets = targets_for(score, bounds, [staves for _, _, staves in crops])
    except UnwritableScore as problem:
        # Tot ValueError, dar alta cauza: notatie pe care Humdrum n-o poate scrie, nu rupturi.
        raise SystemExit(f"referinta nu se poate scrie ca tinta: {problem}") from problem
    except ValueError as problem:
        raise SystemExit(f"decupajele nu se potrivesc cu referinta: {problem}") from problem
    for index, ((_, image, staves), (text, tokens)) in enumerate(zip(crops, targets), 1):
        name = f"{args.imagine.stem}_sistem{index:02d}"
        cv2.imwrite(str(args.out / f"{name}.png"), image)
        (args.out / f"{name}.krn").write_text(text, encoding="utf-8")
        # `staves` se scrie fiindca de el depinde scara la care se aduce
        # imaginea inainte de model (`resize_for_model`). Decupajul e salvat
        # brut, ca sa se poata privi cu ochiul; scara se aplica la incarcare,
        # exact ca la inferenta. Fara numarul asta, antrenarea ar scala altfel
        # decat inferenta, si nimic n-ar semnala diferenta.
        rows.append({
            "image": f"{name}.png",
            "transcription": tokens,
            "staves": staves,
            "writer": args.scriitor,
        })
        first, last = bounds[index - 1]
        print(
            f"  sistem {index}: masurile {first}-{last}, {staves} portative, "
            f"{len(tokens.split())} tokeni"
        )

    lines = args.out / "date.jsonl"
    with lines.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} perechi adaugate in {lines} (scriitor: {args.scriitor})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
