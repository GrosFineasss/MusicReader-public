"""Cat citeste modelul dintr-un manuscris, sistem cu sistem, fara sa depinda de impartirea pe voci.

    python scripts/masoara_manuscris.py --imagine data/raw/scris-de-mana/doamne-bun-asculta-p1.png \
        --perechi data/doamne-manuscris --portative 2,2,2,3,3 --greutati models/x/pas9500.pt \
        --json iesire.json

Tintele sunt `.krn`-urile scrise de `dataset.py` langa decupaje (`<imagine>_sistemNN.krn`).
Tinta si ce scrie modelul trec prin **acelasi** cititor de kern, cu aceeasi asezare, iar pe fiecare
portativ si masura se compara multimile de (debut, inaltime) ale notelor, din toate vocile
portativului la un loc. Plus SER pe tokeni, pe sistem.

De ce nu `evaluate.py`: compara voce cu voce, iar pe o cantare cu solo si pian vocile se despart
si se unesc altfel de la o scriere la alta. Pe „Doamne bun asculta" (2026-10-04) dadea 24 de note
din 327 pe o pagina pe care modelul o scria aproape identic cu tinta — o voce despartita cu o
masura mai devreme muta toate notele pe alta voce. Aici un unison scris de doua ori sau o voce
despartita altfel nu mai conteaza; inaltimea, debutul si portativul, da.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from app.music_model.model import Note  # noqa: E402
from app.recognition.bekern import from_kern  # noqa: E402
from app.recognition.kern import parse_open_system  # noqa: E402
from app.staff_detection.staves import parse_grouping  # noqa: E402

# Asezarea dupa numarul de portative: unul = melodie (MusiCorpus), doua = pian/cor strans, trei = solo deasupra.
ASEZARE = {1: [["S", "A"]], 2: [["S", "A"], ["T", "B"]], 3: [["X1"], ["S", "A"], ["T", "B"]]}

Notes = dict[tuple[int, int], set[tuple[Fraction, str, int, int]]]


def levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def notes_per_staff(text: str, staves: int) -> tuple[Notes | None, str | None]:
    """{(portativ de sus, masura): {(debut, pas, octava, alteratie)}}; `None` + motivul daca nu se citeste."""
    try:
        score = parse_open_system(text, staves=staves, layout=ASEZARE[staves]).score
    except Exception as problem:  # noqa: BLE001 — o iesire ilizibila e un rezultat, nu o eroare
        return None, str(problem)
    staff_of = {voice: k for k, voices in enumerate(score.staves) for voice in voices}
    for part in score.parts:
        if part.id not in staff_of and part.id.startswith("X") and "_" in part.id:
            # voce in plus, numita dupa portativul ei numarat de jos (`X{k}_i`, `kern.py`)
            staff_of[part.id] = staves - int(part.id[1:].split("_")[0])
    found: Notes = {}
    for part in score.parts:
        k = staff_of.get(part.id)
        if k is None:
            continue
        for index, measure in enumerate(part.measures, 1):
            onset = Fraction(0)
            for event in measure.events:
                if isinstance(event, Note):
                    p = event.pitch
                    found.setdefault((k, index), set()).add((onset, p.step, p.octave, p.alter))
                onset += event.quarter_length
    return found, None


def compare_system(target: str, written: str, staves: int) -> dict:
    """Un sistem: note ale tintei, scrise, corecte (pe portativ), distanta de editare pe tokeni."""
    target_tokens = from_kern(target)
    written_tokens = from_kern(written) if written.strip() else []
    expected, _ = notes_per_staff(target, staves)
    if expected is None:
        raise ValueError("tinta nu se poate citi")
    got, problem = notes_per_staff(written, staves) if written.strip() else (None, "gol")
    correct = 0 if got is None else sum(len(expected.get(key, set()) & s) for key, s in got.items())
    return {
        "note_tinta": sum(len(s) for s in expected.values()),
        "scrise": 0 if got is None else sum(len(s) for s in got.values()),
        "corecte": correct,
        "distanta": levenshtein(written_tokens, target_tokens),
        "tokeni": len(target_tokens),
        "problema": problem,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--imagine", type=Path, required=True, help="pagina scrisa de mana")
    parser.add_argument("--perechi", type=Path, required=True,
                        help="folderul cu `.krn`-urile de la dataset.py")
    parser.add_argument("--portative", type=parse_grouping, required=True, metavar="N,N,...")
    parser.add_argument("--greutati", type=Path, default=None, help="implicit: modelul publicat")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--beam", type=int, default=1, help="ipoteze la decodare; 1 = lacoma")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from app.recognition.engine import Kind, PageImage
    from app.recognition.grandstaff_engine import GrandstaffEngine

    engine = GrandstaffEngine(
        staves_per_system=args.portative, clean=True, weights=args.greutati, threads=args.threads,
        beam=args.beam,
    )
    engine.load()
    result = engine.transcribe(PageImage(path=args.imagine, kind=Kind.HANDWRITTEN))
    counts = args.portative if isinstance(args.portative, list) else [args.portative] * len(result.staves)
    targets = sorted(args.perechi.glob(f"{args.imagine.stem}_sistem*.krn"))
    if not len(result.staves) == len(counts) == len(targets):
        # Un sistem scapat de detector ar scoate tinta lui din total si ar compara fiecare
        # sistem de dupa el cu tinta altuia (code review, 2026-10-04).
        raise SystemExit(
            f"{len(result.staves)} sisteme gasite, {len(counts)} declarate, {len(targets)} tinte "
            f"in {args.perechi} — nu se pot imperechea"
        )

    systems, total = [], {"note_tinta": 0, "scrise": 0, "corecte": 0, "distanta": 0, "tokeni": 0}
    for index, (staff, staves) in enumerate(zip(result.staves, counts), 1):
        target = (args.perechi / f"{args.imagine.stem}_sistem{index:02d}.krn").read_text(encoding="utf-8")
        row = compare_system(target, staff.kern or "", staves)
        row.update(sistem=index, portative=staves, ser=round(row["distanta"] / row["tokeni"], 3))
        systems.append(row)
        for key in total:
            total[key] += row[key]
        print(f"  sistem {index} ({staves} port.): {row['corecte']:3d}/{row['note_tinta']:3d} note pe "
              f"portativ, scrise {row['scrise']:3d}, SER {row['ser']:.3f}"
              + (f"  [{row['problema'][:60]}]" if row["problema"] else ""))

    ser = round(total["distanta"] / total["tokeni"], 3)
    print(f"TOTAL {args.imagine.stem}: {total['corecte']}/{total['note_tinta']} note pe portativ "
          f"({total['corecte'] / total['note_tinta']:.1%}), precizie "
          f"{total['corecte'] / max(1, total['scrise']):.1%}, SER {ser}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "imagine": str(args.imagine), "greutati": str(args.greutati), "sisteme": systems,
            "note_tinta": total["note_tinta"], "scrise": total["scrise"],
            "corecte": total["corecte"], "ser": ser,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
