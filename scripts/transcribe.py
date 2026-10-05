"""Transcrie una sau cateva pagini, cu tot ce s-a intamplat afisat pe ecran.

Pentru teancuri foloseste `scripts/batch.py` — acesta e pentru depanare: cand
vrei sa vezi imediat ce a citit modelul dintr-o poza anume.

Nu contine logica proprie de recunoastere. Toata e in `app/recognition/`, in
spatele interfetei `Engine`, ca sa nu ajungem cu doua implementari ale aceleiasi
conducte care se despart pe nesimtite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from app.batch import inputs  # noqa: E402
from app.batch.runner import write_outputs  # noqa: E402
from app.recognition.engine import Kind, PageImage  # noqa: E402
from app.recognition.smt_engine import engine_for  # noqa: E402
from app.staff_detection.staves import parse_grouping, parse_layout  # noqa: E402


def report(result) -> None:
    print(f"\n{result.page}: {len(result.staves)} portative in {result.seconds:.1f}s")
    for staff in result.staves:
        print(f"  portativ {staff.index}: {staff.tokens:>3} tokeni in {staff.seconds:5.1f}s")

    if result.score is None:
        print("  asamblare esuata:")
        for problem in result.problems:
            print(f"    {problem}")
        print("  (`**kern`-ul brut si decupajele sunt scrise — ele spun de ce)")
        return

    counts = {part.id: len(part.measures) for part in result.score.parts}
    print(f"  {result.systems} sisteme, masuri per voce: "
          + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  validare: {len(result.errors)} erori, {len(result.warnings)} avertismente")
    for line in (result.errors + result.warnings)[:10]:
        print(f"    {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--tip", choices=[k.value for k in Kind], default=Kind.HANDWRITTEN.value)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=None)
    # Pe scris de mana cu versuri, spatierea nu spune unde se termina un
    # sistem — masurat in PROJECT.md §14.6. Atunci se declara, in loc sa se
    # ghiceasca; fara optiune, gruparea se deduce ca pana acum.
    parser.add_argument(
        "--portative-pe-sistem", type=parse_grouping, default=None,
        metavar="N|N,N,...",
        help="cate portative are un sistem: un numar (`2`) sau cate unul pe "
             "sistem (`2,2,2,3,3`)",
    )
    parser.add_argument(
        "--asezare", type=parse_layout, default=None, metavar="S,A/T,B",
        help="ce voci stau pe fiecare portativ, de sus in jos; pentru o cantare "
             "cu solo pe portativ separat: `X1/S,A/T,B`",
    )
    # Etapa 6: indreapta pagina si sterge ce nu atinge un portativ — versuri,
    # titlu, numere de masura. Implicit oprita: pe tiparit curat n-are ce
    # repara, iar un pas care sterge cerneala se cere, nu se presupune.
    parser.add_argument(
        "--greutati", type=Path, default=None, metavar="CHECKPOINT",
        help="checkpoint de la `scripts/finetune.py`, aplicat peste greutatile publicate",
    )
    parser.add_argument("--curata", action="store_true")
    args = parser.parse_args()

    kind = Kind(args.tip)
    engine = engine_for(
        kind,
        threads=args.threads,
        max_tokens=args.max_tokens,
        staves_per_system=args.portative_pe_sistem,
        layout=args.asezare,
        weights=args.greutati,
        clean=args.curata,
    )

    # Fara coada si fara baza de date: depanarea reproceseaza de fiecare data,
    # chiar daca fisierul a mai fost transcris. Asta e diferenta fata de lot.
    for page in inputs.collect(list(args.images), kind):
        resolved = inputs.resolve(page, args.out)
        result = engine.transcribe(PageImage(path=resolved, kind=page.kind, page=0))
        write_outputs(page, result, args.out)
        report(result)

    print(f"\niesiri in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
