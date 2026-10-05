"""Conversie in lot: un teanc de partituri -> MusicXML + raport.

    python scripts/batch.py data/raw --tip handwritten
    python scripts/batch.py --reia            # continua lotul intrerupt
    python scripts/batch.py --raport          # regenereaza raportul din baza

Lotul e reluabil: daca opresti procesul (Ctrl+C sau inchizi laptopul), o noua
rulare cu `--reia` continua de unde a ramas. Fisierele deja terminate se sar,
in afara de cazul in care s-au schimbat pe disc.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Consola Windows nu e pe UTF-8 implicit, iar mesajele lotului contin diacritice.
# Raportul HTML e oricum UTF-8; asta e doar pentru ce se scrie in terminal.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from app.batch import report  # noqa: E402
from app.batch.runner import Progress, Runner  # noqa: E402
from app.batch.store import State, Store  # noqa: E402
from app.recognition.engine import Kind  # noqa: E402
from app.recognition.smt_engine import engine_for  # noqa: E402
from app.staff_detection.staves import parse_grouping, parse_layout  # noqa: E402


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f}m"


def _describe(result) -> str:
    """Ce s-a intamplat cu o pagina, in cateva cuvinte.

    Distinctia care conteaza: "0 erori" pe o pagina din care nu a iesit nicio
    partitura ar suna a succes, desi e opusul.
    """
    if result is None:
        return "ESUAT"
    staves = f"{len(result.staves)} portative"
    if result.score is None:
        return f"{staves}, ASAMBLARE ESUATA"
    if result.errors:
        return f"{staves}, {len(result.errors)} erori"
    if result.warnings:
        return f"{staves}, curat ({len(result.warnings)} avertismente)"
    return f"{staves}, curat"


def _on_progress(progress: Progress) -> None:
    print(
        f"[{progress.done}/{progress.total}] {progress.current}: "
        f"{_describe(progress.last)} "
        f"- ramas ~{_format_duration(progress.seconds_remaining)}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("surse", nargs="*", type=Path,
                        help="fisiere sau foldere de procesat (recursiv)")
    parser.add_argument("--tip", choices=[k.value for k in Kind], default=Kind.HANDWRITTEN.value,
                        help="tipul partiturilor din surse; se declara explicit, nu se ghiceste")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--db", type=Path, default=None,
                        help="baza de date a lotului (implicit <out>/batch.db)")
    parser.add_argument("--reia", action="store_true",
                        help="continua lotul existent, fara a adauga fisiere noi")
    parser.add_argument("--raport", action="store_true",
                        help="doar regenereaza raportul, fara sa proceseze nimic")
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
    parser.add_argument("--curata", action="store_true")
    args = parser.parse_args()

    if not args.surse and not (args.reia or args.raport):
        parser.error("da cel putin o sursa, sau foloseste --reia / --raport")

    out = args.out
    store = Store(args.db or out / "batch.db")
    report_path = out / "raport.html"

    if args.raport:
        jobs = store.all()
        report.write(jobs, report_path)
        print(f"raport regenerat din {len(jobs)} intrari: {report_path}")
        store.close()
        return 0

    engine = engine_for(
        Kind(args.tip),
        threads=args.threads,
        max_tokens=args.max_tokens,
        staves_per_system=args.portative_pe_sistem,
        layout=args.asezare,
        clean=args.curata,
    )
    runner = Runner(store=store, engine=engine, out_dir=out)

    if args.surse:
        remaining = runner.enqueue(list(args.surse), Kind(args.tip))
    else:
        remaining = len(store.pending())

    if not remaining:
        print("nimic de procesat — totul e deja terminat")
        report.write(store.all(), report_path)
        store.close()
        return 0

    print(f"{remaining} unitati de lucru · iesire in {out}")

    # Ctrl+C opreste curat dupa fisierul curent, ca sa nu ramana un job
    # `running` in baza si sa nu pierdem munca facuta pana atunci.
    cancel = threading.Event()

    def _stop(signum, frame):  # noqa: ARG001
        if not cancel.is_set():
            print("\nse opreste dupa fisierul curent… (inca un Ctrl+C forteaza)", flush=True)
            cancel.set()
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _stop)

    started = time.perf_counter()
    jobs = runner.run(progress=_on_progress, cancel=cancel)
    elapsed = time.perf_counter() - started

    counts = store.counts()
    report.write(jobs, report_path)
    store.close()

    print(f"\ngata in {_format_duration(elapsed)}: "
          + ", ".join(f"{n} {state}" for state, n in sorted(counts.items())))
    print(f"raport: {report_path}")
    if counts.get(State.PENDING.value):
        print("lot incomplet — reia cu: python scripts/batch.py --reia")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
