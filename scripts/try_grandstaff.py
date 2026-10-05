"""Compara `smt-grandstaff` cu `jazzmus-model` pe aceeasi pagina.

Ipoteza de verificat (PROJECT.md §3.9 si §12.4): pentru partituri **tiparite**
in partitura stransa, `smt-grandstaff` ar trebui sa fie superior, fiindca e
antrenat exact pe grand staff de pian tiparit — doua portative cu polifonie,
adica structural acelasi lucru cu S&A + T&B.

Diferenta care conteaza la intrare: grandstaff asteapta **un sistem intreg**
(doua portative intr-o singura imagine, inaltime 256), nu un portativ decupat
la 128 px ca jazzmus. Fiecare motor isi taie deci pagina in felul lui — de asta
decupajele se salveaza pe disc, si de asta sunt primul lucru de privit.

Scriptul nu contine logica de recunoastere: pune cele doua motoare pe aceeasi
pagina si masoara. Ce masoara e ce spune §9 — nu un „procent de acuratete"
global, pe care nu avem cum sa-l calculam fara ground truth pentru pagina asta.

    python scripts/try_grandstaff.py "C:\\Users\\...\\images.png"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

# Tokenii kern contin caractere pe care codificarea implicita a consolei
# Windows nu le poate scrie; fara asta, procesul moare fara niciun mesaj.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from app.recognition.engine import Kind, PageImage  # noqa: E402
from app.recognition.kern import parse_document, parse_open_system  # noqa: E402
from app.staff_detection.staves import parse_grouping, parse_layout  # noqa: E402


@dataclass
class Run:
    """Ce a iesit dintr-un motor, pregatit pentru tabel."""

    engine: str
    unit: str  # ce citeste odata: un portativ sau un sistem
    units: int = 0
    tokens: int = 0
    unrecognised: list[str] = field(default_factory=list)
    voices: int = 0
    measures: int = 0
    errors: int = 0
    warnings: int = 0
    seconds: float = 0.0
    load_seconds: float = 0.0
    problems: list[str] = field(default_factory=list)


# Ce nu stim sa citim din ce a scos modelul. Se calculeaza aici, nu in motor,
# fiindca cele doua modele scriu documente de forme diferite: grandstaff un
# sistem intreg cu toate vocile, jazzmus cate o coloana per portativ.
def _system_tokens(kern: str) -> list[str]:
    return parse_open_system(kern).unrecognised


def _staff_tokens(kern: str) -> list[str]:
    return [item for result in parse_document(kern) for item in result.unrecognised]


def unrecognised_tokens(parser, kern: str) -> list[str]:
    try:
        return parser(kern)
    except ValueError as error:
        return [f"(documentul nu s-a putut citi: {error})"]


def run_engine(engine, page: PageImage, out: Path, unit: str, name: str, parser) -> Run:
    import cv2

    from app.musicxml import export

    t0 = time.perf_counter()
    engine.load()
    load_seconds = time.perf_counter() - t0

    result = engine.transcribe(page)

    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    for staff in result.staves:
        cv2.imwrite(str(folder / f"{unit}{staff.index:02d}.png"), staff.image)
        (folder / f"{unit}{staff.index:02d}.krn").write_text(staff.kern, encoding="utf-8")

    run = Run(
        engine=name,
        unit=unit,
        units=len(result.staves),
        tokens=sum(staff.tokens for staff in result.staves),
        unrecognised=[
            item for staff in result.staves for item in unrecognised_tokens(parser, staff.kern)
        ],
        errors=len(result.errors),
        warnings=len(result.warnings),
        seconds=result.seconds,
        load_seconds=load_seconds,
        problems=list(result.problems),
    )

    if result.score is not None:
        run.voices = len(result.score.parts)
        run.measures = result.score.measure_count
        # MusicXML se scrie **si cand asamblarea a esuat**: o transcriere pe
        # care o poti privi in MuseScore bate un esec din care nu iese nimic.
        export.write(result.score, folder / f"{page.stem}.musicxml")

    return run


def report(runs: list[Run]) -> None:
    columns = [
        ("motor", lambda r: r.engine),
        ("unitate", lambda r: r.unit),
        ("bucati", lambda r: str(r.units)),
        ("tokeni", lambda r: str(r.tokens)),
        ("necunoscuti", lambda r: str(len(r.unrecognised))),
        ("voci", lambda r: str(r.voices)),
        ("masuri", lambda r: str(r.measures)),
        ("erori", lambda r: str(r.errors)),
        ("averti.", lambda r: str(r.warnings)),
        ("secunde", lambda r: f"{r.seconds:.1f}"),
    ]
    widths = [
        max(len(name), *(len(get(run)) for run in runs)) for name, get in columns
    ]

    print("\n" + "  ".join(name.ljust(w) for (name, _), w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for run in runs:
        print("  ".join(get(run).ljust(w) for (_, get), w in zip(columns, widths)))

    for run in runs:
        print(f"\n--- {run.engine} ---")
        print(f"incarcarea modelului: {run.load_seconds:.1f} s")
        if run.problems:
            for problem in run.problems:
                print(f"  ! {problem}")
        else:
            print("  fara probleme raportate")
        if run.unrecognised:
            unique = sorted(set(run.unrecognised))
            print(f"  tokeni necunoscuti ({len(run.unrecognised)}): "
                  + ", ".join(unique[:12]) + (" ..." if len(unique) > 12 else ""))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--only", choices=("jazzmus", "grandstaff"), default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    # Pe scris de mana cu versuri, spatierea nu spune unde se termina un sistem
    # (PROJECT.md §14.5). Atunci se declara, in loc sa se ghiceasca.
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
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data" / "processed" / "comparatie"
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"nu exista fisierul {args.image}")
        return 2

    from app.recognition.grandstaff_engine import GrandstaffEngine
    from app.recognition.smt_engine import SmtEngine

    # Tipul e explicit, ca peste tot in proiect: pagina de test e tiparita.
    page = PageImage(path=args.image, kind=Kind.PRINTED)
    args.out.mkdir(parents=True, exist_ok=True)

    # Numele din tabel e al modelului, nu al clasei: `SmtEngine.name` e "smt",
    # dar ce comparam e `jazzmus-model`.
    candidates = [
        ("jazzmus", lambda: SmtEngine(max_tokens=args.max_tokens), "portativ", _staff_tokens),
        ("grandstaff", lambda: GrandstaffEngine(
                max_tokens=args.max_tokens,
                staves_per_system=args.portative_pe_sistem,
                layout=args.asezare,
                clean=args.curata,
            ), "sistem", _system_tokens),
    ]

    runs = []
    for name, build, unit, parser in candidates:
        if args.only and args.only != name:
            continue
        print(f"\n=== {name} ===")
        runs.append(run_engine(build(), page, args.out, unit, name, parser))

    report(runs)
    print(f"\niesiri in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
