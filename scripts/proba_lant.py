"""Proba lantului intreg, dintr-o comanda.

Testele acopera piesele; asta acopera **legaturile** — care nu se pot testa in
`pytest`, fiindca cer Smashcima si modelul, adica doua medii diferite.

Secventa e cea pe care o refaceam de mana la fiecare schimbare:

    foi de glife -> repository -> pagina sintetica -> poarta de tinta
                 -> perechi -> antrenare -> (optional) o cifra

    python scripts/proba_lant.py --glife data/glife --ref data/validation/pagina3.musicxml
    python scripts/proba_lant.py --glife models/glife.pkl --ref ... --masoara

Fara `--masoara` dureaza ~2 minute si nu atinge modelul de recunoastere decat
pentru antrenare. Cu ea, se adauga o masuratoare intreaga (minute bune), care
verifica si ultima veriga.

Se ruleaza in mediul cu Smashcima. Cand ceva se rupe, iese la primul pas care
cade, cu numarul pasului — nu la sfarsit, cu un mesaj de care nu stii de unde
vine.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def pas(numar: int, ce: str) -> None:
    print(f"\n[{numar}] {ce}")


def ruleaza(comanda: list[str], numar: int, ce: str) -> str:
    """Ruleaza un pas si se opreste la primul esec, spunand care a fost."""
    rezultat = subprocess.run(comanda, capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
    if rezultat.returncode:
        print(rezultat.stdout[-2000:])
        print(rezultat.stderr[-2000:])
        raise SystemExit(f"pasul {numar} ({ce}) a cazut cu codul {rezultat.returncode}")
    return rezultat.stdout


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--glife", type=Path, required=True,
                        help="folder cu subfoldere de om, SAU un repository .pkl gata facut")
    parser.add_argument("--ref", type=Path, required=True, help="referinta MusicXML")
    parser.add_argument("--masoara", action="store_true",
                        help="adauga si masuratoarea finala (minute bune)")
    parser.add_argument("--pastreaza", type=Path, default=None,
                        help="unde raman fisierele; implicit un folder temporar")
    parser.add_argument(
        "--python-model", type=Path, default=ROOT / ".venv" / "Scripts" / "python.exe",
        help="interpretorul cu torch; implicit `.venv` al proiectului",
    )
    args = parser.parse_args()

    lucru = args.pastreaza or Path(tempfile.mkdtemp(prefix="proba-lant-"))
    lucru.mkdir(parents=True, exist_ok=True)

    # **Lantul trece prin doua medii**, si scriptul trebuie s-o stie (§15.14):
    # Smashcima nu e in `.venv`-ul proiectului, fiindca `stringzilla` cere
    # compilator pe Python 3.14; iar torch nu e in mediul cu Smashcima. Deci
    # sinteza ruleaza cu interpretorul curent, antrenarea si masuratoarea cu
    # celalalt. A pretinde ca unul singur le face pe toate ar fi facut proba sa
    # cada exact acolo unde conducta e in regula.
    python = sys.executable
    python_model = str(args.python_model)
    if not Path(python_model).exists():
        raise SystemExit(
            f"nu exista interpretorul cu torch: {python_model}; "
            "da-l cu `--python-model`"
        )
    print(f"lucrez in {lucru}")
    print(
        f"  sinteza: {Path(python).parent.parent.name}"
        f" | model: {args.python_model.parent.parent.name}"
    )

    # 1. Repository — din foi, daca s-a dat un folder.
    if args.glife.is_dir():
        pas(1, "foi de glife -> repository (segmentare + capete de linie)")
        repository = lucru / "glife.pkl"
        iesire = ruleaza(
            [python, str(ROOT / "scripts" / "glife_repository.py"),
             "--glife", str(args.glife), "--out", str(repository)], 1, "repository")
        print(iesire.strip().splitlines()[-2])
    else:
        pas(1, f"repository gata facut: {args.glife.name}")
        repository = args.glife

    # 2. Verificarea referintei — inainte de a genera din ea.
    pas(2, "referinta: citire, rupturi, validator, dus-intors")
    iesire = ruleaza(
        [python, str(ROOT / "scripts" / "verifica_referinta.py"), str(args.ref)],
        2, "verifica_referinta")
    print(iesire.strip().splitlines()[-1])

    # 3. O pereche sintetica, cu poarta de tinta inauntru.
    pas(3, "pagina sintetica -> pereche de antrenare")
    perechi = lucru / "perechi"
    iesire = ruleaza(
        [python, str(ROOT / "scripts" / "sintetic.py"),
         "--ref", str(args.ref), "--glife", str(repository),
         "--out", str(perechi), "--per-stil", "1"], 3, "sintetic")
    print(iesire.strip().splitlines()[-1])

    date = perechi / "date.jsonl"
    if not date.exists():
        raise SystemExit("pasul 3 n-a produs `date.jsonl` — nicio pereche")

    # 4. Antrenare scurta: verifica tokenizarea, augmentarea si checkpointul.
    pas(4, "doi pasi de antrenare + checkpoint")
    model = lucru / "model"
    iesire = ruleaza(
        [python_model, str(ROOT / "scripts" / "finetune.py"),
         "--date", str(date), "--out", str(model), "--pasi", "2",
         "--checkpoint-la", "2"], 4, "finetune")
    print([l for l in iesire.splitlines() if "pasi in" in l][-1].strip())

    checkpoint = model / "checkpoint.pt"
    if not checkpoint.exists():
        raise SystemExit("pasul 4 n-a scris checkpointul")

    # 5. Masuratoarea — ultima veriga, si cea mai scumpa.
    if args.masoara:
        pas(5, "masuratoare cu greutatile antrenate")
        set_file = lucru / "set.json"
        set_file.write_text(
            '[{"imagine": "data/raw/scris-de-mana/pagina3.png", '
            '"referinta": "data/validation/pagina3.musicxml", '
            '"scriitor": "reper", "portative": 2}]', encoding="utf-8")
        iesire = ruleaza(
            [python_model, str(ROOT / "scripts" / "masoara.py"),
             "--set", str(set_file), "--greutati", str(checkpoint)], 5, "masoara")
        print([l for l in iesire.splitlines() if l.startswith("TOTAL")][-1])

    print("\nLANTUL E INTREG.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
