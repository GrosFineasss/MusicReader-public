"""Pune un checkpoint masurat in locul din care il ia aplicatia (`models/produs/greutati.pt`).

    python scripts/publica_greutati.py models/pdmx-repetitii/pas11000.pt --nota "pagina3 tiparita 484/624"

Din checkpointul de antrenare se pastreaza **doar greutatile modelului**: starea optimizatorului
(Adam, de doua ori marimea modelului) si a generatoarelor aleatoare nu-i trebuie aplicatiei. Pe
`--mod tot`, fisierul scade de la ~257 MB la ~86 MB.

Se scrie si de unde vin greutatile (`sursa`, `step`, `nota`), ca o transcriere sa poata fi pusa
langa masuratoarea care le-a ales. Greutatile antrenate pe coralele Bach (§15.59) **nu** se publica:
licenta editiei e neclara pentru uz comercial — scriptul refuza un checkpoint din `models/bach-*`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.recognition.smt_engine import PRODUS, PRODUS_MANA  # noqa: E402


# Datele cu licenta necomerciala, dupa numele folderului din care vine checkpointul.
NECOMERCIALE = {
    "bach": "coralele Bach din music21 (editia Greentree, licenta neclara pentru uz comercial, §15.59)",
    "musicorpus": "MusiCorpus v1.0 (LINDAT, CC BY-NC-SA 4.0)",
}


def licente_necomerciale(sursa: Path) -> list[str]:
    """Ce date necomerciale au intrat in checkpoint, dupa folderele de pe calea lui rezolvata.

    Pe calea **rezolvata** si pe fiecare folder al ei: `pas2000.pt` dat din `models/bach-capat`
    sau `models/bach-capat/sub/x.pt` ar fi trecut de o verificare pe parintele scris.
    """
    gasite = []
    for folder in sursa.resolve().parents:
        for prefix, licenta in NECOMERCIALE.items():
            if folder.name.startswith(prefix) and licenta not in gasite:
                gasite.append(licenta)
    return gasite


def publica(
    sursa: Path, destinatie: Path = PRODUS, nota: str = "", necomercial: bool = False
) -> dict:
    """Copiaza greutatile modelului din `sursa` in `destinatie`; intoarce ce s-a scris.

    Greutatile antrenate pe date necomerciale se publica doar cu `necomercial=True`. Aplicatia e
    pentru uz personal (Fineas, 2026-10-05: „nu vreau sa comercializez acest produs, e doar pentru
    mine"), deci e voie — dar trebuie spus explicit, iar licentele se scriu in fisier.
    """
    import torch

    licente = licente_necomerciale(sursa)
    if licente and not necomercial:
        raise ValueError(
            f"{sursa}: antrenat pe date necomerciale — {'; '.join(licente)}. "
            "Se publica doar pentru uz necomercial: da `necomercial=True` (`--necomercial`)"
        )
    state = torch.load(sursa, map_location="cpu", weights_only=False)
    model = state["model"] if "model" in state else state
    publicat = {
        "model": model,
        "step": state.get("step"),
        "sursa": sursa.as_posix(),
        "nota": nota,
        "licente_necomerciale": licente,
        "data": date.today().isoformat(),
    }
    destinatie.parent.mkdir(parents=True, exist_ok=True)
    temporar = destinatie.with_suffix(".tmp")
    torch.save(publicat, temporar)
    temporar.replace(destinatie)
    return publicat


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path, help="checkpointul masurat, de la finetune.py")
    parser.add_argument("--nota", default="", help="de ce acesta: masuratoarea care l-a ales")
    parser.add_argument(
        "--mana", action="store_true",
        help=f"greutatile pentru scris de mana (`{PRODUS_MANA.name}`), nu pentru tiparit",
    )
    parser.add_argument("--out", type=Path, default=None, help="implicit: locul din aplicatie")
    parser.add_argument(
        "--necomercial", action="store_true",
        help="permite greutati antrenate pe date necomerciale (Bach, MusiCorpus): aplicatia e pentru "
             "uz personal; licentele se scriu in fisierul publicat",
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = PRODUS_MANA if args.mana else PRODUS
    publicat = publica(args.checkpoint, args.out, args.nota, args.necomercial)
    marime = args.out.stat().st_size / 1e6
    print(
        f"{args.out}: {len(publicat['model'])} tensori, pasul {publicat['step']}, {marime:.0f} MB "
        f"(din {args.checkpoint.stat().st_size / 1e6:.0f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
