"""Se poate folosi referinta asta? Un raspuns, dintr-o singura rulare.

Cand vine un `.mscz` nou, intrebarea e mereu aceeasi, iar pana acum raspunsul
cerea patru comenzi si cititul cu ochiul. Scriptul le pune cap la cap. Nu scrie
nicio logica noua — cheama piesele care exista deja.

    python scripts/verifica_referinta.py data/validation/pagina3.musicxml

Ce verifica, in ordinea in care conteaza:

1. **Se citeste?** Notatia pe care v1 n-o reprezinta opreste totul aici.
2. **Rupturile de sistem si de pagina.** Daca sunt zero, n-au fost puse cu
   Ctrl+Enter in MuseScore, iar `dataset.py` nu va putea imperechea nimic
   (§15.7). E cea mai frecventa problema, si cea mai usor de reparat.
3. **Validatorul, pe referinta.** Nu e formalitate: §15.2 a gasit asa 21 de
   constatari reale **in referinta**, nu in transcriere — masuri de un timp
   raportate ca erori, care erau de fapt anacruze nemarcate.
4. **Dus-intorsul, pe fiecare sistem.** `Score` -> Humdrum -> inapoi ->
   `compare()`. Ce nu se inchide `exact` nu poate deveni tinta de antrenare.
5. **Vocabularul modelului**, daca se cere cu `--vocabular`. Incarca modelul,
   deci dureaza; fara el, primele patru raspund oricum la majoritatea
   intrebarilor.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from app.evaluation.metrics import compare  # noqa: E402
from app.evaluation.reference import load_with_layout  # noqa: E402
from app.music_model.model import measures_slice  # noqa: E402
from app.recognition.bekern import from_kern  # noqa: E402
from app.recognition.kern import parse_open_system  # noqa: E402
from app.recognition.kern_writer import to_kern_text  # noqa: E402
from app.validation.rules import Severity, validate  # noqa: E402
from scripts.dataset import system_bounds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("referinta", type=Path, help="fisierul MusicXML de verificat")
    parser.add_argument(
        "--vocabular", action="store_true",
        help="verifica si ca tokenii sunt in vocabularul modelului (incarca modelul)",
    )
    args = parser.parse_args()

    probleme = 0

    # 1. Se citeste?
    try:
        score, starts, pages = load_with_layout(args.referinta)
    except Exception as eroare:
        print(f"NU SE CITESTE: {type(eroare).__name__}: {eroare}")
        return 1

    bounds = system_bounds(starts, score.measure_count)
    print(f"{args.referinta.name}")
    print(f"  {score.measure_count} masuri, {len(score.parts)} voci, asezare {score.staves}")

    # 2. Rupturile
    print(f"  sisteme: {len(bounds)} {bounds}")
    print(f"  pagini incep la masurile: {pages}")
    if len(bounds) == 1 and score.measure_count > 8:
        probleme += 1
        print(
            "  PROBLEMA: un singur sistem pentru toata partitura. Rupturile de rand"
            " nu s-au exportat — pune-le in MuseScore cu Ctrl+Enter, acolo unde le"
            " are manuscrisul, si exporta din nou (§15.7)."
        )

    # 3. Validatorul, pe referinta
    findings = validate(score)
    erori = [f for f in findings if f.severity is Severity.ERROR]
    avertismente = [f for f in findings if f.severity is not Severity.ERROR]
    print(f"  validator: {len(erori)} erori, {len(avertismente)} avertismente")
    for finding in erori[:5]:
        print(f"    [error] {finding}")
    if len(erori) > 5:
        print(f"    ... si inca {len(erori) - 5}")
    if erori:
        probleme += 1
        print(
            "    Erorile astea sunt **in referinta**, nu intr-o transcriere. §15.2:"
            " masurile intentionat scurte se marcheaza in MuseScore, altfel se"
            " numara drept masuri stricate."
        )

    # 4. Dus-intorsul, pe fiecare sistem
    nereusite = []
    tinte = []
    for index, (first, last) in enumerate(bounds, 1):
        felie = measures_slice(score, first, last)
        try:
            text = to_kern_text(felie, ties=False)
        except Exception as eroare:
            nereusite.append(f"sistemul {index} (m{first}-{last}): nu se scrie — {eroare}")
            continue
        rezultat = compare(felie, parse_open_system(text, layout=felie.staves).score)
        if not rezultat.exact:
            nereusite.append(
                f"sistemul {index} (m{first}-{last}): dus-intorsul nu se inchide — "
                f"{rezultat.total.matched} din {rezultat.total.ref_events} note"
            )
        else:
            tinte.append((index, first, last, from_kern(text)))

    print(f"  dus-intors: {len(tinte)} din {len(bounds)} sisteme se inchid exact")
    for nereusita in nereusite[:5]:
        print(f"    {nereusita}")
    if nereusite:
        probleme += 1

    # 5. Vocabularul
    if args.vocabular and tinte:
        import torch

        from app.recognition.grandstaff_engine import load_grandstaff

        print("  se incarca modelul pentru vocabular...")
        model, _ = load_grandstaff(torch.device("cpu"))
        cazute = []
        for index, first, last, tokens in tinte:
            lipsa = sorted({t for t in tokens if t not in model.w2i})
            if lipsa:
                cazute.append(f"sistemul {index} (m{first}-{last}): {', '.join(lipsa[:6])}")
        print(f"  vocabular: {len(tinte) - len(cazute)} din {len(tinte)} sisteme trec")
        for cazuta in cazute[:5]:
            print(f"    {cazuta}")
        if cazute:
            probleme += 1

    print()
    if probleme:
        print(f"{probleme} lucruri de reparat inainte ca referinta sa poata fi folosita.")
    else:
        print(
            f"Se poate folosi: {len(tinte)} sisteme, "
            f"{sum(len(t[3]) for t in tinte)} tokeni in total."
        )
    return 1 if probleme else 0


if __name__ == "__main__":
    raise SystemExit(main())
