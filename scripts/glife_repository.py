"""Foi de glife completate -> `MungSymbolRepository`, cate un stil de om.

Ultimul pas dintre oameni si paginile sintetice (PROJECT.md §15.15). Se asteapta
un folder cu **cate un subfolder de persoana**, iar numele subfolderului devine
numele stilului:

    data/glife/
      harta.json          <- scris de `scripts/foaie_glife.py`
      ana/  pagina-1.png  pagina-2.png
      radu/ pagina-1.png  pagina-2.png

    python scripts/glife_repository.py --glife data/glife --out models/glife.pkl

Convenția pe foldere, si nu o optiune repetata in linia de comanda, fiindca la
10-15 oameni a doua varianta devine o comanda pe care n-o mai scrie nimeni
corect. Numele folderului e si eticheta de stil: nu se cere nicaieri sa fie un
nume adevarat, cum nu se cere nici la `--scriitor` din `dataset.py`.

Ce se raporteaza la final e **acoperirea**: cate exemplare are fiecare simbol,
pentru fiecare om. Un simbol cu zero exemplare inseamna ca stilul acela nu-l
poate desena, iar Smashcima va cadea pe sintetizatorul de rezerva — bine de
stiut inainte de a genera mii de pagini, nu dupa.

Ruleaza in mediul in care e instalat Smashcima, nu in `.venv`-ul proiectului
(§15.14 explica de ce nu e acolo).
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

from app.training.glife import coverage, load_plan, segment  # noqa: E402


def raporteaza_lipsurile(acoperiri: dict, simboluri: list[str]) -> None:
    """Ce lipseste, si **ce se intampla din cauza asta** — doua lucruri diferite.

    Smashcima imprumuta intre maini de la sine: `RepositoryGlyphSynthesizer.pick`
    cade pe `glyphs_by_label` cand stilul curent n-are eticheta. Deci o casuta
    sarita de un om se acopera din scrisul altuia, **tacut**.

    Tacerea aia conteaza. Daca cineva a sarit 10 din 36 de simboluri, stilul lui
    e pe sfert scrisul altcuiva — iar validarea pe mana tinuta deoparte e mai
    putin curata decat pare. Raportul o spune, chiar daca nu schimba nimic.

    Ce lipseste de la **toti** e altceva: acolo nu are cine imprumuta, iar
    generarea se opreste la primul sistem care cere simbolul (§15.21).
    """
    if not acoperiri:
        return

    peste_tot = {
        label: sum(a.get(label, 0) for a in acoperiri.values()) for label in simboluri
    }
    nicaieri = [label for label, n in peste_tot.items() if n == 0]

    for nume, acoperire in acoperiri.items():
        imprumutate = [
            label for label in simboluri
            if acoperire.get(label, 0) == 0 and peste_tot[label] > 0
        ]
        if imprumutate:
            print(
                f"      {nume} n-are {len(imprumutate)} simboluri; se vor desena cu "
                f"scrisul altora: {', '.join(imprumutate[:8])}"
                + (" ..." if len(imprumutate) > 8 else "")
            )

    if nicaieri:
        print()
        print(f"  {len(nicaieri)} simboluri lipsesc de la TOTI: {', '.join(nicaieri)}")
        print("  N-are cine sa le imprumute, deci generarea se va opri la primul")
        print("  sistem care le cere. Completeaza casutele si ruleaza din nou.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--glife", type=Path, required=True, help="folderul cu subfoldere de om")
    parser.add_argument("--out", type=Path, required=True, help="unde se scrie repository-ul")
    parser.add_argument(
        "--harta", type=Path, action="append", default=[],
        help="harta casutelor; se poate da de mai multe ori. Implicit se iau "
             "toate `*.json` de langa folderele de oameni, plus, pentru fiecare "
             "om, cele din folderul lui — care au intaietate",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # Hartile comune: cele date explicit, altfel orice `.json` de langa foldere.
    comune = [load_plan(h) for h in args.harta] if args.harta else [
        load_plan(h) for h in sorted(args.glife.glob("*.json"))
    ]
    if not comune:
        raise SystemExit(f"nicio harta in {args.glife} — da-o cu `--harta`")

    oameni = sorted(p for p in args.glife.iterdir() if p.is_dir())
    if not oameni:
        raise SystemExit(f"niciun subfolder de persoana in {args.glife}")

    toate = []
    acoperiri: dict[str, dict[str, int]] = {}
    simboluri: list[str] = []
    for folder in oameni:
        # **Harta proprie a omului trece inaintea celei comune.** Cine a
        # completat o foaie mai veche o pastreaza: asezarea se muta cand un
        # simbol capata casuta de exemplu, deci o foaie tiparita atunci nu se
        # mai decupeaza cu harta de azi. Isi tine harta ei in folderul lui.
        harti = [load_plan(h) for h in sorted(folder.glob("*.json"))] + comune
        for plan in harti:
            for label in plan["symbols"]:
                if label not in simboluri:
                    simboluri.append(label)

        # Potrivirea se face pe **numele fisierului**, scris in harta la
        # generare. Pe pozitia in folder, un fisier in plus sau o pagina
        # fotografiata in alta ordine muta toate casutele fara sa spuna nimic —
        # si chiar asa a venit primul PDF, cu paginile in ordinea 2, 3, 1.
        dupa_nume = {}
        for plan in harti:
            for page in plan["pages"]:
                dupa_nume.setdefault(page.get("image"), page)

        scrise = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
        if not scrise:
            print(f"  {folder.name}: nicio imagine, sarit")
            continue

        crops, sarite = [], []
        for imagine in scrise:
            page = dupa_nume.get(imagine.name)
            if page is None:
                stiute = ", ".join(sorted(n for n in dupa_nume if n))
                print(f"  {folder.name}: {imagine.name} — nu e in nicio harta, sarit"
                      f" (hartile stiu: {stiute})")
                continue
            bucata, lipsa = segment(imagine, page, writer=folder.name)
            crops.extend(bucata)
            sarite.extend(lipsa)

        print(
            f"  {folder.name}: {len(crops)} glife din {len(scrise)} pagini"
            + (f", {len(sarite)} casute sarite" if sarite else "")
        )
        acoperiri[folder.name] = coverage(crops, simboluri)
        toate.extend(crops)

    raporteaza_lipsurile(acoperiri, simboluri)

    if not toate:
        raise SystemExit("nicio glifa — verifica foile si harta")

    from app.training.repository import build_repository, save

    repository = build_repository(toate, dpi=args.dpi)
    path = save(repository, args.out)
    stiluri = sorted(repository.get_all_styles())
    etichete = sorted(repository.get_all_glyph_labels())
    print(
        f"\n{len(toate)} glife | {len(stiluri)} stiluri ({', '.join(stiluri)}) "
        f"| {len(etichete)} simboluri"
    )
    print(f"scris in {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
