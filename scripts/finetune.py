"""Etapa 11: antreneaza `smt-grandstaff` pe paginile tale.

Perechile vin din `scripts/dataset.py`. Implicit se antreneaza **capatul**
(ultimul strat de decoder plus proiectia de vocabular, 6,0 M de parametri), la
**3,3 s/pas** remasurat pe 2026-09-01 — deci 500 de pasi in ~28 de minute.
`--mod tot` costa 6,9 s/pas, adica 57 de minute; nu mai e exclus, cum spunea
§8.1 cu cifrele ei de ~20x mai pesimiste.

    python scripts/finetune.py --date data/antrenare/date.jsonl --out models/capat --pasi 500
    python scripts/finetune.py --date data/antrenare/date.jsonl --out models/capat --pasi 500
    # aceeasi comanda a doua oara continua din checkpoint

**Rularea se reia singura.** Daca laptopul se inchide, a doua pornire continua
din ultimul checkpoint; `--pasi` e numarul total, nu cati mai raman. Ca sa
porneasca de la zero, sterge folderul sau da `--de-la-capat`.

Cand s-a terminat, cifra care conteaza se ia rerulland masuratoarea cu
greutatile noi — comanda se tipareste la final. Reperul e cel inghetat din
`data/validation/masuratoare-pagina3-curat.json`: 16 note corecte din 182,
SER 1,643.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from app.training.scurgere import (  # noqa: E402
    masoara as masoara_scurgerea,
    piesa_din_imagine,
    piesa_din_referinta,
    raport as raport_scurgere,
)
from app.training.data import (  # noqa: E402
    describe_writers,
    load_pairs,
    usable_pairs,
    split_by_fraction,
    split_by_writer,
    writers,
)
from app.training.loop import (  # noqa: E402
    CHECKPOINT_EVERY,
    LEARNING_RATE,
    checkpoint_path,
    finetune,
    history_path,
    scor_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=Path, required=True, help="`date.jsonl` de la dataset.py")
    parser.add_argument("--out", type=Path, required=True, help="unde se scriu checkpoint-urile")
    parser.add_argument("--pasi", type=int, default=500, help="numarul TOTAL de pasi")
    parser.add_argument(
        "--mod", choices=("capat", "decoder", "tot"), default="capat",
        help="ce se antreneaza; toate trei sunt fezabile de la remasurarea din "
             "§15.11, iar `capat` e implicit fiindca pe cateva zeci de pagini "
             "6 M de parametri au mai putin loc sa se supraantreneze decat 21 M",
    )
    parser.add_argument("--rata", type=float, default=LEARNING_RATE)
    parser.add_argument("--checkpoint-la", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--seminte", type=int, default=0)
    parser.add_argument(
        "--de-la-capat", action="store_true",
        help="ignora checkpointul existent si porneste de la greutatile publicate",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--evalueaza-la", type=int, default=None, metavar="N",
        help="masoara pe pagina de referinta la fiecare N pasi (multiplu de "
             "`--checkpoint-la`). O masuratoare costa minute, deci pe o rulare "
             "lunga se alege rar — dar da o CURBA, nu un punct",
    )
    parser.add_argument(
        "--ref-evaluare", type=Path, default=Path("data/validation/pagina3.musicxml"),
    )
    parser.add_argument(
        "--imagine-evaluare", type=Path,
        default=Path("data/raw/scris-de-mana/pagina3.png"),
    )
    validare = parser.add_mutually_exclusive_group()
    validare.add_argument(
        "--scriitor-validare", default=None, metavar="ID[,ID...]",
        help="ce maini se tin deoparte, ca sa se poata masura pe scris nevazut; "
             "fara ele cifra de dupa nu spune daca modelul citeste sau isi aminteste",
    )
    validare.add_argument(
        "--procent-validare", type=int, default=None, metavar="N",
        help="tine deoparte N%% din **maini** (nu din pagini), ales stabil. "
             "Varianta potrivita cand fiecare pagina e alt scris si eticheta e "
             "identificatorul paginii",
    )
    parser.add_argument(
        "--tine-deoparte", nargs="+", default=[], metavar="PIESA",
        help="scoate din corpus perechile randate din piesele astea, cu tot cu "
             "transpozitiile lor. Piesa masurata (`--ref-evaluare`) se scoate din "
             "oficiu cand se da optiunea",
    )
    parser.add_argument(
        "--accept-scurgere", action="store_true",
        help="porneste chiar daca piesa masurata e in datele de antrenare. Uneori "
             "vrei asta dinadins; niciodata tacut",
    )
    parser.add_argument(
        "--fara-augmentare", action="store_true",
        help="opreste augmentarea; implicit e pornita, fiindca eroziunea si "
             "distorsiunea elastica apropie modelul de scrisuri nevazute",
    )
    args = parser.parse_args()

    if args.evalueaza_la and args.evalueaza_la % args.checkpoint_la:
        raise SystemExit(
            f"`--evalueaza-la {args.evalueaza_la}` trebuie sa fie multiplu de "
            f"`--checkpoint-la {args.checkpoint_la}`: masuratoarea citeste "
            "checkpointul de pe disc, deci se face doar cand exista unul proaspat"
        )

    pairs = load_pairs(args.date)

    # §15.41: din 299 de perechi, 69 erau chiar pagina pe care se masoara — si
    # nimeni n-a hotarat asta. Optiunile de validare despart **maini**, nu piese,
    # deci cu o singura mana n-aveau ce desparti. Poarta asta face separarea
    # ceruta explicit: ori se scoate piesa, ori se accepta scurgerea in scris.
    masurata = piesa_din_referinta(args.ref_evaluare)
    cerute = [*args.tine_deoparte, *([masurata] if args.tine_deoparte else [])]
    de_scos = list(dict.fromkeys(cerute))
    if de_scos:
        inainte = len(pairs)
        pairs = [p for p in pairs if piesa_din_imagine(p.image.name) not in set(de_scos)]
        print(f"tinute deoparte: {inainte - len(pairs)} perechi din {', '.join(sorted(de_scos))}")
    scurgere = masoara_scurgerea([p.image.name for p in pairs], [masurata])
    if not scurgere.curat:
        if not args.accept_scurgere:
            iesiri = (
                f"Scoate-o cu `--tine-deoparte {masurata}`, masoara pe alta pagina "
                "cu `--ref-evaluare`, sau accepta scurgerea in scris cu "
                "`--accept-scurgere`."
            )
            raise SystemExit(raport_scurgere(scurgere, [masurata]) + "\n\n" + iesiri)
        print("  ATENTIE: " + raport_scurgere(scurgere, [masurata]).replace("\n", "\n  "))

    print(f"scriitori: {describe_writers(pairs)}")

    validation = []
    if args.scriitor_validare:
        wanted = {name.strip() for name in args.scriitor_validare.split(",") if name.strip()}
        pairs, validation = split_by_writer(pairs, wanted)
    elif args.procent_validare:
        pairs, validation = split_by_fraction(pairs, args.procent_validare / 100, args.seminte)
    elif len(writers(pairs)) > 1:
        print(
            "  ATENTIE: se antreneaza pe toate mainile. Cifra de dupa va fi "
            "optimista — da `--scriitor-validare` sau `--procent-validare` ca sa "
            "masori pe scris nevazut."
        )

    if validation:
        tinute = sorted(writers(validation))
        cate = f"{len(tinute)} maini" if len(tinute) > 8 else ", ".join(tinute)
        print(
            f"tinut deoparte ({cate}): {len(validation)} perechi de validare, "
            f"{len(pairs)} de antrenare"
        )
    lungimi = sorted(len(pair.tokens) for pair in pairs)
    print(
        f"{len(pairs)} perechi | tokeni: mediana {lungimi[len(lungimi) // 2]}, "
        f"cel mai lung {lungimi[-1]}"
    )
    if len(pairs) < 10:
        print(
            f"  ATENTIE: {len(pairs)} perechi e putin. §15.5 spune ca decalajul se "
            "inchide cu paginile tale, iar cateva zeci sunt minimul rezonabil."
        )

    import torch

    from app.recognition.grandstaff_engine import load_grandstaff

    torch.set_num_threads(args.threads)
    print("se incarca modelul...")
    model, config = load_grandstaff(torch.device("cpu"))

    # Verificarea vocabularului se face **acum**, nu la nimereala in timpul
    # antrenarii: perechile se aleg aleator, deci un token necunoscut intr-una
    # singura ar opri rularea dupa ore, cu un mesaj care n-ar spune ca problema
    # era acolo de la inceput.
    pairs, refuzate = usable_pairs(pairs, model.w2i)
    if refuzate:
        print(f"  {len(refuzate)} perechi nu se pot folosi:")
        for pair, motiv in refuzate[:5]:
            print(f"    {pair.image.name} ({pair.writer}): {motiv}")
        if len(refuzate) > 5:
            print(f"    ... si inca {len(refuzate) - 5}")
    if not pairs:
        raise SystemExit(
            "nicio pereche nu se poate folosi — vezi motivele de mai sus"
        )

    existing = checkpoint_path(args.out)
    if existing.exists() and not args.de_la_capat:
        state = torch.load(existing, map_location="cpu", weights_only=False)
        print(f"se reia din {existing}, pasul {state['step']}")

    started = time.perf_counter()

    # Cel mai bun checkpoint, nu doar ultimul. Bucla tine un singur fisier si
    # il suprascrie, deci punctul cel mai bun al unei rulari nu mai putea fi
    # reexaminat — nici masurat altfel, nici pus pe alta pagina (§15.31).
    cel_mai_bun = {"scor": (-1, 0), "step": 0}

    def masoara(progress, cale) -> None:
        """Cifra la checkpoint, pe **acelasi drum** ca masuratoarea finala.

        Se cheama `transcribe` din `scripts/evaluate.py`, nu se scrie o a doua
        masuratoare: doua drumuri care ar putea sa nu fie de acord ar face
        curba de neinterpretat.
        """
        if not args.evalueaza_la or progress.step % args.evalueaza_la:
            return
        if not (args.ref_evaluare.exists() and args.imagine_evaluare.exists()):
            return

        from app.evaluation.metrics import compare
        from app.evaluation.reference import load
        from scripts.evaluate import transcribe

        ipoteza = transcribe(
            args.imagine_evaluare, "handwritten", None,
            staves_per_system=2, clean=True, weights=cale,
        )
        rezultat = compare(load(args.ref_evaluare), ipoteza)
        total = rezultat.total
        # `ser` e `Fraction` — metrica il tine exact (§11.4). Pentru afisare si
        # pentru JSON se converteste o singura data, aici.
        ser = float(total.ser)
        # `scrise` sta langa `corecte` fiindca SER singur nu deosebeste un model
        # care citeste mai bine de unul care a amutit: un model care nu scrie
        # nimic iese fix la SER 1,0, iar unul care scrie 14 note din 182 iese
        # sub 1 fara sa fi citit nimic (§15.28).
        print(
            f"    masurat la pasul {progress.step}: "
            f"{total.matched_notes}/{total.ref_notes} note "
            f"(+{total.matched_rests} pauze potrivite), "
            f"{total.hyp_events} scrise ({total.extra} in plus), SER {ser:.3f}"
        )
        scor = scor_checkpoint(total.matched_notes, total.hyp_events)
        if scor > cel_mai_bun["scor"]:
            cel_mai_bun.update(scor=scor, step=progress.step)
            shutil.copy2(cale, cale.with_name("checkpoint-cel-mai-bun.pt"))
            print("      pastrat ca cel mai bun de pana acum")

        with history_path(args.out).open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({
                    "step": progress.step,
                    "corecte": total.matched,
                    "corecte_note": total.matched_notes,
                    "corecte_pauze": total.matched_rests,
                    "note_referinta": total.ref_events,
                    "note_referinta_adevarate": total.ref_notes,
                    "note_transcriere": total.hyp_events,
                    "in_plus": total.extra,
                    "lipsa": total.missing,
                    "ser": ser,
                }) + chr(10)
            )

    def report(progress) -> None:
        if progress.step % 5 and progress.step != progress.steps:
            return
        print(
            f"  pas {progress.step}/{progress.steps} | pierdere {progress.loss:.4f} "
            f"| {progress.seconds_per_step:.1f} s/pas | ~{progress.remaining_hours:.1f} h ramase"
        )

    progress = finetune(
        model,
        pairs,
        steps=args.pasi,
        out_dir=args.out,
        mode=args.mod,
        maxh=int(config.maxh),
        maxw=int(config.maxw),
        learning_rate=args.rata,
        checkpoint_every=args.checkpoint_la,
        seed=args.seminte,
        resume=not args.de_la_capat,
        augment=not args.fara_augmentare,
        on_step=report,
        on_checkpoint=masoara,
    )

    minutes = (time.perf_counter() - started) / 60
    first = progress.history[0] if progress.history else float("nan")
    last = progress.history[-1] if progress.history else float("nan")
    print(
        f"\n{progress.step} pasi in {minutes:.1f} min | pierdere {first:.4f} -> {last:.4f}"
    )
    print(f"checkpoint: {checkpoint_path(args.out)}")
    if cel_mai_bun["step"]:
        cale_buna = checkpoint_path(args.out).with_name("checkpoint-cel-mai-bun.pt")
        corecte, minus_scrise = cel_mai_bun["scor"]
        print(f"cel mai bun: {cale_buna} — pasul {cel_mai_bun['step']}, "
              f"{corecte} note corecte, {-minus_scrise} scrise")
    print(
        "\nCifra se ia rerulland masuratoarea cu greutatile noi:\n"
        f"  python scripts/evaluate.py --ref data/validation/pagina3.musicxml \\\n"
        f"      --imagine data/raw/scris-de-mana/pagina3.png --portative-pe-sistem 2 --curata \\\n"
        f"      --greutati {checkpoint_path(args.out)} --json data/processed/dupa-antrenare.json\n"
        "Reper: 16 note corecte din 182, SER 1,643 "
        "(data/validation/masuratoare-pagina3-curat.json)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
