"""Etapa 11: cat costa un pas de antrenare pe *acest* CPU.

Fratele lui `benchmark_smt.py`, care masoara inferenta. Intrebarea nu e „merita
fine-tuning?" — asta o decide acuratetea — ci „e fezabil sa antrenam pe un
laptop fara CUDA, si in care configuratie?".

PROJECT.md §8.1 spune despre fine-tuning pe CPU „fezabil, dar lent — ore pana
la zile", si recunoaste imediat: **nu am masurat nimic, toate estimarile sunt
rationamente pe baza marimii modelului**. Scriptul asta inlocuieste
rationamentul cu cifre, si de el atarna daca mai are rost sa construim conducta
de date sintetice sau intra planul B din §8.3.

Se masoara un **pas de antrenare adevarat**, nu o trecere inainte. Din
`vendor/SMT/smt_trainer.py:49`, un pas e:

    outputs = model(encoder_input=x, decoder_input=di, labels=y)
    outputs.loss.backward()
    optimizer.step()

Deci forward + pierdere + backward + optimizator, cu modelul in `train()`.
Backward-ul si Adam sunt de obicei de doua-trei ori mai scumpe decat forward-ul;
o masuratoare care le sare ar da un numar fals de optimist.

Formele sunt cele **reale**, luate din rularile pe scanul scris de mana:
decupajele curatate ies 1215x256 px, iar un sistem produce ~1130 de tokeni.

    python scripts/benchmark_finetune.py
    python scripts/benchmark_finetune.py --repetari 5 --exemple 50000
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from app.recognition.grandstaff_engine import load_grandstaff  # noqa: E402
from app.training.loop import trainable_for  # noqa: E402

# Masurat pe decupajele curatate ale paginii 1 (PROJECT.md §14.7).
SYSTEM_HEIGHT = 256
SYSTEM_WIDTH = 1215
SEQUENCE_LENGTH = 1130

# Cate exemple are GrandStaff, ca ordin de marime. Serveste doar la a traduce
# secunde/pas in ore/epoca; se poate schimba din linia de comanda.
DATASET_EXAMPLES = 50_000

LEARNING_RATE = 1e-4  # ca in `configure_optimizers` din smt_trainer.py


def synthetic_system(height: int = SYSTEM_HEIGHT, width: int = SYSTEM_WIDTH) -> np.ndarray:
    """Un sistem de doua portative, desenat programatic.

    Nu produce muzica reala si nu pretinde asta: costul unui pas de antrenare
    depinde de *forma* tensorilor, nu de ce e desenat in ei. Serveste ca
    encoderul sa primeasca o intrare de dimensiunea corecta.
    """
    image = np.full((height, width), 255, dtype=np.uint8)
    for staff in (0, 1):
        top = 40 + staff * 128
        for line in range(5):
            y = top + line * 12
            image[y:y + 2, 20:width - 20] = 0
        for x in range(60, width - 40, 70):
            image[top + 22:top + 34, x:x + 14] = 0
    return image


def make_batch(config, height: int, width: int, length: int):
    """Un lot de un exemplu, in forma pe care o cere modelul.

    Lotul e de **unu** intentionat: `forward_decoder` din `modeling_smt.py`
    trece `memory_key_padding_mask=None` cu un „[TODO] This only works with one
    sample per batch", iar configuratia oficiala de antrenare foloseste tot
    `batch_size: 1`. Un lot mai mare ar masura altceva decat se antreneaza.
    """
    image = synthetic_system(height, width)
    encoder_input = torch.from_numpy(image).float().div(255.0).unsqueeze(0).unsqueeze(0)

    # Tokeni oarecare din vocabular; continutul nu schimba costul, doar forma.
    vocabulary = int(config.out_categories)
    tokens = torch.randint(low=0, high=vocabulary, size=(1, length), dtype=torch.long)
    labels = torch.randint(low=0, high=vocabulary, size=(1, length), dtype=torch.long)
    return encoder_input, tokens, labels


def memory_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except Exception:
        return float("nan")


def measure(model, batch, mode: str, repeats: int) -> dict:
    """Secunde pe pas, ca mediana a `repeats` masuratori dupa o incalzire.

    Mediana, nu media: pe masina asta o singura rulare poate fi de cateva ori
    mai lenta din cauze care n-au legatura cu codul (§3.10). Media ar purta
    valoarea aberanta cu ea, mediana nu.
    """
    encoder_input, tokens, labels = batch
    parameters = trainable_for(model, mode)
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE, amsgrad=False)
    model.train()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        output = model(encoder_input=encoder_input, decoder_input=tokens, labels=labels)
        output.loss.backward()
        optimizer.step()

    step()  # incalzire: prima trecere plateste alocari care nu se repeta

    before = memory_mb()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        step()
        timings.append(time.perf_counter() - start)
    peak = memory_mb()

    return {
        "mode": mode,
        "seconds": statistics.median(timings),
        "fastest": min(timings),
        "slowest": max(timings),
        "params": sum(p.numel() for p in parameters),
        "memory": peak - before,
    }


def measure_encoder(model, batch, repeats: int) -> float:
    """Cat costa doar trecerea prin encoder, fara gradienti.

    Conteaza pentru ca in configuratiile in care encoderul e **inghetat**,
    iesirea lui nu se schimba niciodata pentru aceeasi imagine — deci poate fi
    calculata o singura data si pastrata pe disc. Cat din costul unui pas ar
    disparea asa e diferenta dintre „nefezabil" si „o rulare peste noapte".
    """
    encoder_input, _, _ = batch
    model.eval()
    with torch.no_grad():
        model.forward_encoder(encoder_input)  # incalzire
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.forward_encoder(encoder_input)
            timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def report(results: list[dict], examples: int, encoder_seconds: float | None = None) -> None:
    names = {
        "tot": "tot modelul",
        "decoder": "decoder (encoder inghetat)",
        "capat": "ultimul strat + proiectia",
    }
    print(f"\n{'configuratie':28} {'parametri':>12} {'s/pas':>8} {'interval':>16} {'ore/epoca':>10}")
    print("-" * 78)
    for item in results:
        hours = item["seconds"] * examples / 3600
        interval = f"{item['fastest']:.1f}-{item['slowest']:.1f}"
        print(
            f"{names[item['mode']]:28} {item['params'] / 1e6:>10.2f}M "
            f"{item['seconds']:>8.2f} {interval:>16} {hours:>10.1f}"
        )

    if encoder_seconds is not None:
        print(f"\ntrecerea prin encoder, fara gradienti: {encoder_seconds:.1f} s")
        print("cu iesirile encoderului precalculate o data si pastrate pe disc:")
        for item in results:
            if item["mode"] == "tot":
                continue  # aici encoderul se antreneaza, deci nu poate fi cache-uit
            saved = max(item["seconds"] - encoder_seconds, 0.0)
            print(
                f"  {names[item['mode']]:26} {saved:6.1f} s/pas  "
                f"({saved * examples / 3600:7.1f} ore/epoca)"
            )

    print(f"\nore/epoca calculate pentru {examples:_} exemple, lot de 1.".replace("_", " "))
    print(
        "Cifrele sunt un **plafon inferior**: excluderea Defender din §3.10 nu e inca\n"
        "facuta, iar pe masina asta aceeasi suita de teste a rulat si in 2,4 s si in 64 s.\n"
        "Ce ramane valid indiferent de viteza masinii sunt **rapoartele** dintre configuratii."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repetari", type=int, default=3, help="masuratori per configuratie")
    parser.add_argument("--exemple", type=int, default=DATASET_EXAMPLES)
    parser.add_argument("--latime", type=int, default=SYSTEM_WIDTH)
    parser.add_argument("--tokeni", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--doar", choices=("tot", "decoder", "capat"), default=None,
        help="masoara o singura configuratie",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    print(f"torch {torch.__version__} | cpu | threads {torch.get_num_threads()}")

    start = time.perf_counter()
    model, config = load_grandstaff(torch.device("cpu"))
    print(
        f"model incarcat in {time.perf_counter() - start:.1f} s | "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parametri"
    )
    print(
        f"lot: imagine {SYSTEM_HEIGHT}x{args.latime}, {args.tokeni} tokeni "
        f"(formele masurate pe scanul real, §14.7)"
    )

    batch = make_batch(config, SYSTEM_HEIGHT, args.latime, args.tokeni)
    modes = [args.doar] if args.doar else ["tot", "decoder", "capat"]

    results = []
    for mode in modes:
        print(f"  masor '{mode}' ...", flush=True)
        results.append(measure(model, batch, mode, args.repetari))

    print("  masor encoderul singur ...", flush=True)
    encoder_seconds = measure_encoder(model, batch, args.repetari)

    report(results, args.exemple, encoder_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
