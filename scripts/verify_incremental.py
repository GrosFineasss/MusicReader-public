"""Verifica faptul ca decodarea incrementala da acelasi rezultat ca referinta.

O optimizare care schimba iesirea modelului nu e o optimizare, e un bug.
Comparam logits pozitie cu pozitie, nu doar tokenul ales: doua implementari pot
alege acelasi argmax si totusi sa calculeze lucruri diferite, iar diferenta ar
iesi la iveala abia pe o partitura reala.

Se verifica **amandoua** modelele, fiindca difera si aritmetica, nu doar numele
modulelor: forkul JAZZMUS nu imparte la sqrt(head_dim), implementarea oficiala
imparte. O portare facuta „dupa nume" ar produce tokeni plauzibili si gresiti.

    python scripts/verify_incremental.py --model jazzmus
    python scripts/verify_incremental.py --model grandstaff
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ISMIR-Jazzmus"))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

from app.recognition.incremental import IncrementalSMT  # noqa: E402
from scripts.benchmark_smt import load_model, synthetic_staff  # noqa: E402


def load_grandstaff_model(device):
    from app.recognition.grandstaff_engine import load_grandstaff

    return load_grandstaff(device)


def synthetic_system(height: int = 256, width: int = 1024) -> np.ndarray:
    """Doua portative desenate cu linii, la inaltimea pe care o cere grandstaff.

    Nu e muzica; e o intrare valida, stabila intre rulari. Pentru echivalenta
    conteaza ca amandoua implementarile primesc acelasi tensor, nu ce scrie in el.
    """
    image = np.full((height, width), 255, dtype=np.uint8)
    for top in (height // 4, 3 * height // 4):
        for offset in range(-2, 3):
            image[top + offset * 8, 20 : width - 20] = 0
    return image


@torch.no_grad()
def compare(model, image, steps: int, tensor=None) -> tuple[float, int, float, float]:
    """Ruleaza referinta si decodarea incrementala pas cu pas, pe acelasi prefix.

    `tensor` se poate da gata pregatit, pentru modelele care nu folosesc
    preprocesarea din repo-ul Jazzmus.
    """
    if tensor is None:
        from jazzmus.dataset.data_preprocessing import convert_img_to_tensor

        tensor = convert_img_to_tensor(image)

    fast = IncrementalSMT(model)
    memory = fast.encode(tensor)
    encoder_output = model.forward_encoder(tensor.unsqueeze(0))

    prefix = [model.w2i["<bos>"]]
    worst = 0.0
    mismatches = 0
    slow_seconds = 0.0
    fast_seconds = 0.0

    for _ in range(steps):
        t0 = time.perf_counter()
        logits = model.forward_decoder(
            encoder_output, torch.tensor([prefix], dtype=torch.long)
        ).logits
        # Forkul JAZZMUS da logits ca (batch, vocabular, pozitie); implementarea
        # oficiala ca (batch, pozitie, vocabular). Aceeasi ultima pozitie, alta axa.
        reference = logits[:, -1, :] if fast.layout.logits_last_dim else logits[:, :, -1]
        slow_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        incremental = fast.step(memory, prefix[-1])
        fast_seconds += time.perf_counter() - t0

        worst = max(worst, (reference - incremental).abs().max().item())
        if int(reference.argmax()) != int(incremental.argmax()):
            mismatches += 1
        prefix.append(int(reference.argmax()))

    return worst, mismatches, slow_seconds, fast_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--model", choices=("jazzmus", "grandstaff"), default="jazzmus")
    args = parser.parse_args()

    torch.set_num_threads(1)
    if args.model == "jazzmus":
        model, _ = load_model(torch.device("cpu"))
        image, tensor = synthetic_staff(), None
    else:
        from app.recognition.grandstaff_engine import GrandstaffEngine

        model, _ = load_grandstaff_model(torch.device("cpu"))
        image = synthetic_system()
        tensor = GrandstaffEngine._to_tensor(image)

    worst, mismatches, slow, fast = compare(model, image, args.steps, tensor=tensor)

    print(f"model:                     {args.model}")
    print(f"pasi comparati:            {args.steps}")
    print(f"diferenta maxima de logit: {worst:.3e}")
    print(f"tokeni diferiti:           {mismatches}")
    print(f"referinta:                 {slow:.2f} s ({slow / args.steps * 1000:.0f} ms/pas)")
    print(f"incremental:               {fast:.2f} s ({fast / args.steps * 1000:.1f} ms/pas)")
    print(f"accelerare:                x{slow / fast:.0f}")

    ok = worst < 1e-3 and mismatches == 0
    print("\nREZULTAT:", "echivalent" if ok else "DIFERIT — nu se poate folosi")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
