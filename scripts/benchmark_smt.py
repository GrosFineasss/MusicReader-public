"""Etapa 1: cat de repede ruleaza SMT/jazzmus pe *acest* CPU.

Intrebarea la care raspunde scriptul nu e "e modelul bun?", ci "e viabil sa
rulam un portativ pe un laptop fara GPU?". Calitatea transcrierii se masoara
abia pe o poza reala.

Masuram separat trei lucruri, pentru ca se comporta foarte diferit:

  1. incarcarea modelului      — se plateste o singura data, la pornire
  2. encoderul ConvNeXt        — o singura trecere pe portativ, cost fix
  3. decodarea autoregresiva   — costul real

Punctul 3 e cel care decide. `SMTModelForCausalLM.predict()` nu foloseste
KV cache: la fiecare token re-ruleaza decoderul peste tot prefixul de pana
atunci. Costul total nu creste liniar cu lungimea secventei, ci **patratic**.
De asta masuram costul pe pas la mai multe lungimi de prefix, nu doar o medie.
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

REPO = "JuanCarlosMartinezSevilla/jazzmus-model"
STAFF_HEIGHT = 128  # impus de model
STAFF_WIDTH = 1000  # latimea maxima acceptata


def synthetic_staff(width: int = STAFF_WIDTH) -> np.ndarray:
    """Un portativ desenat programatic: cinci linii si cateva capete de nota.

    Nu produce muzica reala si nu pretinde asta. Serveste doar ca encoderul sa
    primeasca o intrare de dimensiunea corecta, ca sa putem cronometra.
    """
    img = np.full((STAFF_HEIGHT, width), 255, dtype=np.uint8)
    for i in range(5):
        y = 40 + i * 12
        img[y : y + 2, :] = 0
    for x in range(60, width - 40, 70):
        img[62:74, x : x + 14] = 0
    return img


def load_model(device: torch.device):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torch.nn import Conv1d

    from jazzmus.model.smt.configuration_smt import SMTConfig
    from jazzmus.model.smt.modeling_smt import SMTModelForCausalLM

    config = SMTConfig.from_pretrained(REPO)
    weights = load_file(hf_hub_download(repo_id=REPO, filename="model.safetensors"))

    # Repo-ul face aceeasi corectie in predict.py: greutatile salvate au
    # embedding si out_layer de dimensiuni diferite fata de config.
    embed_size = weights["decoder.embedding.weight"].shape[0]
    out_size = weights["decoder.out_layer.weight"].shape[0]
    config.out_categories = embed_size
    model = SMTModelForCausalLM(config)
    if embed_size != out_size:
        model.decoder.out_layer = Conv1d(config.d_model, out_size, kernel_size=1)
    model.load_state_dict(weights, strict=True)

    model.to(device)
    model.positional_2D.pe = model.positional_2D.pe.to(device)
    model.decoder.positional_1D.pe = model.decoder.positional_1D.pe.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def measure(model, image: np.ndarray, device: torch.device, probe_lengths: list[int]):
    from jazzmus.dataset.data_preprocessing import convert_img_to_tensor

    tensor = convert_img_to_tensor(image).to(device)

    t0 = time.perf_counter()
    encoder_output = model.forward_encoder(tensor.unsqueeze(0))
    encoder_seconds = time.perf_counter() - t0

    bos = torch.tensor([[model.w2i["<bos>"]]], device=device)
    step_seconds: dict[int, float] = {}
    for length in probe_lengths:
        prefix = bos.repeat(1, length)
        model.forward_decoder(encoder_output, prefix.long())  # incalzire
        t0 = time.perf_counter()
        model.forward_decoder(encoder_output, prefix.long())
        step_seconds[length] = time.perf_counter() - t0

    return encoder_seconds, step_seconds


def projected_total(step_seconds: dict[int, float], maxlen: int) -> float:
    """Timp total estimat pentru `maxlen` pasi, integrand costul pe pas.

    Costul unui pas creste cu lungimea prefixului; aproximam liniar intre
    punctele masurate si insumam.
    """
    lengths = sorted(step_seconds)
    return float(np.trapezoid([step_seconds[n] for n in lengths], lengths) / (lengths[-1] - lengths[0]) * maxlen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=None, help="numar de thread-uri torch")
    parser.add_argument("--width", type=int, default=STAFF_WIDTH)
    parser.add_argument("--decode", action="store_true", help="ruleaza si o decodare completa reala")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    device = torch.device("cpu")
    print(f"torch {torch.__version__} | device {device} | threads {torch.get_num_threads()}")

    t0 = time.perf_counter()
    model, config = load_model(device)
    load_seconds = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters())
    print(f"model incarcat in {load_seconds:.1f}s | {params / 1e6:.1f}M parametri | maxlen={config.maxlen}")

    image = synthetic_staff(args.width)
    probes = [1, 32, 64, 128, 200, 276]
    encoder_seconds, step_seconds = measure(model, image, device, probes)

    print(f"\nencoder ConvNeXt (o data pe portativ): {encoder_seconds * 1000:.0f} ms")
    print("\ncost pe pas de decodare, dupa lungimea prefixului:")
    for length in probes:
        print(f"  prefix {length:>4} tokeni: {step_seconds[length] * 1000:7.1f} ms")

    growth = step_seconds[probes[-1]] / step_seconds[probes[0]]
    total = encoder_seconds + projected_total(step_seconds, config.maxlen)
    print(f"\ncrestere de la 1 la {probes[-1]} tokeni: x{growth:.1f}")
    print(f"estimare portativ complet ({config.maxlen} tokeni): {total:.1f} s")
    print(f"estimare sistem SATB (4 portative):            {total * 4:.1f} s")

    if args.decode:
        t0 = time.perf_counter()
        tokens, _ = model.predict(
            input=__import__("jazzmus.dataset.data_preprocessing", fromlist=["x"])
            .convert_img_to_tensor(image)
            .to(device)
        )
        elapsed = time.perf_counter() - t0
        print(f"\ndecodare reala: {len(tokens)} tokeni in {elapsed:.1f}s "
              f"({elapsed / max(len(tokens), 1):.2f} s/token)")
        print("primii 40 tokeni:", tokens[:40])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
