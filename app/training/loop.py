"""Bucla de fine-tuning partial, reluabila.

Etapa 11. Configuratia se alege prin masuratoare, nu aici. Cifrele, remasurate
pe 2026-09-01 cu `scripts/benchmark_finetune.py`, pe aceleasi forme reale
(imagine 256x1215, 1130 de tokeni):

| ce se antreneaza | parametri | s/pas | §8.1, 29.08 |
|---|---|---|---|
| tot modelul | 21,4 M | **6,89** | 152,9 |
| decoderul, encoderul inghetat | 15,8 M | **5,14** | 121,2 |
| capatul — ultimul strat de decoder + proiectia de vocabular | 6,0 M | **3,34** | 66-70 |

**Cifrele vechi erau de ~20x pesimiste**, pe toate trei configuratiile — exact
avertismentul pe care §8.1 si-l dadea singur, si problema de masurare din §3.10.
Deci 500 de pasi inseamna 28 de minute pe `capat` si 57 pe tot modelul, nu o
noapte, iar „fine-tuning complet exclus in practica" nu mai e adevarat. Ce a
ramas valid din §8.1 sunt **rapoartele** dintre configuratii, cum prezisese.

`capat` ramane implicit, dar nu din constrangere de timp: pe cateva zeci de
pagini, 21,4 M de parametri au mult mai mult loc sa se supraantreneze decat 6 M.
Care e mai bun se decide prin masuratoare, ca tot restul.

**Reluarea nu e optionala.** Chiar si scurta, o rulare care ar trebui luata de
la capat cand se inchide laptopul e munca aruncata. Starea se scrie periodic si
rularea urmatoare continua din ea — aceeasi hotarare ca `batch.db` din §12.2.

Ce se salveaza e **doar ce se antreneaza**: restul greutatilor sunt inghetate,
deci identice cu cele publicate, si se aduc oricum de la HuggingFace la fiecare
pornire. Pe `capat`, un checkpoint iese de ~24 MB in loc de ~86 MB.
"""

from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from app.training.data import Pair, encode_pair

# Ca in `configure_optimizers` din `vendor/SMT/smt_trainer.py`.
LEARNING_RATE = 1e-4

# Cat de des se scrie starea. La 3,3 s/pas, 25 de pasi inseamna ~1,4 minute
# pierdute in cel mai rau caz, si o scriere de 24 MB la fel de rar — destul de
# des cat sa nu doara, destul de rar cat sa nu coste.
CHECKPOINT_EVERY = 25


def trainable_for(model, mode: str = "capat") -> list:
    """Ce se antreneaza in fiecare configuratie, si ce se ingheata.

    Mutata aici din `scripts/benchmark_finetune.py`, care o importa acum de la
    noi: masuratoarea si antrenarea trebuie sa inghete **exact** aceleasi
    straturi, altfel cifra masurata nu descrie rularea reala.

    `capat` e ultimul strat de decoder plus proiectia de vocabular — cel mai
    mare strat al modelului (`out_categories` x `d_model`).
    """
    # Verificarea e **inaintea** inghetarii: altfel o greseala de scriere ar
    # lasa modelul cu toti parametrii inghetati si ar ridica abia dupa aceea,
    # iar apelantul care prinde exceptia ar ramane cu un model mut.
    if mode not in ("tot", "decoder", "capat"):
        raise ValueError(f"configuratie necunoscuta: {mode}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if mode == "tot":
        chosen = list(model.encoder.parameters()) + list(model.decoder.parameters())
    elif mode == "decoder":
        chosen = list(model.decoder.parameters())
    else:
        chosen = list(model.decoder.decoder.layers[-1].parameters()) + list(
            model.decoder.vocab_projection.parameters()
        )

    for parameter in chosen:
        parameter.requires_grad_(True)
    return chosen


def trainable_state(model, mode: str) -> dict:
    """Numele si valorile parametrilor antrenati, pentru checkpoint.

    Se merge dupa `requires_grad`, nu dupa o lista de nume scrisa de mana:
    `trainable_for` a hotarat deja, si doua surse de adevar s-ar putea
    contrazice tacut.
    """
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


@dataclass
class Progress:
    """Cat s-a facut si cum merge, ca sa se poata citi din afara buclei."""

    step: int = 0
    steps: int = 0
    loss: float = float("nan")
    seconds_per_step: float = float("nan")
    history: list[float] = field(default_factory=list)

    @property
    def remaining_hours(self) -> float:
        left = max(self.steps - self.step, 0)
        return left * self.seconds_per_step / 3600


@contextmanager
def rng_neatins():
    """Tine cele trei generatoare neatinse peste o bucata de cod strain.

    `--evalueaza-la` parea o optiune pasiva — masoara si atat. Nu era:
    masuratoarea trece o imagine prin model si consuma din generatoarele
    globale, iar pasul urmator trage alta augmentare si alt dropout. Masurat pe
    doua rulari cu aceeasi samanta: identice pana la pasul 25 si divergente
    **exact la 26**, imediat dupa prima masuratoare (§15.29). Aceeasi pereche
    trasa, alta pierdere.

    Deci observatia schimba ce observa, si o curba masurata din 25 in 25 nu mai
    e traiectoria pe care ar fi avut-o rularea nemasurata.

    Reparatia sta in bucla, nu in masuratoare: **orice** apel din afara poate
    consuma aleator, iar reproductibilitatea e o promisiune pe care o face
    bucla. Cine scrie a doua zi alt `on_checkpoint` n-ar avea de unde sa stie
    ca trebuie sa si-o pazeasca singur.
    """
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def scor_checkpoint(matched_notes: int, hyp_events: int) -> tuple[int, int]:
    """Cat de bun e un checkpoint, cand se pastreaza cel mai bun, nu ultimul.

    Note corecte intai; la egalitate, cine scrie mai putin. Alegerea nu e
    neutra, si celelalte doua candidate sunt amandoua gresite:

    - **SER-ul** pune primul un model care tace. Unul care nu scrie nimic iese
      fix la 1,000, iar unul care a scris 14 note din 182 a iesit la 0,940 si
      parea cel mai bun punct al unei rulari (§15.28).
    - **Precizia** e pacalita in aceeasi directie: cine scrie o nota si o
      nimereste are 100%.

    **Corectat pe 2026-09-04.** Aici scria „notele corecte nu se pot castiga
    tacand — cine nu scrie nimic nimereste zero". Se pot: §15.42 a aratat ca o
    pauza potrivita cu o pauza intra in `matched`, fiindca niciuna n-are
    inaltime. Un model care umple pagina cu tacere nimereste tacerea referintei
    — 11 din cele 16 „corecte" ale reperului erau asa. De aceea se numara acum
    `matched_notes`, nu `matched`: acela chiar nu se poate castiga tacand.

    La egalitate conteaza cat are omul de sters dupa: 16 note bune din 37
    scrise e o pagina de corectat, 16 din 202 e una de rescris.
    """
    return (matched_notes, -hyp_events)


def checkpoint_path(out_dir: Path) -> Path:
    return Path(out_dir) / "checkpoint.pt"


def history_path(out_dir: Path) -> Path:
    return Path(out_dir) / "istoric.jsonl"


def finetune(
    model,
    pairs: list[Pair],
    steps: int,
    out_dir: str | Path,
    *,
    mode: str = "capat",
    maxh: int = 256,
    maxw: int = 1024,
    learning_rate: float = LEARNING_RATE,
    checkpoint_every: int = CHECKPOINT_EVERY,
    seed: int = 0,
    resume: bool = True,
    augment: bool = True,
    on_step=None,
    on_checkpoint=None,
) -> Progress:
    """Antreneaza `steps` pasi, reluand din checkpoint daca exista unul.

    `steps` e numarul **total**, nu cati mai raman: o rulare reluata cu acelasi
    `--pasi` se opreste unde s-ar fi oprit si daca n-ar fi fost intrerupta. Asa
    „500 de pasi" inseamna acelasi lucru indiferent de cate ori s-a inchis
    laptopul.

    `pairs` sunt **numai** perechile de antrenare. Impartirea pe scriitor se
    face inainte, de apelant (`split_by_writer`): bucla n-are de unde sti ce
    scris se tine deoparte, iar daca ar ghici, ar putea antrena pe exact
    paginile pe care se masoara.

    `augment` e pornit implicit — la antrenare imaginile trec prin augmentarea
    din vendor, care perturba grosimea trasaturii si deformarile locale.

    `on_checkpoint(progress, cale)` se cheama dupa fiecare scriere de stare.
    Exista ca sa se poata **masura pe parcurs**: fara ea, o rulare de 2000 de
    pasi spune abia la sfarsit daca a ajutat.
    """
    import torch

    if not pairs:
        raise ValueError("nu exista perechi de antrenare")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parameters = trainable_for(model, mode)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, amsgrad=False)
    rng = random.Random(seed)
    _seed_everything(seed)
    progress = Progress(steps=steps)

    path = checkpoint_path(out_dir)
    if resume and path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(_tuple_state(state["rng"]))
        _restore_everything(state)
        progress.step = state["step"]
        progress.history = state.get("history", [])

    model.train()
    started = time.perf_counter()
    done_here = 0

    while progress.step < steps:
        pair = pairs[rng.randrange(len(pairs))]
        encoder_input, decoder_input, labels = encode_pair(
            pair, model.w2i, maxh, maxw, rng=rng, augment=augment
        )

        optimizer.zero_grad(set_to_none=True)
        output = model(
            encoder_input=encoder_input, decoder_input=decoder_input, labels=labels
        )
        output.loss.backward()
        optimizer.step()

        progress.step += 1
        done_here += 1
        progress.loss = float(output.loss.detach())
        progress.history.append(progress.loss)
        progress.seconds_per_step = (time.perf_counter() - started) / done_here

        with history_path(out_dir).open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": progress.step,
                        "loss": progress.loss,
                        "image": pair.image.name,
                        "seconds": round(progress.seconds_per_step, 2),
                    }
                )
                + "\n"
            )

        if on_step is not None:
            with rng_neatins():
                on_step(progress)

        if progress.step % checkpoint_every == 0 or progress.step == steps:
            save_checkpoint(path, model, optimizer, rng, progress, mode)
            # **Dupa** scriere, nu inainte: cine vrea sa masoare la checkpoint
            # are nevoie de fisierul de pe disc, nu de promisiunea lui.
            if on_checkpoint is not None:
                with rng_neatins():
                    on_checkpoint(progress, path)

    return progress


def save_checkpoint(path: Path, model, optimizer, rng, progress: Progress, mode: str) -> None:
    """Scrie starea **atomic**: intai alaturi, apoi se muta peste.

    O intrerupere in mijlocul scrierii ar lasa un checkpoint trunchiat, iar
    rularea urmatoare ar cadea la incarcare — adica exact noaptea pe care
    reluarea trebuia s-o salveze.
    """
    import torch

    temporary = path.with_suffix(".pt.nou")
    torch.save(
        {
            "step": progress.step,
            "mode": mode,
            "model": trainable_state(model, mode),
            "optimizer": optimizer.state_dict(),
            "rng": rng.getstate(),
            "rng_numpy": _numpy_state(),
            "rng_torch": torch.get_rng_state(),
            "history": progress.history,
        },
        temporary,
    )
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    """Sarea toate cele trei generatoare, nu doar pe al nostru.

    `--seminte` parea sa faca rularea repetabila, si nu o facea: alegerea
    perechilor mergea pe `random.Random(seed)`, dar **augmentarea** foloseste
    generatorul global al lui numpy (`vendor/SMT/data_augmentation`), iar dropoutul
    pe al lui torch. Amandoua ramaneau nesamantate.

    Masurat pe 2026-09-03: doua rulari cu aceeasi comanda au dat, la pasul 150,
    **15/182 note** si **0/182**. Adica orice comparatie intre doua rulari masura
    zgomotul, nu schimbarea — exact ce nu-ti trebuie cand incerci sa afli daca
    antrenarea ajuta.
    """
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def _numpy_state():
    import numpy as np

    return np.random.get_state()


def _restore_everything(state: dict) -> None:
    """Starea generatoarelor dintr-un checkpoint, daca o are.

    Checkpoint-urile scrise inainte de §15.27 n-au campurile astea; se reiau ca
    pana acum, doar ca fara repetabilitate deplina. Mai bine o reluare care merge
    decat una care refuza sa porneasca.
    """
    import numpy as np
    import torch

    if state.get("rng_numpy") is not None:
        numpy_state = state["rng_numpy"]
        if isinstance(numpy_state, list):
            numpy_state = tuple(numpy_state)
        np.random.set_state(numpy_state)
    if state.get("rng_torch") is not None:
        torch.set_rng_state(state["rng_torch"].to("cpu").to(torch.uint8))


def _tuple_state(state):
    """`random.setstate` cere tupluri; `torch.load` poate intoarce liste."""
    if isinstance(state, list):
        return tuple(_tuple_state(item) for item in state)
    return state
