"""Perechile scrise de `scripts/dataset.py` -> tensorii pe care ii cere SMT.

Etapa 11. `dataset.py` produce (decupaj de sistem, sir de tokeni); aici se
transforma in ce primeste modelul: o imagine adusa la scara, o intrare de
decoder si etichetele.

**Conventia nu e inventata de noi** — e cea din `vendor/SMT/data.py`, fiindca
greutatile au fost antrenate asa si orice abatere ar invata modelul altceva
decat stie:

- sirul e `<bos>` + tokeni + `<eos>` (`data.py:40`);
- `decoder_input` e sirul **fara ultimul** token, etichetele sunt sirul **fara
  primul** (`data.py:104-111`);
- intrarea de decoder trece printr-un „teacher forcing" cu zgomot: ~20% dintre
  pozitii se inlocuiesc cu tokeni la intamplare (`apply_teacher_forcing`,
  `data.py:128`). Nu e o scapare, e regimul lor de antrenare — modelul invata
  sa nu se sprijine orbeste pe propria iesire de dinainte.

Imaginea se aduce la scara cu **`resize_for_model`, aceeasi functie pe care o
foloseste inferenta**. Daca cele doua ar diverge, modelul ar invata pe imagini
de alta marime decat cele pe care le primeste apoi, si nimic n-ar semnala-o.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

# Cat de des se strica un token din intrarea de decoder. Valoarea e cea din
# `vendor/SMT/smt_trainer.py`, unde `OMRIMG2SEQDataset` primeste 0.2 implicit.
TEACHER_FORCING_ERROR_RATE = 0.2


class UnknownTokens(ValueError):
    """Tinta contine tokeni care nu sunt in vocabularul modelului.

    Se ridica in loc sa se inlocuiasca cu `<pad>` sau cu un token oarecare: o
    tinta pe care modelul n-o poate reprezenta e o problema de date, iar
    inlocuirea tacuta ar antrena modelul sa scrie altceva decat scrie pagina.

    Masurat pe pagina 3 la 2026-09-01: **0 tokeni necunoscuti din 648**, deci
    scriitorul nostru produce deja vocabularul lui `smt-grandstaff`. Verificarea
    ramane fiindca o partitura viitoare poate scoate un token nou.
    """


@dataclass(frozen=True)
class Pair:
    """O pereche de antrenare: un sistem si transcrierea lui."""

    image: Path
    tokens: list[str]
    staves: int
    writer: str


def load_pairs(path: str | Path) -> list[Pair]:
    """Citeste `date.jsonl` scris de `scripts/dataset.py`.

    Caile imaginilor sunt relative la fisier, ca setul sa se poata muta intreg.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"nu exista fisierul {path}")

    pairs = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("image", "transcription", "staves", "writer"):
            if field not in row:
                raise ValueError(
                    f"{path}, linia {number}: lipseste `{field}` — perechile scrise "
                    "vechi n-au toate campurile: fara `staves` imaginea s-ar aduce "
                    "la alta scara decat la inferenta, iar fara `writer` validarea "
                    "n-ar putea tine un scris deoparte; "
                    "regenereaza-le cu `scripts/dataset.py`"
                )
        pairs.append(
            Pair(
                image=path.parent / row["image"],
                tokens=row["transcription"].split(),
                staves=int(row["staves"]),
                writer=str(row["writer"]),
            )
        )
    return pairs


def encode_target(tokens: list[str], w2i: dict) -> list[int]:
    """Tokenii unei tinte -> id-uri, cu `<bos>` si `<eos>` la capete."""
    sequence = ["<bos>"] + list(tokens) + ["<eos>"]
    unknown = sorted({token for token in sequence if token not in w2i})
    if unknown:
        raise UnknownTokens(
            f"{len(unknown)} tokeni nu sunt in vocabularul modelului: "
            f"{', '.join(repr(token) for token in unknown[:10])}"
            + (" ..." if len(unknown) > 10 else "")
        )
    return [w2i[token] for token in sequence]


def noisy_decoder_input(ids: list[int], vocabulary: int, rate: float, rng: random.Random):
    """Intrarea de decoder, cu zgomotul din `apply_teacher_forcing`.

    Prima pozitie (`<bos>`) se lasa neatinsa, ca in implementarea lor: bucla
    incepe de la 1.
    """
    noised = list(ids)
    for index in range(1, len(noised)):
        if rng.random() < rate:
            noised[index] = rng.randrange(vocabulary)
    return noised


def augmented_tensor(resized):
    """Imaginea trecuta prin augmentarea din `vendor/SMT/data_augmentation`.

    E codul lor, MIT, folosit exact unde il folosesc si ei — dupa aducerea la
    scara, inainte de model (`OMRIMG2SEQDataset.__getitem__`). Face perspectiva,
    distorsiune elastica, **eroziune/dilatare** si ajustari de luminozitate si
    contrast.

    Conteaza pentru problema noastra: cele ~500 de partituri sunt fiecare in alt
    scris, iar eroziunea/dilatarea si distorsiunea elastica sunt tocmai
    grosimea penitei si deformarea trasaturii — o buna parte din ce deosebeste
    doua maini. Nu inventeaza scrisuri noi, dar face modelul mai putin sensibil
    la cel pe care nu l-a vazut.

    Se aplica **numai la antrenare**. La masurare imaginea trebuie sa fie exact
    cea de pe pagina, altfel cifra ar descrie o pagina care nu exista.
    """
    import warnings

    from app.recognition.grandstaff_engine import _ensure_vendor_on_path

    _ensure_vendor_on_path()
    from data_augmentation.data_augmentation import augment

    # `transforms_custom.py` cheama `tform.estimate(...)`, depreciat in skimage
    # 0.26. E cod vendor, nu al nostru, si proiectul rideaza avertismentele la
    # eroare (`filterwarnings = ["error"]`) — deci fara filtrul asta augmentarea
    # ar cadea, si ar si murdari iesirea fiecarei rulari de antrenare.
    #
    # Se taie **doar** avertismentul asta, dupa mesaj: politica stricta ramane
    # in picioare pentru orice altceva, inclusiv pentru alte FutureWarning-uri.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*`estimate` is deprecated.*", category=FutureWarning
        )
        # `augment` intoarce (1, H, W) in [0, 1] — lantul lor se termina cu
        # `Grayscale()` si `ToTensor()`. Mai lipseste dimensiunea de lot, si o
        # punem aici, ca apelantul sa primeasca aceeasi forma (1, 1, H, W) ca pe
        # drumul fara augmentare.
        return augment(resized).unsqueeze(0)


def encode_pair(
    pair: Pair,
    w2i: dict,
    maxh: int,
    maxw: int,
    rate: float = TEACHER_FORCING_ERROR_RATE,
    rng: random.Random | None = None,
    augment: bool = False,
):
    """O pereche -> `(encoder_input, decoder_input, labels)`, lot de unu.

    Lotul e de **unu** fiindca asa cere modelul: `forward_decoder` din
    `modeling_smt.py` trece `memory_key_padding_mask=None` cu un „[TODO] This
    only works with one sample per batch", iar configuratia oficiala de
    antrenare foloseste tot `batch_size: 1`.

    `augment` se da la antrenare si **nu** la masurare — vezi `augmented_tensor`.
    """
    import numpy as np
    import torch
    from PIL import Image

    from app.recognition.grandstaff_engine import resize_for_model

    rng = rng or random.Random()

    page = np.array(Image.open(pair.image).convert("L"))
    resized, _ = resize_for_model(page, pair.staves, maxh, maxw)
    if augment:
        encoder_input = augmented_tensor(resized)
    else:
        encoder_input = (
            torch.from_numpy(resized).float().div(255.0).unsqueeze(0).unsqueeze(0)
        )

    ids = encode_target(pair.tokens, w2i)
    noised = noisy_decoder_input(ids, len(w2i), rate, rng)

    decoder_input = torch.tensor([noised[:-1]], dtype=torch.long)
    labels = torch.tensor([ids[1:]], dtype=torch.long)
    return encoder_input, decoder_input, labels


def usable_pairs(pairs: list[Pair], w2i: dict) -> tuple[list[Pair], list[tuple[Pair, str]]]:
    """Perechile pe care modelul le poate reprezenta, si de ce cad celelalte.

    Se cheama **inainte** de primul pas, nu la nimereala in timpul antrenarii.
    Perechile se aleg aleator din set, deci un token necunoscut intr-una singura
    ar opri rularea abia la pasul 137, dupa ore de calcul — si cu un mesaj care
    n-ar spune ca problema era acolo de la inceput.

    Masurat pe 2026-09-02: pagina 3 trece intreaga (648 de tokeni, 0
    necunoscuti), pe cand „Doamne bun asculta" cade pe `[2d#`, `[2e`, `[2f#` —
    inceputuri de legato, notatie pe care `smt-grandstaff` n-o are in vocabular
    si pe care §5.2 o si tine in afara scopului v1.
    """
    good: list[Pair] = []
    bad: list[tuple[Pair, str]] = []
    for pair in pairs:
        try:
            encode_target(pair.tokens, w2i)
        except UnknownTokens as problem:
            bad.append((pair, str(problem)))
        else:
            good.append(pair)
    return good, bad


def writers(pairs: list[Pair]) -> dict[str, int]:
    """Cate perechi are fiecare scriitor, ca sa se vada inainte de impartire."""
    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.writer] = counts.get(pair.writer, 0) + 1
    return dict(sorted(counts.items()))


def split_by_writer(
    pairs: list[Pair], held_out: str | set[str]
) -> tuple[list[Pair], list[Pair]]:
    """Imparte perechile in (antrenare, validare), **pe scriitor**.

    Asta e piesa de care atarna orice cifra de dupa. Cele ~500 de partituri sunt
    fiecare in alt scris, deci intrebarea produsului nu e „cat citeste dintr-o
    pagina pe care a mai vazut-o", ci „cat citeste dintr-un **scris** pe care nu
    l-a vazut niciodata". O impartire pe pagini ar lasa acelasi scris in
    amandoua partile si ar raporta memorare drept citire — cu atat mai inselator
    cu cat setul e mai mic.

    Se refuza, in loc sa se intoarca o parte goala:

    * un scriitor cerut care nu exista — aproape sigur o greseala de scriere,
      iar rezultatul ar fi o validare goala din care n-ar iesi nicio cifra;
    * o impartire care goleste antrenarea — daca tii deoparte singurul scriitor,
      n-a mai ramas ce antrena.
    """
    wanted = {held_out} if isinstance(held_out, str) else set(held_out)
    known = set(writers(pairs))
    missing = sorted(wanted - known)
    if missing:
        raise ValueError(
            f"scriitor necunoscut: {', '.join(missing)} — in set sunt "
            f"{', '.join(sorted(known)) or '(niciunul)'}"
        )

    train = [pair for pair in pairs if pair.writer not in wanted]
    validation = [pair for pair in pairs if pair.writer in wanted]
    if not train:
        raise ValueError(
            f"tinand deoparte {', '.join(sorted(wanted))} nu mai ramane nimic de "
            "antrenat — setul are nevoie de cel putin doua scrisuri"
        )
    return train, validation


def describe_writers(pairs: list[Pair], show_all_under: int = 9) -> str:
    """Cati scriitori si cate pagini fiecare, pe un rand.

    Exista fiindca eticheta de scriitor nu e un om, ci o **grupare**: cand nu se
    stie cine a scris o pagina — cazul obisnuit pentru cele ~500 de partituri —
    eticheta ajunge sa fie identificatorul paginii, si atunci sunt sute. Un
    dictionar cu sute de chei varsat pe ecran nu se poate citi, deci peste un
    prag se rezuma.
    """
    counts = writers(pairs)
    if not counts:
        return "niciun scriitor"
    if len(counts) < show_all_under:
        return ", ".join(f"{name}: {n}" for name, n in counts.items())

    values = sorted(counts.values())
    median = values[len(values) // 2]
    return (
        f"{len(counts)} maini, {sum(values)} perechi "
        f"(mediana {median}/mana, min {values[0]}, max {values[-1]})"
    )


def split_by_fraction(
    pairs: list[Pair], fraction: float, seed: int = 0
) -> tuple[list[Pair], list[Pair]]:
    """Tine deoparte o cota din **maini**, nu din pagini.

    Varianta potrivita cand fiecare pagina e alt scris si eticheta ajunge sa fie
    identificatorul paginii: a tine deoparte o singura eticheta ar da o validare
    de o pagina, prea subtire ca sa insemne ceva.

    Alegerea e **stabila**: aceleasi maini si aceeasi samanta dau aceeasi
    impartire. Nu e o comoditate — o rulare reluata din checkpoint trebuie sa
    gaseasca exact aceeasi impartire, altfel ar antrena pe pagini care erau
    validare inainte, si cifra de la final ar fi masurata pe date vazute.

    Se pastreaza cel putin o mana de fiecare parte, oricat de mica ar fi cota.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"cota de validare se da intre 0 si 1, nu {fraction}")

    names = sorted(writers(pairs))
    if len(names) < 2:
        raise ValueError(
            f"impartirea pe maini cere cel putin doua, iar setul are {len(names)}"
        )

    chosen = random.Random(seed).sample(names, len(names))
    count = min(max(round(len(names) * fraction), 1), len(names) - 1)
    return split_by_writer(pairs, set(chosen[:count]))
