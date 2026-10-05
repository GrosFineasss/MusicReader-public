"""Motorul `smt-grandstaff`: citeste un **sistem intreg**, nu un portativ.

Diferenta fata de `SmtEngine` nu e modelul, ci unitatea de lucru. `jazzmus-model`
a fost antrenat pe lead sheet-uri — un portativ, o voce — deci pagina se taie in
portative de 128 px. `smt-grandstaff` a fost antrenat pe **grand staff de pian**:
doua portative intr-o singura imagine, cu polifonie pe amandoua. Structural,
asta e exact partitura stransa corala (S+A pe cheia Sol, T+B pe Fa), si de asta
merita incercat pe partiturile tiparite — vezi PROJECT.md §12.4.

Consecinta practica: pagina se taie in **sisteme**, la inaltimea 256, iar
modelul scoate toate vocile sistemului intr-un singur document. Traducerea din
tokenii lui si asamblarea sunt cod determinist, in `bekern.py` si `kern.py`.

Licenta: cod SMT MIT, greutati MIT, dataset GrandStaff MIT. E singura combinatie
curata pentru un produs comercial dintre cele evaluate (§3.6).
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.recognition.engine import PageImage, PageResult, StaffCrop
from app.staff_detection.staves import (
    SYSTEM_GAP,
    bounding_box,
    detect_boxes,
    group,
)

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "SMT"
JAZZMUS_VENDOR = ROOT / "vendor" / "ISMIR-Jazzmus"
YOLO_WEIGHTS = JAZZMUS_VENDOR / "yolo_weigths" / "yolov11s_20241108.pt"

# Greutatile implicite. Nu sunt singurele publicate: `vendor/SMT/README.md`
# listeaza si `antoniorv6/smt-camera-grandstaff`, antrenat pe **poze** de
# partituri in loc de randari curate, si `antoniorv6/smt-string-quartets`.
# Materialul nostru e o fotocopie, deci varianta „camera" nu e o curiozitate;
# de asta repo-ul se poate schimba din afara, iar cifra decide.
REPO = "PRAIG/smt-grandstaff"

# Cateva zeci de pixeli in jurul sistemului: taietura lipita de portativ pierde
# liniile suplimentare si codite, iar modelul le-a vazut la antrenare.
SYSTEM_PADDING = 4

# Peste cat de multa comprimare orizontala merita sa ne plangem. Acelasi prag ca
# la portative in `smt_engine.py`.
SQUASH_WARNING_RATIO = 1.15

# Cate portative are un sistem „intreg" — grand staff de pian la antrenare,
# partitura stransa corala la noi. Serveste ca unitate de scara, nu ca cerinta:
# un sistem cu un singur portativ se redimensioneaza la jumatate din inaltime.
STAVES_PER_SYSTEM = 2


def resize_for_model(
    system: np.ndarray, staves: int, maxh: int, maxw: int
) -> tuple[np.ndarray, float]:
    """Sistemul, adus la scara pe care a vazut-o modelul la antrenare.

    Sta separat de motor fiindca **antrenarea trebuie sa scaleze exact la fel ca
    inferenta**. Daca cele doua ar diverge, modelul ar invata pe imagini de alta
    marime decat cele pe care le primeste apoi, iar diferenta n-ar da nicio
    eroare — doar o transcriere mai proasta, fara cauza vizibila. O singura
    functie, doi apelanti.

    Intoarce imaginea si **cat de mult s-a turtit orizontal**: peste `maxw`,
    codificarea pozitionala 2D nu mai are pozitii, deci latimea se taie.
    """
    import cv2

    height = min(int(maxh * max(staves, 1) / STAVES_PER_SYSTEM), maxh)
    natural = int(np.ceil(system.shape[1] * height / system.shape[0]))
    width = min(natural, maxw)
    resized = cv2.resize(system, (width, height), interpolation=cv2.INTER_CUBIC)
    return resized, natural / max(width, 1)


def _ensure_vendor_on_path() -> None:
    for path in (VENDOR, JAZZMUS_VENDOR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))


def implementation_for(weights) -> str:
    """Care implementare SMT se potriveste cu greutatile astea.

    Exista **doua**, si nu difera prin arhitectura, ci prin nume: oficiala
    (`vendor/SMT`) scrie `cross_attn`, `ffn`, `vocab_projection`, iar forkul
    JAZZMUS (`vendor/ISMIR-Jazzmus`) scrie `cross_attention`, `ffNet`,
    `out_layer`. Greutatile publicate sunt impartite intre ele: `PRAIG/*` sunt
    oficiale, `antoniorv6/*` sunt din fork.

    Hotararea se ia dupa **numele tensorilor**, nu dupa numele depozitului — la
    fel ca `detect_layout`, care o ia mai tarziu dupa modulele modelului.
    Greutatile se pot muta dintr-un depozit in altul; un tensor numit
    `cross_attention.lq` nu poate fi din implementarea oficiala.
    """
    keys = "\n".join(weights)
    if ".cross_attention." in keys:
        return "jazzmus"
    if ".cross_attn." in keys:
        return "official"
    raise ValueError(
        "greutatile nu seamana cu niciuna dintre cele doua implementari SMT: "
        "nu au nici `cross_attention`, nici `cross_attn`"
    )


def _build_official(config_repo: str, weights):
    from smt_model.configuration_smt import SMTConfig
    from smt_model.modeling_smt import SMTModelForCausalLM

    # Construim si incarcam manual, in loc de `from_pretrained`: mecanismul de
    # incarcare din transformers 5 asteapta atribute pe care un model scris
    # pentru 4.x nu le are, si fiecare ocolit l-ar scoate la iveala pe urmatorul.
    config = SMTConfig.from_pretrained(config_repo)
    model = SMTModelForCausalLM(config)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    return model, config, missing, unexpected


def _build_jazzmus(config_repo: str, weights):
    """Aceeasi constructie, cu clasele forkului JAZZMUS.

    Corectia de dimensiuni e cea pe care o face si `predict.py` din repo-ul lor,
    si e reprodusa la fel in `scripts/benchmark_smt.py`: greutatile salvate au
    `embedding` si `out_layer` de marimi diferite fata de config.
    """
    from torch.nn import Conv1d

    from jazzmus.model.smt.configuration_smt import SMTConfig
    from jazzmus.model.smt.modeling_smt import SMTModelForCausalLM

    config = SMTConfig.from_pretrained(config_repo)
    embed_size = weights["decoder.embedding.weight"].shape[0]
    out_size = weights["decoder.out_layer.weight"].shape[0]
    config.out_categories = embed_size
    model = SMTModelForCausalLM(config)
    if embed_size != out_size:
        model.decoder.out_layer = Conv1d(config.d_model, out_size, kernel_size=1)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    return model, config, missing, unexpected


def load_grandstaff(device, repo: str = REPO):
    """Incarca greutatile cu implementarea careia ii apartin.

    Vezi `implementation_for`: exista doua implementari SMT cu aceeasi
    arhitectura si nume diferite, iar greutatile publicate sunt impartite intre
    ele. Alegerea gresita da 210 tensori lipsa si 210 in plus — masurat pe
    `antoniorv6/smt-camera-grandstaff`.

    Intoarce (model, config).
    """
    _ensure_vendor_on_path()

    from transformers import PretrainedConfig

    # Repo-ul SMT e scris pentru transformers 4.x. In 5.x, `PreTrainedModel`
    # cere un atribut pe care `SMTConfig` nu-l are, si constructia crapa
    # inainte sa apuce sa incarce ceva.
    if not hasattr(PretrainedConfig, "_experts_implementation_internal"):
        PretrainedConfig._experts_implementation_internal = None

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights = load_file(hf_hub_download(repo_id=repo, filename="model.safetensors"))
    build = _build_official if implementation_for(weights) == "official" else _build_jazzmus
    model, config, missing, unexpected = build(repo, weights)
    if missing or unexpected:
        raise ValueError(
            f"greutatile nu se potrivesc cu modelul: {len(missing)} tensori lipsa, "
            f"{len(unexpected)} in plus — o transcriere facuta asa n-ar insemna nimic"
        )

    model.to(device)
    # Codificarile pozitionale sunt buffere ne-persistente: `.to()` nu le muta.
    for module in model.modules():
        if hasattr(module, "pe") and hasattr(module.pe, "to"):
            module.pe = module.pe.to(device)
    model.eval()
    return model, config


class GrandstaffEngine:
    """Recunoastere cu `smt-grandstaff`, un sistem odata."""

    name = "grandstaff"

    def __init__(
        self,
        threads: int = 1,
        max_tokens: int | None = None,
        gap: int = SYSTEM_GAP,
        staves_per_system: int | Sequence[int] | None = None,
        clean: bool = False,
        repo: str = REPO,
        voices_per_staff: int | None = None,
        strict_structure: bool = False,
        margin_spaces: float | None = None,
        layout: Sequence[Sequence[Sequence[str]]] | None = None,
        weights: Path | None = None,
        beam: int = 1,
    ) -> None:
        # Cate ipoteze tine decodarea in paralel (`IncrementalSMT.generate_beam`); 1 = lacoma,
        # decodarea dupa care sunt masurate toate reperele de pana pe 2026-10-04.
        self.beam = beam
        # Un singur thread e de ~5x mai rapid aici: operatiile modelului sunt
        # mici, iar sincronizarea intre thread-uri costa mai mult decat calculul.
        self.threads = threads
        self.max_tokens = max_tokens
        self.gap = gap
        # `None` = gruparea se deduce din spatiere, potrivit pentru tiparit.
        # Un numar = se declara, fiindca pe scris de mana cu versuri spatierea
        # nu poarta semnalul (§14.5). O lista = se declara pe fiecare sistem in
        # parte, cand numarul de portative nu e acelasi peste toata pagina.
        self.staves_per_system = staves_per_system
        # Ce voci stau pe fiecare portativ, de sus in jos. `None` = partitura
        # stransa corala, repartizata dupa cheie. Se declara fiindca pe o
        # cantare cu solo cheia nu mai poate deosebi portativul de solo de cel
        # de sopran — amandoua in cheia Sol (§15.10).
        self.layout = layout
        # Etapa 6, implicit oprita: pe partituri tiparite curate nu are ce
        # repara, iar orice pas care sterge cerneala e mai bine sa fie cerut
        # decat presupus.
        self.clean = clean
        # Care greutati. Vocabularul vine cu ele (`model.w2i`), deci schimbarea
        # repo-ului schimba si vocabularul, fara nimic de facut in `bekern.py`.
        self.repo = repo
        # Cate voci poarta un portativ. Declarat, ca tot ce nu se poate citi
        # din imagine (§14.6): pe partitura stransa corala sunt doua. Cand e
        # dat, decodarea nu mai poate deschide voci peste plafon.
        self.voices_per_staff = voices_per_staff
        # Decodarea nu mai are voie sa scrie un tabel care nu se inchide.
        # Implicit oprita — vezi `_grammar_for` pentru cifrele care explica
        # de ce nu e o imbunatatire curata.
        self.strict_structure = strict_structure
        # Cat de larg se pastreaza in jurul portativului cand se sterge
        # textul. `None` = valoarea din `app/preprocessing/text.py`; exista
        # ca parametru ca sa poata fi maturata, nu ghicita (§15.6).
        self.margin_spaces = margin_spaces
        # Un checkpoint de la fine-tuning (etapa 11). Contine **doar**
        # parametrii antrenati, deci se aplica peste greutatile publicate,
        # cu `strict=False`. Fara optiunea asta, o antrenare n-ar putea fi
        # masurata: cifra se ia rerulland conducta cu greutatile noi.
        self.weights = weights
        self.cleaned = None
        self._model = None
        self._config = None
        self._decoder = None
        self._yolo = None

    # -- incarcare lenesa, o singura data --------------------------------

    def load(self) -> None:
        if self._model is not None:
            return

        _ensure_vendor_on_path()
        import torch
        from ultralytics import YOLO

        from app.recognition.incremental import IncrementalSMT

        torch.set_num_threads(self.threads)
        model, config = load_grandstaff(torch.device("cpu"), self.repo)
        if self.weights is not None:
            state = torch.load(self.weights, map_location="cpu", weights_only=False)
            trained = state["model"] if "model" in state else state
            missing, unexpected = model.load_state_dict(trained, strict=False)
            if unexpected:
                raise ValueError(
                    f"{self.weights}: {len(unexpected)} parametri care nu exista in "
                    f"model (primul — {unexpected[0]}); checkpointul nu e al acestei "
                    "arhitecturi"
                )
            print(
                f"greutati de la fine-tuning: {len(trained)} tensori din "
                f"{self.weights}, pasul {state.get('step', '?')}"
            )
        self._model = model
        self._config = config
        self._decoder = IncrementalSMT(model)
        self._yolo = YOLO(str(YOLO_WEIGHTS))

    # -- etapele --------------------------------------------------------

    def detect_systems(
        self, image_path: Path
    ) -> list[tuple[tuple[int, int, int, int], np.ndarray, int]]:
        """Decupeaza sistemele, de sus in jos.

        Intoarce si **cate portative** are fiecare sistem: de asta depinde
        inaltimea la care se redimensioneaza. Un sistem poate avea si unul
        singur — pe prima pagina din „Iata-L pe Hristos", T&B intra abia la
        masura 6.
        """
        from PIL import Image

        page = np.array(Image.open(image_path).convert("L"))

        # Curatarea **inainte** de detectie, nu dupa: indreptarea roteste
        # pagina, iar cutii masurate pe imaginea stramba n-ar mai cadea peste
        # muzica din cea dreapta. Tot ce urmeaza lucreaza in acelasi sistem de
        # coordonate.
        self.cleaned = None
        if self.clean:
            from app.preprocessing import clean_page

            self.cleaned = (
                clean_page(page, self.margin_spaces)
                if self.margin_spaces is not None
                else clean_page(page)
            )
            page = self.cleaned.image

        # YOLO primeste imaginea, nu calea, tocmai ca sa vada ce vedem si noi.
        source = np.repeat(page[:, :, None], 3, axis=2)
        systems = group(
            detect_boxes(self._yolo, source), self.staves_per_system, self.gap
        )

        height, width = page.shape[:2]
        crops = []
        for boxes in systems:
            x1, y1, x2, y2 = bounding_box(boxes, SYSTEM_PADDING, (width, height))
            crops.append(((x1, y1, x2, y2), page[y1:y2, x1:x2], len(boxes)))
        return crops

    def _preprocess(self, system: np.ndarray, staves: int) -> tuple[object, np.ndarray, float]:
        """Aduce sistemul la scara pe care a vazut-o modelul la antrenare.

        `prepare_data` din `vendor/SMT/data.py` scaleaza imaginile cu un raport
        fix, cu inaltimea ridicata intai la cel putin 256:
        `height = ceil(max(h, 256) * reduce_ratio)`, cu `reduce_ratio` 0.5
        pentru GrandStaff. Un grand staff de pian are **doua** portative, deci
        cele ~256 px se impart intre ele: aproximativ 128 px de portativ.

        Ce conteaza pentru model nu e inaltimea imaginii, ci **cat de mare e un
        portativ in ea**. De asta scalam dupa numarul de portative, nu la o
        inaltime fixa. Masurat pe „Iata-L pe Hristos": primul sistem are un
        singur portativ (T&B intra mai tarziu); dus la 256 px, portativul ajunge
        de doua ori mai mare decat orice a vazut modelul la antrenare, iar
        transcrierea lui a iesit un sir de pauze. Sistemele de doua portative,
        la aceeasi inaltime, au iesit corecte.

        Latimea se taie la `maxw`: peste atat, codificarea pozitionala 2D nu
        mai are pozitii. Intoarcem si cat de mult s-a turtit orizontal, ca sa se
        vada in raport — un sistem turtit e citit mai prost, si vrem sa stim ca
        de acolo vine problema, nu din model.
        """
        resized, squash = resize_for_model(
            system, staves, int(self._config.maxh), int(self._config.maxw)
        )
        return self._to_tensor(resized), resized, squash

    @staticmethod
    def _to_tensor(image: np.ndarray):
        """Aceeasi conversie ca `convert_img_to_tensor` din amandoua repo-urile:
        tonuri de gri, valori in [0, 1], forma (1, H, W)."""
        import torch

        return torch.from_numpy(image).float().div(255.0).unsqueeze(0)

    def _grammar_for(self, staves: int):
        """Structura pe care documentul sistemului trebuie s-o respecte, daca s-a cerut.

        Regula de baza nu presupune nimic despre pagina: **antetul hotaraste**
        cate coloane are documentul, iar randurile de dupa trebuie sa aiba fix
        atatea. E consistenta documentului cu el insusi, nu o asteptare a
        noastra.

        **Si totusi e implicit oprita.** Masurat pe pagina 3 (§15.6): randurile
        stricate dispar cu totul, 59 → 0, iar notele inventate scad de la 200 la
        72 — dar notele *corecte* scad si ele, de la 20 la 16. Constransa,
        decodarea gasita cu argmax o ia pe alt drum, si nu totdeauna pe unul mai
        bun. O schimbare care imbunatateste o cifra si o strica pe cealalta nu
        se ia pe o singura pagina, deci se cere, ca `--curata`.

        Plafonul de voci — `staves * voices_per_staff` — se declara separat, si
        aduce cu el consistenta randurilor: nu are sens sa numeri coloane fara
        ca randurile sa le respecte.
        """
        from app.recognition.bekern import Grammar

        if not (self.strict_structure or self.voices_per_staff):
            return None
        return Grammar(
            max_columns=staves * self.voices_per_staff if self.voices_per_staff else None,
        )

    # -- interfata Engine ------------------------------------------------

    def transcribe(self, page: PageImage) -> PageResult:
        self.load()
        from app.recognition.bekern import Grammar, to_kern

        started = time.perf_counter()
        result = PageResult(page=page)

        squashed = 0
        # Cate portative are fiecare sistem, pastrat pentru asamblare: e singura
        # cale prin care partea determinista poate observa ca modelul a inventat
        # un portativ.
        counts: list[int] = []
        for index, (box, system, staves) in enumerate(self.detect_systems(page.path), 1):
            counts.append(staves)
            tensor, resized, squash = self._preprocess(system, staves)
            if squash >= SQUASH_WARNING_RATIO:
                squashed += 1
            t0 = time.perf_counter()
            tokens = self._decoder.generate(
                tensor, max_tokens=self.max_tokens, grammar=self._grammar_for(staves),
                beam=self.beam,
            )
            conversion = to_kern(tokens)
            if conversion.unknown:
                result.problems.append(
                    f"sistemul {index}: {len(conversion.unknown)} marcaje "
                    f"netraduse ({', '.join(sorted(set(conversion.unknown)))})"
                )
            result.staves.append(
                StaffCrop(
                    index=index,
                    box=box,
                    image=resized,
                    kern=conversion.kern,
                    tokens=len(tokens),
                    taiat=len(tokens) >= self._decoder.plafon(self.max_tokens),
                    seconds=round(time.perf_counter() - t0, 2),
                )
            )

        if result.taiate:
            result.problems.append(
                f"{len(result.taiate)} sisteme au atins plafonul de decodare "
                f"({self._decoder.plafon(self.max_tokens)} tokeni) si sunt taiate "
                f"la mijloc: {', '.join(str(i) for i in result.taiate)}. "
                "Ce s-a citit din ele nu e o pagina scurta, e o pagina rupta"
            )

        if squashed:
            result.problems.append(
                f"{squashed} sisteme au fost comprimate orizontal ca sa incapa in "
                f"{int(self._config.maxw)} px — transcrierea lor poate fi mai slaba"
            )

        if not result.staves:
            result.problems.append("nu s-a detectat niciun portativ")
            result.seconds = round(time.perf_counter() - started, 2)
            return result

        self._assemble(result, counts)
        result.seconds = round(time.perf_counter() - started, 2)
        return result

    def _assemble(self, result: PageResult, staves: list[int] | None = None) -> None:
        """Sisteme -> partitura -> validare.

        Esecul de aici nu arunca: `**kern`-ul brut si decupajele raman in
        rezultat si sunt exact ce trebuie ca sa intelegem de ce a esuat.
        """
        from app.recognition.page import assemble_systems
        from app.validation.rules import Severity, validate

        page = assemble_systems([staff.kern for staff in result.staves],
                                title=result.page.stem, staves=staves,
                                layout=self.layout)
        result.score = page.score
        result.systems = page.systems
        result.problems.extend(page.problems)

        findings = validate(page.score)
        result.errors = [str(f) for f in findings if f.severity is Severity.ERROR]
        result.warnings = [str(f) for f in findings if f.severity is not Severity.ERROR]
