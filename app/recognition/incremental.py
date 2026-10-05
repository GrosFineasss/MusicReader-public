"""Decodare incrementala pentru SMT, cu KV cache.

De ce exista acest modul
------------------------
`SMTModelForCausalLM.predict()` din implementarea de referinta re-ruleaza
intreg decoderul peste tot prefixul, la fiecare token generat. Doua consecinte,
amandoua masurate pe acest laptop:

  * proiectiile K/V ale memoriei encoderului (496 pozitii x 256 canale, in
    fiecare din cele 8 straturi) se recalculeaza la fiecare pas, desi encoderul
    a rulat o singura data si rezultatul lui nu se schimba. Cost fix irosit:
    ~400 ms per token.
  * self-attention-ul reproceseaza pozitiile deja generate, deci costul pe pas
    creste cu lungimea prefixului. Total patratic.

Masurat pe un portativ de 1000 px: ~745 s. Estimare pentru un sistem SATB:
~50 de minute. Neviabil.

Modelul e cauzal (masca de atentie e triunghiulara), deci iesirea fiecarui
strat la pozitiile deja generate **nu se schimba** cand adaugam un token nou.
Asta face un KV cache corect, nu doar o aproximare.

Doua implementari, nu una
-------------------------
`jazzmus-model` si `smt-grandstaff` sunt amandoua SMT, dar vin din versiuni
diferite ale codului: difera numele modulelor **si aritmetica**. Diferentele de
nume dau erori zgomotoase; cele de aritmetica dau tokeni plauzibili si gresiti.
Sunt stranse toate intr-un `Layout`, ales o singura data, la constructie.

Fidelitate
----------
Pentru forkul JAZZMUS, codul reproduce exact aritmetica din `MHA.forward`,
inclusiv o particularitate a implementarii: **produsul scalar nu e impartit la
sqrt(head_dim)**. `MHA` isi calculeaza `scale_factor`, dar nu il foloseste
niciodata. Corectarea acestei "greseli" ar schimba iesirea modelului fata de
greutatile antrenate, deci o pastram intentionat. Implementarea oficiala **isi
foloseste** scalarea, si de asta `Layout` o poarta ca valoare, nu ca presupunere.

Echivalenta cu implementarea de referinta e verificata, pentru amandoua
modelele, in `tests/test_incremental_decode.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange


# --------------------------------------------------------------------------
# Ce difera intre cele doua implementari
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Unde stau piesele modelului si cum se calculeaza atentia.

    Numele se folosesc **o singura data**, la constructie: `step()` lucreaza
    numai cu module deja rezolvate. Un `getattr` per token ar costa mai mult
    decat aritmetica in sine, la tensorii astia mici.
    """

    name: str
    self_attn: str
    cross_attn: str
    projections: tuple[str, str, str]  # q, k, v
    norms: tuple[str, str, str] | str  # trei atribute, sau o `ModuleList`
    ffn: str
    head_dim: str
    positional_2d: str
    positional_1d: str
    out_layer: str
    relu_before_out: bool
    scaled_attention: bool
    # `True` daca logits vin ca (batch, pozitie, vocabular). Forkul JAZZMUS le
    # da transpuse, (batch, vocabular, pozitie) — conteaza pentru cine compara
    # cu implementarea de referinta.
    logits_last_dim: bool


JAZZMUS = Layout(
    name="jazzmus",
    self_attn="input_attention",
    cross_attn="cross_attention",
    projections=("lq", "lk", "lv"),
    norms=("norm1", "norm2", "norm3"),
    ffn="ffNet",
    head_dim="head_dim",
    positional_2d="positional_2D",
    positional_1d="positional_1D",
    out_layer="out_layer",
    relu_before_out=True,
    scaled_attention=False,
    logits_last_dim=False,
)

OFFICIAL = Layout(
    name="official",
    self_attn="self_attn",
    cross_attn="cross_attn",
    projections=("q_proj", "k_proj", "v_proj"),
    norms="norm_layers",
    ffn="ffn",
    head_dim="d_head",
    positional_2d="pos2D",
    positional_1d="position_encoding",
    out_layer="vocab_projection",
    relu_before_out=False,
    scaled_attention=True,
    logits_last_dim=True,
)


def detect_layout(model) -> Layout:
    """Alege profilul dupa modulele pe care le are modelul.

    Dupa structura, nu dupa numele repo-ului: greutatile se pot muta dintr-un
    depozit in altul, dar un `DecoderLayer` care are `input_attention` nu poate
    fi cel oficial.
    """
    layer = model.decoder.decoder.layers[0]
    if hasattr(layer, JAZZMUS.self_attn):
        return JAZZMUS
    if hasattr(layer, OFFICIAL.self_attn):
        return OFFICIAL
    raise ValueError(
        "structura de decoder necunoscuta: nu are nici "
        f"`{JAZZMUS.self_attn}`, nici `{OFFICIAL.self_attn}`"
    )


@dataclass(frozen=True)
class LayerModules:
    """Modulele unui strat, rezolvate o data si folosite la fiecare token."""

    self_q: Any
    self_k: Any
    self_v: Any
    self_out: Any
    cross_q: Any
    cross_k: Any
    cross_v: Any
    cross_out: Any
    norm1: Any
    norm2: Any
    norm3: Any
    ffn: Any


def _resolve(layer, layout: Layout) -> LayerModules:
    self_attn = getattr(layer, layout.self_attn)
    cross_attn = getattr(layer, layout.cross_attn)
    q, k, v = layout.projections
    if isinstance(layout.norms, str):
        norms = getattr(layer, layout.norms)
    else:
        norms = [getattr(layer, name) for name in layout.norms]
    return LayerModules(
        self_q=getattr(self_attn, q),
        self_k=getattr(self_attn, k),
        self_v=getattr(self_attn, v),
        self_out=self_attn.out_proj,
        cross_q=getattr(cross_attn, q),
        cross_k=getattr(cross_attn, k),
        cross_v=getattr(cross_attn, v),
        cross_out=cross_attn.out_proj,
        norm1=norms[0],
        norm2=norms[1],
        norm3=norms[2],
        ffn=getattr(layer, layout.ffn),
    )


# --------------------------------------------------------------------------
# Starea pastrata intre tokeni
# --------------------------------------------------------------------------


@dataclass
class LayerCache:
    """Starea pastrata pentru un strat de decoder.

    `cross_kt` e pastrat deja transpus si contiguu: `bmm` peste o vedere
    transpusa costa la fel de mult ca inmultirea in sine cand tensorii sunt
    mici, iar aici transpunerea ar fi refacuta la fiecare token.
    """

    cross_kt: torch.Tensor  # (heads, head_dim, memorie) — fix pe tot portativul
    cross_v: torch.Tensor  # (heads, memorie, head_dim)
    qkv_weight: torch.Tensor  # q/k/v ale self-attention, concatenate
    qkv_bias: torch.Tensor
    self_kt: torch.Tensor | None = None  # (heads, head_dim, pozitii) — creste
    self_v: torch.Tensor | None = None


@dataclass
class Memory:
    """Rezultatul encoderului, pregatit pentru decodare."""

    layers: list[LayerCache]
    length: int = 0  # cati tokeni s-au generat pana acum

    def reset(self) -> None:
        for layer in self.layers:
            layer.self_kt = None
            layer.self_v = None
        self.length = 0


def _to_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """(L, b, c) -> (b*heads, L, head_dim), exact ca in `MHA.forward`.

    Implementarea oficiala imparte capetele pornind de la (b, L, c), dar pe
    aceleasi felii de canale: pentru b=1 rezultatul e identic, cu conditia ca
    memoria sa fie adusa intai la forma (L, b, c). Vezi `encode`.
    """
    length, batch, _ = x.size()
    return torch.reshape(x, (length, batch * heads, head_dim)).transpose(0, 1)


def _attend(
    q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor, out_proj, scale: float
) -> torch.Tensor:
    """Atentie fara masca, cu o singura pozitie de interogare.

    Nu e nevoie de masca: cache-ul contine exact pozitiile pe care masca
    cauzala le-ar fi permis. `kt` vine deja transpus.

    `scale` e 1.0 pentru forkul JAZZMUS: vezi nota de fidelitate din capul
    fisierului.
    """
    scores = torch.bmm(q, kt)
    if scale != 1.0:
        scores = scores * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.bmm(weights, v)  # (heads, 1, head_dim)
    return out_proj(out.transpose(0, 1).reshape(1, 1, -1))


class IncrementalSMT:
    """Genereaza tokeni pentru un portativ, reutilizand tot ce se poate reutiliza."""

    def __init__(self, model, layout: Layout | None = None) -> None:
        self.model = model
        self.layout = layout or detect_layout(model)
        self.decoder = model.decoder
        self.layers = [
            _resolve(layer, self.layout) for layer in model.decoder.decoder.layers
        ]

        attention = getattr(model.decoder.decoder.layers[0], self.layout.self_attn)
        self.heads = attention.num_heads
        self.head_dim = getattr(attention, self.layout.head_dim)
        self.scale = self.head_dim ** -0.5 if self.layout.scaled_attention else 1.0

        self.maxlen = int(model.maxlen)
        self.w2i = model.w2i
        self.i2w = model.i2w

        self._positional_2d = getattr(model, self.layout.positional_2d)

        # Codificarea pozitionala 1D, pregatita ca `_pe[pozitie]` sa fie direct
        # vectorul de adunat, in forma (1, 1, d_model). Forkul JAZZMUS o tine
        # ca (1, d_model, maxlen), implementarea oficiala ca (maxlen, d_model).
        pe = getattr(model.decoder, self.layout.positional_1d).pe
        if pe.dim() == 3:
            pe = pe[0].t()
        self._pe = pe.contiguous().view(pe.size(0), 1, 1, pe.size(1))

        # Stratul de iesire e un `Linear` in implementarea oficiala si un
        # `Conv1d` cu kernel 1 in fork — adica tot un `Linear`, deghizat.
        out_layer = getattr(model.decoder, self.layout.out_layer)
        weight = out_layer.weight
        if weight.dim() == 3:
            weight = weight.squeeze(-1)
        self._out_weight = weight.t().contiguous()
        self._out_bias = out_layer.bias
        self._end_relu = model.decoder.end_relu if self.layout.relu_before_out else None

    @torch.no_grad()
    def encode(self, image_tensor: torch.Tensor) -> Memory:
        """Ruleaza encoderul si pre-proiecteaza memoria pentru fiecare strat.

        Aici se plateste, o singura data, costul care in implementarea de
        referinta se platea la fiecare token.
        """
        encoder_output = self.model.forward_encoder(image_tensor.unsqueeze(0))
        positional = self._positional_2d(encoder_output)

        # Amandoua implementarile aplatizeaza (b, c, h, w) in (b, h*w, c); noi o
        # ducem la (h*w, b, c), forma pe care o asteapta `_to_heads`. Pentru
        # b=1 feliile de canale ies identice, deci si capetele de atentie.
        raw = rearrange(encoder_output, "b c h w -> (h w) b c")
        enhanced = rearrange(positional, "b c h w -> (h w) b c")

        caches = []
        for modules in self.layers:
            # Cheia vine din memoria cu pozitionare, valoarea din cea bruta —
            # aceeasi separare ca in `DecoderLayer.forward`, in amandoua
            # implementarile.
            k = _to_heads(modules.cross_k(enhanced), self.heads, self.head_dim)
            v = _to_heads(modules.cross_v(raw), self.heads, self.head_dim)

            # Cele trei proiectii ale self-attention-ului pornesc din acelasi
            # tensor, deci le facem intr-un singur matmul. Pe tensori de o
            # pozitie, costul unei operatii e dominat de dispatch, nu de
            # aritmetica: trei operatii mici sunt mai scumpe decat una mare.
            qkv_weight = torch.cat(
                [modules.self_q.weight, modules.self_k.weight, modules.self_v.weight]
            ).contiguous()
            qkv_bias = torch.cat(
                [modules.self_q.bias, modules.self_k.bias, modules.self_v.bias]
            ).contiguous()

            caches.append(
                LayerCache(
                    cross_kt=k.transpose(1, 2).contiguous(),
                    cross_v=v.contiguous(),
                    qkv_weight=qkv_weight,
                    qkv_bias=qkv_bias,
                )
            )
        return Memory(layers=caches)

    @torch.no_grad()
    def step(self, memory: Memory, token: int) -> torch.Tensor:
        """Consuma un token si intoarce logits pentru pozitia urmatoare."""
        position = memory.length
        heads, head_dim = self.heads, self.head_dim

        x = self.decoder.embedding.weight[token].view(1, 1, -1)
        x = x + self._pe[position]

        for modules, cache in zip(self.layers, memory.layers):
            qkv = torch.addmm(cache.qkv_bias, x.view(1, -1), cache.qkv_weight.t())
            q, k_new, v_new = qkv.view(3, heads, 1, head_dim).unbind(0)

            cache.self_kt = (
                k_new.transpose(1, 2)
                if cache.self_kt is None
                else torch.cat([cache.self_kt, k_new.transpose(1, 2)], dim=2)
            )
            cache.self_v = v_new if cache.self_v is None else torch.cat([cache.self_v, v_new], dim=1)

            attended = _attend(q, cache.self_kt, cache.self_v, modules.self_out, self.scale)
            x = modules.norm1(x + attended)

            q = _to_heads(modules.cross_q(x), heads, head_dim)
            attended = _attend(q, cache.cross_kt, cache.cross_v, modules.cross_out, self.scale)
            x = modules.norm2(x + attended)

            x = modules.norm3(x + modules.ffn(x))

        memory.length += 1
        activated = self._end_relu(x) if self._end_relu is not None else x
        return torch.addmm(self._out_bias, activated.view(1, -1), self._out_weight)

    @torch.no_grad()
    def plafon(self, max_tokens: int | None = None) -> int:
        """Cati tokeni poate scrie decodarea inainte sa fie taiata.

        Exista ca metoda, si nu ca expresie repetata, fiindca cine compara o
        lungime cu ea trebuie sa obtina **acelasi** numar ca bucla care taie.
        §15.36: o iesire care atinge plafonul nu e o masuratoare, e o margine —
        si arata exact ca orice alta valoare daca nimeni n-o marcheaza.
        """
        return min(max_tokens or self.maxlen, self.maxlen) - 1

    @torch.no_grad()
    def generate_beam(
        self,
        image_tensor: torch.Tensor,
        beam: int,
        max_tokens: int | None = None,
        grammar=None,
    ) -> list[str]:
        """Ca `generate`, dar cu `beam` ipoteze tinute in paralel; castiga media log-probabilitatii.

        Decodarea lacoma alege la fiecare pas tokenul cel mai probabil si nu se mai intoarce: o
        bara pusa cu un token prea devreme strica tot sistemul. Aici fiecare ipoteza isi are
        cache-ul ei de self-attention. Cache-urile se pot imparti fara copii: `step` le creste cu
        `torch.cat`, care face un tensor nou, deci o ipoteza nu o atinge pe alta. Memoria
        encoderului (cross-attention) e aceeasi pentru toate. Alegerea finala e pe log-probabilitatea
        **medie** pe token, altfel o ipoteza scurta, terminata devreme, ar castiga doar fiindca
        aduna mai putini termeni negativi.
        """
        import copy

        limit = self.plafon(max_tokens)
        bos, eos = self.w2i["<bos>"], self.w2i["<eos>"]

        def word_of(index: int) -> str:
            return self.i2w[str(index)] if str(index) in self.i2w else self.i2w[index]

        def fork(memory: Memory) -> Memory:
            return Memory(
                layers=[
                    LayerCache(
                        cross_kt=layer.cross_kt, cross_v=layer.cross_v,
                        qkv_weight=layer.qkv_weight, qkv_bias=layer.qkv_bias,
                        self_kt=layer.self_kt, self_v=layer.self_v,
                    )
                    for layer in memory.layers
                ],
                length=memory.length,
            )

        # (scor, tokeni fara <bos>, ultimul token, memorie, gramatica)
        alive = [(0.0, [], bos, self.encode(image_tensor), grammar)]
        finished: list[tuple[float, list[str]]] = []
        for _ in range(limit):
            candidates = []
            for score, words, last, memory, rules in alive:
                logits = self.step(memory, last).view(-1)
                if rules is not None:
                    for word in rules.forbidden():
                        index = self.w2i.get(word)
                        if index is not None:
                            logits[index] = float("-inf")
                logp = torch.log_softmax(logits, dim=-1)
                values, indices = torch.topk(logp, beam)
                for value, index in zip(values.tolist(), indices.tolist()):
                    if value == float("-inf"):
                        continue
                    candidates.append((score + value, words, index, memory, rules))
            candidates.sort(key=lambda c: c[0], reverse=True)
            alive = []
            for score, words, index, memory, rules in candidates:
                if index == eos:
                    finished.append((score / (len(words) + 1), words))
                    continue
                if len(alive) == beam:
                    continue
                word = word_of(index)
                if rules is not None:
                    rules = copy.deepcopy(rules)
                    rules.accept(word)
                alive.append((score, words + [word], index, fork(memory), rules))
            # Oprire cand s-au terminat `beam` ipoteze: media unei ipoteze vii mai poate urca (un
            # token aproape sigur trage media spre 0), deci „cea mai buna terminata o bate deja pe
            # cea mai buna vie" ar taia uneori castigatoarea.
            if not alive or len(finished) >= beam:
                break
        if not finished:  # plafonul atins: ca la `generate`, se intoarce ce s-a scris
            score, words, _, _, _ = max(alive, key=lambda a: a[0])
            return words
        return max(finished, key=lambda f: f[0])[1]

    def generate(
        self,
        image_tensor: torch.Tensor,
        max_tokens: int | None = None,
        grammar=None,
        beam: int = 1,
    ) -> list[str]:
        """Transcrie un portativ. Intoarce tokenii, fara `<bos>`/`<eos>`.

        `grammar` — un `bekern.Grammar` — poate **interzice** tokenii care ar
        strica structura de tabel a documentului, punandu-le logitul la minus
        infinit inainte de alegere. Nu schimba ce a vazut modelul; il impiedica
        doar sa scrie un tabel care nu se inchide, si pe care `kern.py` l-ar
        taia tacut. Fara `grammar`, decodarea e exact cea de pana acum.

        `beam` > 1 trece la `generate_beam`; 1 (implicit) e decodarea lacoma de pana acum.
        """
        if beam > 1:
            return self.generate_beam(image_tensor, beam, max_tokens, grammar)
        limit = self.plafon(max_tokens)
        memory = self.encode(image_tensor)

        tokens: list[str] = []
        current = self.w2i["<bos>"]
        for _ in range(limit):
            logits = self.step(memory, current)
            if grammar is not None:
                for word in grammar.forbidden():
                    index = self.w2i.get(word)
                    if index is not None:
                        logits.view(-1)[index] = float("-inf")
            current = int(torch.argmax(logits).item())
            word = self.i2w[str(current)] if str(current) in self.i2w else self.i2w[current]
            if word == "<eos>":
                break
            tokens.append(word)
            if grammar is not None:
                grammar.accept(word)
        return tokens
