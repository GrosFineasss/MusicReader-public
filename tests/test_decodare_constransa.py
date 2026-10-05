"""Masca de decodare: tokenii care ar strica structura devin imposibili.

Testul nu incarca niciun model. `generate` are nevoie doar de `encode`, `step`
si vocabular, deci le punem noi: asa se verifica **exact** ce face masca, in
milisecunde, si separat de ce ar raspunde modelul adevarat.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ISMIR-Jazzmus"))

torch = pytest.importorskip("torch")

from app.recognition.bekern import COLUMN, LINE, Grammar  # noqa: E402
from app.recognition.incremental import IncrementalSMT  # noqa: E402

VOCAB = ["<bos>", "<eos>", COLUMN, LINE, "**ekern_1.0", "4c", "*^", "*"]


class FakeDecoder(IncrementalSMT):
    """Un decoder care vrea mereu acelasi token, ca sa se vada ce-l opreste."""

    def __init__(self, wanted: str) -> None:
        self.w2i = {word: index for index, word in enumerate(VOCAB)}
        self.i2w = {index: word for word, index in self.w2i.items()}
        self.maxlen = 12
        self._wanted = wanted
        self.asked: list[tuple[str, ...]] = []

    def encode(self, image_tensor):
        return None

    def step(self, memory, token):
        logits = torch.full((1, len(VOCAB)), -1.0)
        logits[0, self.w2i[self._wanted]] = 10.0
        # al doilea favorit, ca sa existe unde cadea cand primul e interzis
        logits[0, self.w2i["4c"]] = 5.0
        return logits


def test_fara_gramatica_nimic_nu_se_schimba():
    decoder = FakeDecoder(LINE)
    assert decoder.generate(None, max_tokens=4) == [LINE] * 3


def test_randul_neterminat_nu_se_poate_inchide():
    """Modelul vrea `<b>` de la primul camp; documentul are doua coloane."""
    decoder = FakeDecoder(LINE)
    tokens = decoder.generate(None, max_tokens=4, grammar=Grammar(spines=2))
    assert tokens[0] != LINE


def test_dupa_ce_randul_e_plin_tabul_devine_imposibil():
    decoder = FakeDecoder(COLUMN)
    tokens = decoder.generate(None, max_tokens=6, grammar=Grammar(spines=2))
    # primul `<t>` e legitim (trece de la campul 0 la 1), al doilea nu
    assert tokens.count(COLUMN) <= 1


def test_plafonul_de_voci_opreste_desfacerea():
    decoder = FakeDecoder("*^")
    grammar = Grammar(columns=4, max_columns=4)
    assert "*^" not in decoder.generate(None, max_tokens=3, grammar=grammar)


def test_gramatica_vede_tokenii_alesi():
    """Starea trebuie sa avanseze, altfel masca ar ramane blocata pe primul camp."""
    decoder = FakeDecoder("4c")
    grammar = Grammar(spines=2)
    decoder.generate(None, max_tokens=3, grammar=grammar)
    assert grammar.field_index == 0
