"""`IncrementalSMT.generate_beam`: cand decodarea lacoma alege prost, beam search gaseste mai bine.

Modelul e inlocuit cu un lant Markov de patru tokeni, ca testul sa nu incarce greutati.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from app.recognition.incremental import IncrementalSMT, Memory  # noqa: E402

W2I = {"<bos>": 0, "<eos>": 1, "a": 2, "b": 3}
# Dupa fiecare token, probabilitatile urmatorului (doar ultimul token conteaza).
URMATORUL = {
    0: {2: 0.6, 3: 0.4},  # <bos>: „a" pare mai bun...
    2: {2: 0.34, 3: 0.33, 1: 0.33},  # ...dar dupa „a" modelul nu se mai hotaraste
    3: {1: 0.99},  # dupa „b" se termina sigur
}


def _model(limita=12):
    smt = object.__new__(IncrementalSMT)
    smt.w2i = W2I
    smt.i2w = {i: w for w, i in W2I.items()}
    smt.maxlen = limita

    def encode(_imagine):
        return Memory(layers=[])

    def step(memory, token):
        memory.length += 1
        logits = torch.full((1, len(W2I)), -1e9)
        for urmator, p in URMATORUL[token].items():
            logits[0, urmator] = math.log(p)
        return logits

    smt.encode = encode
    smt.step = step
    return smt


def test_lacom_se_blocheaza_in_bucla():
    tokeni = _model().generate(torch.zeros(1))
    assert tokeni[:3] == ["a", "a", "a"]


def test_beam_gaseste_secventa_mai_probabila():
    assert _model().generate(torch.zeros(1), beam=3) == ["b"]


def test_beam_unu_e_decodarea_lacoma():
    assert _model().generate(torch.zeros(1), beam=1) == _model().generate(torch.zeros(1))
