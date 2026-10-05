"""Decoderul incremental trebuie sa dea exact ce da implementarea de referinta.

Testele sunt marcate `slow`: au nevoie de greutatile descarcate si de ~2 minute
doar ca sa incarce modelul. Nu ruleaza implicit.

    pytest -m slow
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ISMIR-Jazzmus"))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

pytestmark = pytest.mark.slow

WEIGHTS = ROOT / "models" / "hf" / "hub" / "models--JuanCarlosMartinezSevilla--jazzmus-model"


# `torch` se importa in fixtures, nu la nivel de modul: importul lui dureaza
# minute pe acest sistem, iar colectarea testelor rapide nu are de ce sa-l
# plateasca doar pentru ca fisierul exista.
@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def loaded(torch):
    if not WEIGHTS.exists():
        pytest.skip("greutatile jazzmus-model nu sunt descarcate")
    torch.set_num_threads(1)
    from scripts.benchmark_smt import load_model, synthetic_staff

    model, _ = load_model(torch.device("cpu"))
    from jazzmus.dataset.data_preprocessing import convert_img_to_tensor

    return model, convert_img_to_tensor(synthetic_staff())


@pytest.fixture(scope="module")
def comparison(loaded):
    from scripts.verify_incremental import compare

    model, _ = loaded
    from scripts.benchmark_smt import synthetic_staff

    worst, mismatches, slow_seconds, fast_seconds = compare(model, synthetic_staff(), steps=8)
    return worst, mismatches, slow_seconds, fast_seconds


class TestEquivalence:
    def test_logits_match_the_reference(self, comparison):
        worst, _, _, _ = comparison
        assert worst < 1e-3, f"diferenta maxima de logit: {worst:.3e}"

    def test_same_tokens_are_chosen(self, comparison):
        _, mismatches, _, _ = comparison
        assert mismatches == 0

    def test_incremental_is_faster(self, comparison):
        _, _, slow_seconds, fast_seconds = comparison
        assert fast_seconds < slow_seconds


class TestCacheBehaviour:
    def test_cross_attention_memory_is_computed_once(self, loaded, torch):
        """Cache-ul incrucisat nu se schimba pe masura ce generam tokeni."""
        from app.recognition.incremental import IncrementalSMT

        model, tensor = loaded
        fast = IncrementalSMT(model)
        memory = fast.encode(tensor)
        before = memory.layers[0].cross_kt.clone()

        token = model.w2i["<bos>"]
        for _ in range(3):
            token = int(torch.argmax(fast.step(memory, token)).item())

        assert torch.equal(before, memory.layers[0].cross_kt)

    def test_self_attention_cache_grows_by_one_per_token(self, loaded, torch):
        from app.recognition.incremental import IncrementalSMT

        model, tensor = loaded
        fast = IncrementalSMT(model)
        memory = fast.encode(tensor)

        token = model.w2i["<bos>"]
        for expected in (1, 2, 3):
            token = int(torch.argmax(fast.step(memory, token)).item())
            assert memory.layers[0].self_kt.shape[2] == expected
            assert memory.layers[-1].self_v.shape[1] == expected
            assert memory.length == expected

    def test_reset_clears_generated_state_but_keeps_the_encoder_work(self, loaded, torch):
        from app.recognition.incremental import IncrementalSMT

        model, tensor = loaded
        fast = IncrementalSMT(model)
        memory = fast.encode(tensor)
        cross = memory.layers[0].cross_kt

        fast.step(memory, model.w2i["<bos>"])
        memory.reset()

        assert memory.length == 0
        assert memory.layers[0].self_kt is None
        assert memory.layers[0].cross_kt is cross


GRANDSTAFF_WEIGHTS = ROOT / "models" / "hf" / "hub" / "models--PRAIG--smt-grandstaff"


@pytest.fixture(scope="module")
def grandstaff_model(torch):
    if not GRANDSTAFF_WEIGHTS.exists():
        pytest.skip("greutatile smt-grandstaff nu sunt descarcate")
    torch.set_num_threads(1)
    from app.recognition.grandstaff_engine import GrandstaffEngine, load_grandstaff
    from scripts.verify_incremental import synthetic_system

    model, _ = load_grandstaff(torch.device("cpu"))
    return model, GrandstaffEngine._to_tensor(synthetic_system())


@pytest.fixture(scope="module")
def grandstaff_comparison(grandstaff_model):
    from scripts.verify_incremental import compare

    model, tensor = grandstaff_model
    return compare(model, None, steps=8, tensor=tensor)


class TestGrandstaff:
    """Aceeasi verificare, pentru implementarea oficiala a SMT.

    Nu e o duplicare de complezenta: `smt-grandstaff` vine dintr-o versiune
    diferita a codului, in care atentia **imparte** la sqrt(head_dim), spre
    deosebire de forkul JAZZMUS. O portare facuta doar „dupa nume" ar produce
    tokeni plauzibili si gresiti, fara sa crape nicaieri.
    """

    def test_profilul_ales_e_cel_oficial(self, grandstaff_model):
        from app.recognition.incremental import OFFICIAL, detect_layout

        model, _ = grandstaff_model
        assert detect_layout(model) is OFFICIAL

    def test_atentia_e_scalata(self, grandstaff_model):
        # Valoarea, nu doar prezenta: cu scale=1.0 modelul ar decoda ceva, dar
        # altceva decat a fost antrenat sa decodeze.
        from app.recognition.incremental import IncrementalSMT

        model, _ = grandstaff_model
        fast = IncrementalSMT(model)
        assert fast.scale == pytest.approx(fast.head_dim ** -0.5)

    def test_logits_match_the_reference(self, grandstaff_comparison):
        worst, _, _, _ = grandstaff_comparison
        assert worst < 1e-3, f"diferenta maxima de logit: {worst:.3e}"

    def test_same_tokens_are_chosen(self, grandstaff_comparison):
        _, mismatches, _, _ = grandstaff_comparison
        assert mismatches == 0
