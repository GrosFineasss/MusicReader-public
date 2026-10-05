"""Ce motor primeste fiecare tip de partitura.

E un singur loc de ramificare, si tocmai de asta merita test: daca se schimba
tacut, tot ce e deasupra — lotul, fereastra, `transcribe.py` — incepe sa
foloseasca alt model fara ca nimeni sa observe. Exact asta s-a si intamplat
pana pe 29.08: comparatia din §12.4 alesese `smt-grandstaff`, dar rutarea
ramasese pe `jazzmus-model`.

Testele nu incarca nimic greu: constructia motorului e lenesa, greutatile se
citesc abia la `load()`.
"""

import pytest

from app.recognition.engine import Kind
from app.recognition.smt_engine import engine_for


@pytest.mark.parametrize("kind", list(Kind))
def test_amandoua_tipurile_merg_pe_grandstaff(kind):
    """Partitura stransa e forma in care sunt scrise toate partiturile noastre.

    `jazzmus-model` nu emite `*^`, deci arunca tacit una din cele doua voci ale
    fiecarui portativ — vezi §12.4. Nepotrivit pentru amandoua tipurile.
    """
    assert engine_for(kind).name == "grandstaff"


def test_argumentele_ajung_la_motor():
    engine = engine_for(Kind.HANDWRITTEN, threads=2, max_tokens=50, staves_per_system=2)
    assert (engine.threads, engine.max_tokens, engine.staves_per_system) == (2, 50, 2)


def test_gruparea_se_deduce_daca_nu_e_declarata():
    """Implicit nu declaram nimic — comportamentul de pe tiparit ramane cel vechi."""
    assert engine_for(Kind.PRINTED).staves_per_system is None
