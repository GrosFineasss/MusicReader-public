"""Greutatile aplicatiei: unde le cauta, si ce pastreaza scriptul care le publica (2026-10-04).

Pana acum aplicatia rula mereu modelul publicat — 32,7% pe `pagina3` tiparita — desi antrenarile
ajunsesera la ~78%. Testele nu incarca modelul: doar caile si continutul fisierului.
"""

import pytest

from app.recognition.smt_engine import engine_for, greutati_produs
from app.recognition.engine import Kind
from scripts.publica_greutati import publica

torch = pytest.importorskip("torch")


def test_fara_fisier_aplicatia_ramane_pe_modelul_publicat(tmp_path):
    assert greutati_produs(Kind.PRINTED, tmp_path / "nu-exista.pt") is None


def test_cu_fisier_aplicatia_il_ia(tmp_path):
    cale = tmp_path / "greutati.pt"
    cale.write_bytes(b"x")
    assert greutati_produs(Kind.PRINTED, cale) == cale


def test_fiecare_fel_de_pagina_isi_are_fisierul(monkeypatch, tmp_path):
    """Scrisul de mana nu cade pe greutatile de tiparit: pe mana erau sub modelul publicat."""
    import app.recognition.smt_engine as smt

    tiparit, mana = tmp_path / "greutati.pt", tmp_path / "greutati-mana.pt"
    monkeypatch.setattr(smt, "PRODUS", tiparit)
    monkeypatch.setattr(smt, "PRODUS_MANA", mana)
    tiparit.write_bytes(b"x")
    assert smt.greutati_produs(Kind.PRINTED) == tiparit
    assert smt.greutati_produs(Kind.HANDWRITTEN) is None
    mana.write_bytes(b"x")
    assert smt.greutati_produs(Kind.HANDWRITTEN) == mana


def test_greutatile_ajung_la_motor(tmp_path):
    cale = tmp_path / "greutati.pt"
    assert engine_for(Kind.PRINTED, weights=cale).weights == cale


@pytest.fixture
def checkpoint(tmp_path):
    """Forma unui checkpoint de la `finetune.py`: model, optimizator, pas, generatoare."""
    cale = tmp_path / "pdmx-produs" / "pas8000.pt"
    cale.parent.mkdir()
    torch.save(
        {
            "model": {"decoder.w": torch.ones(2, 2)},
            "optimizer": {"state": {0: {"exp_avg": torch.zeros(2, 2)}}},
            "step": 8000,
            "rng": [1, 2, 3],
        },
        cale,
    )
    return cale


def test_publicarea_pastreaza_doar_modelul(checkpoint, tmp_path):
    destinatie = tmp_path / "produs" / "greutati.pt"
    publica(checkpoint, destinatie, nota="pagina3 tiparita 484/624")
    scris = torch.load(destinatie, map_location="cpu", weights_only=False)
    assert set(scris) == {"model", "step", "sursa", "nota", "data", "licente_necomerciale"}
    assert scris["licente_necomerciale"] == []
    assert torch.equal(scris["model"]["decoder.w"], torch.ones(2, 2))
    assert scris["step"] == 8000
    assert scris["nota"] == "pagina3 tiparita 484/624"
    assert not destinatie.with_suffix(".tmp").exists()


def test_greutatile_bach_nu_se_publica(tmp_path):
    """Coralele Bach au licenta neclara pentru uz comercial (§15.59): doar diagnostic."""
    cale = tmp_path / "bach-acorduri" / "pas8000.pt"
    cale.parent.mkdir()
    torch.save({"model": {}}, cale)
    with pytest.raises(ValueError, match="Bach"):
        publica(cale, tmp_path / "greutati.pt")


def test_greutatile_musicorpus_nu_se_publica(tmp_path):
    """MusiCorpus e CC BY-NC-SA 4.0 (2026-10-05): cercetare, nu produs."""
    cale = tmp_path / "musicorpus-mana" / "sub" / "pas9000.pt"
    cale.parent.mkdir(parents=True)
    torch.save({"model": {}}, cale)
    with pytest.raises(ValueError, match="MusiCorpus"):
        publica(cale, tmp_path / "greutati.pt")


def test_cu_necomercial_se_publica_si_licenta_ramane_scrisa(tmp_path):
    """Uz personal (2026-10-05): voie, dar spus explicit si scris in fisier."""
    cale = tmp_path / "musicorpus-mana" / "pas9000.pt"
    cale.parent.mkdir()
    torch.save({"model": {"w": torch.ones(1)}, "step": 9000}, cale)
    destinatie = tmp_path / "greutati-mana.pt"
    publica(cale, destinatie, necomercial=True)
    scris = torch.load(destinatie, map_location="cpu", weights_only=False)
    assert any("MusiCorpus" in licenta for licenta in scris["licente_necomerciale"])
