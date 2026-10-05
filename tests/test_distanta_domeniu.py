"""Masuratoarea distantei dintre randare si fotografie (PROJECT.md §15.30).

Motorul nu se incarca aici: ce se poate gresi tacut la scriptul asta nu e
inferenta, ci **imperecherea**. Doua liste de lungimi diferite, imperecheate in
ordine, ar da o cifra care compara sistemul 1 randat cu sistemul 2 fotografiat —
si ar arata la fel de rezonabila ca una corecta.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.distanta_domeniu import main, tokeni_emisi


@dataclass
class FalsStaff:
    kern: str
    tokens: int = 0


class FalsMotor:
    """Un motor care intoarce mereu acelasi `**kern`, ca sa se numere doar."""

    def __init__(self, kerns: list[str], tokeni: list[int] | None = None) -> None:
        self.kerns = kerns
        self.tokeni = tokeni or [0] * len(kerns)

    def transcribe(self, page):
        staves = [FalsStaff(k, t) for k, t in zip(self.kerns, self.tokeni)]
        return type("Rezultat", (), {"staves": staves})()


class TestNumaratoarea:
    def test_se_numara_tokenii_bruti_din_toate_sistemele(self, tmp_path):
        motor = FalsMotor(["**kern **kern 4c 4d", "=- 2e"])
        assert tokeni_emisi(motor, tmp_path / "orice.png") == (6, False)

    def test_un_kern_gol_da_zero(self, tmp_path):
        assert tokeni_emisi(FalsMotor([""]), tmp_path / "orice.png") == (0, False)


class TestPlafonul:
    """O cifra taiata de plafon arata exact ca una masurata (§15.36).

    Cazul real: 1276 de tokeni de model dau 638 de kern, iar 638 a aparut pe
    trei imagini si doua seturi de greutati, semanand a masuratoare.
    """

    def test_atingerea_plafonului_se_raporteaza(self, tmp_path):
        motor = FalsMotor(["4c 4d"], tokeni=[1276])
        assert tokeni_emisi(motor, tmp_path / "o.png", plafon=1276) == (2, True)

    def test_sub_plafon_nu_se_raporteaza(self, tmp_path):
        motor = FalsMotor(["4c 4d"], tokeni=[1275])
        assert tokeni_emisi(motor, tmp_path / "o.png", plafon=1276) == (2, False)

    def test_fara_plafon_dat_nu_se_pretinde_ca_se_stie(self, tmp_path):
        """Fara plafon nu se poate sti daca s-a taiat — si nu se ghiceste."""
        motor = FalsMotor(["4c 4d"], tokeni=[99999])
        assert tokeni_emisi(motor, tmp_path / "o.png") == (2, False)


class TestImperecherea:
    def falsa_imagine(self, tmp_path, nume):
        cale = tmp_path / nume
        cale.write_bytes(b"")
        return cale

    def test_liste_de_lungimi_diferite_se_refuza(self, tmp_path, monkeypatch):
        a = self.falsa_imagine(tmp_path, "a.png")
        b = self.falsa_imagine(tmp_path, "b.png")
        monkeypatch.setattr(
            "sys.argv",
            ["distanta_domeniu.py", "--sintetic", str(a), "--real", str(a), str(b)],
        )
        with pytest.raises(SystemExit) as ridicat:
            main()
        assert "imperecheaza in ordine" in str(ridicat.value)

    def test_tinte_cate_alta_socoteala_se_refuza(self, tmp_path, monkeypatch):
        a = self.falsa_imagine(tmp_path, "a.png")
        tinta = tmp_path / "t.krn"
        tinta.write_text("4c 4d", encoding="utf-8")
        monkeypatch.setattr(
            "sys.argv",
            ["distanta_domeniu.py", "--sintetic", str(a), "--real", str(a),
             "--tinta", str(tinta), str(tinta)],
        )
        with pytest.raises(SystemExit) as ridicat:
            main()
        assert "tinte" in str(ridicat.value)

    def test_un_fisier_care_lipseste_se_spune_pe_nume(self, tmp_path, monkeypatch):
        a = self.falsa_imagine(tmp_path, "a.png")
        lipsa = tmp_path / "nu-exista.png"
        monkeypatch.setattr(
            "sys.argv",
            ["distanta_domeniu.py", "--sintetic", str(a), "--real", str(lipsa)],
        )
        with pytest.raises(SystemExit) as ridicat:
            main()
        assert "nu-exista.png" in str(ridicat.value)
