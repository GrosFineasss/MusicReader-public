"""Generatorul de perechi sintetice (etapa 11, §15.18).

Testele de aici **nu cer Smashcima**, care nu e in `.venv`-ul proiectului
(§15.14). Ce se verifica e partea determinista, si mai ales **poarta**: o tinta
care nu se inchide nu are voie sa ajunga in set. Randarea propriu-zisa se
verifica prin rulare reala, nu prin test — ar cere alt mediu.
"""

import pytest

from app.music_model.model import empty_satb, measures_slice
from scripts.dataset import system_bounds
from scripts.sintetic import target_for


class TestFeliereaPeSisteme:
    """`system_bounds` e refolosit din `dataset.py`, nu rescris."""

    def test_rupturile_devin_intervale(self):
        assert system_bounds([1, 6, 12], 14) == [(1, 5), (6, 11), (12, 14)]

    def test_un_singur_sistem(self):
        assert system_bounds([1], 8) == [(1, 8)]

    def test_ultimul_sistem_merge_pana_la_capat(self):
        prima, ultima = system_bounds([1, 5], 9)[-1]
        assert (prima, ultima) == (5, 9)


class TestPoartaDeTinta:
    """Regula: mai bine nicio pereche decat una care minte."""

    def _felie(self):
        """Un sistem de partitura stransa: S+A pe Sol, T+B pe Fa.

        `empty_satb` da tenorului cheia Sol-8, adica patru portative — si
        scriitorul o refuza pe buna dreptate (§15.7: un portativ are o singura
        cheie). Aici se pune cheia Fa la amandoua vocile de jos, ca sa fie
        partitura stransa, forma in care sunt scrise manuscrisele.
        """
        from app.music_model.model import Clef, Duration, Note, NoteType, Pitch

        score = empty_satb(measures=2)
        inaltimi = {"S": ("C", 5), "A": ("E", 4), "T": ("C", 4), "B": ("C", 3)}
        for part in score.parts:
            step, octave = inaltimi[part.id]
            if part.id in ("T", "B"):
                part.measures[0].clef = Clef.bass()
            for measure in part.measures:
                measure.events = [
                    Note(Pitch(step, octave), Duration(NoteType.QUARTER))
                    for _ in range(4)
                ]
        return measures_slice(score, 1, 2)

    def test_o_tinta_care_se_inchide_trece(self):
        text, tokens = target_for(self._felie())
        assert text.startswith("**kern")
        assert tokens.split()[0] == "**ekern_1.0"

    def test_tokenii_sunt_sirul_pe_care_il_asteapta_modelul(self):
        _, tokens = target_for(self._felie())
        # Acelasi format ca la `dataset.py`: tokeni despartiti prin spatiu.
        assert " " in tokens and "\t" not in tokens

    def test_o_felie_fara_asezare_cade_pe_partitura_stransa(self):
        # `empty_satb` isi declara asezarea; fara ea, scriitorul presupune SATB
        # strans — si tot trebuie sa se inchida.
        felie = self._felie()
        felie.staves = []
        text, tokens = target_for(felie)
        # Doua coloane in antet = doua portative: scriitorul a folosit
        # partitura stransa implicita, iar `target_for` n-a ridicat, deci
        # dus-intorsul s-a inchis si pe drumul dupa cheie.
        assert text.splitlines()[0].split("	").count("**kern") == 2
        assert tokens

    def test_o_felie_cu_masuri_nepotrivite_e_refuzata(self):
        # Vocile trebuie sa aiba aceleasi masuri; altfel documentul Humdrum
        # n-are un singur sir de bare, iar `to_kern_text` refuza.
        from app.recognition.kern_writer import UnwritableScore

        felie = self._felie()
        felie.part("A").measures.pop()
        with pytest.raises((UnwritableScore, ValueError)):
            target_for(felie)


class TestReincercarile:
    """§15.13: blocajul e intermitent, iar o rulare de noapte tine ~4 ore."""

    def _render(self, esecuri):
        """Un randator fals care cade de `esecuri` ori, apoi reuseste."""
        stare = {"n": 0}

        def render(model, xml, staves, dpi):
            stare["n"] += 1
            if stare["n"] <= esecuri:
                raise ImportError("DLL load failed: An Application Control policy")
            return ("imagine", 1)

        return render, stare

    def test_reuseste_dupa_esecuri_trecatoare(self, monkeypatch):
        import scripts.sintetic as sintetic

        monkeypatch.setattr(sintetic, "RETRY_WAIT", 0)
        render, stare = self._render(esecuri=2)
        note = []
        rezultat = sintetic.render_with_retries(render, None, "f.musicxml", 2, 300.0, note)
        assert rezultat == ("imagine", 1)
        assert stare["n"] == 3
        # Nu se ascunde nimic: fiecare reincercare lasa o urma in raport.
        assert len(note) == 2

    def test_dupa_toate_incercarile_ridica(self, monkeypatch):
        # Reincercarea nu inseamna „inghite pentru totdeauna": daca esecul e
        # adevarat, rularea trebuie sa se opreasca si sa spuna de ce.
        import scripts.sintetic as sintetic

        monkeypatch.setattr(sintetic, "RETRY_WAIT", 0)
        render, _ = self._render(esecuri=99)
        with pytest.raises(ImportError):
            sintetic.render_with_retries(render, None, "f.musicxml", 2, 300.0, [])

    def test_o_randare_buna_nu_lasa_urme(self, monkeypatch):
        import scripts.sintetic as sintetic

        monkeypatch.setattr(sintetic, "RETRY_WAIT", 0)
        render, _ = self._render(esecuri=0)
        note = []
        sintetic.render_with_retries(render, None, "f.musicxml", 2, 300.0, note)
        assert note == []


class TestGlifeleLipsa:
    """Cu 2 oameni si 108 casute, cineva va sari una."""

    def test_mesajele_lui_smashcima_se_recunosc(self):
        from scripts.sintetic import eticheta_lipsa

        assert eticheta_lipsa(Exception("Unsupported glyph label: timeSig6")) == "timeSig6"
        assert eticheta_lipsa(
            Exception("The glyph class smufl::restQuarter is not present in the symbol repository")
        ) == "smufl::restQuarter"

    def test_un_blocaj_nu_e_confundat_cu_o_glifa_lipsa(self):
        # §15.13: blocajul e trecator si se reincearca. O eticheta lipsa nu.
        from scripts.sintetic import eticheta_lipsa

        assert eticheta_lipsa(ImportError("DLL load failed: Application Control")) is None

    def test_glifa_lipsa_nu_se_reincearca(self, monkeypatch):
        """Reincercarea are rost doar la ce trece de la sine.

        La ~60 s pe randare, trei incercari pentru un simbol pe care nimeni nu
        l-a scris inseamna trei minute pierdute de fiecare data.
        """
        import scripts.sintetic as sintetic

        monkeypatch.setattr(sintetic, "RETRY_WAIT", 0)
        incercari = {"n": 0}

        def render(model, xml, staves, dpi):
            incercari["n"] += 1
            raise Exception("Unsupported glyph label: timeSig6")

        with pytest.raises(sintetic.GlifaLipsa) as excinfo:
            sintetic.render_with_retries(render, None, "f.musicxml", 2, 300.0, [])
        assert excinfo.value.eticheta == "timeSig6"
        assert incercari["n"] == 1  # o singura data, nu trei


class TestRaportulDeImprumut:
    """Imprumutul intre maini se face tacut; raportul il face vizibil."""

    def _raport(self, capsys, acoperiri, simboluri):
        from scripts.glife_repository import raporteaza_lipsurile

        raporteaza_lipsurile(acoperiri, simboluri)
        return capsys.readouterr().out

    def test_ce_lipseste_de_la_unul_se_imprumuta(self, capsys):
        iesire = self._raport(
            capsys,
            {"ana": {"gClef": 3, "fClef": 0}, "radu": {"gClef": 3, "fClef": 3}},
            ["gClef", "fClef"],
        )
        assert "ana" in iesire and "fClef" in iesire
        assert "scrisul altora" in iesire

    def test_ce_lipseste_de_la_toti_opreste_generarea(self, capsys):
        iesire = self._raport(
            capsys,
            {"ana": {"gClef": 3, "fClef": 0}, "radu": {"gClef": 3, "fClef": 0}},
            ["gClef", "fClef"],
        )
        assert "lipsesc de la TOTI" in iesire
        assert "fClef" in iesire

    def test_un_set_complet_nu_raporteaza_nimic(self, capsys):
        iesire = self._raport(
            capsys, {"ana": {"gClef": 3}, "radu": {"gClef": 3}}, ["gClef"]
        )
        assert iesire.strip() == ""
