"""Perechile de antrenare -> tensori, si bucla reluabila (etapa 11).

Testele de aici nu incarca modelul: conventia de tokeni si citirea perechilor
sunt cod determinist, iar §9 cere ca partea in care se poate gresi tacut sa fie
verificabila in milisecunde. Bucla propriu-zisa e testata mai jos, cu un model
de jucarie — ce se verifica acolo e **reluarea**, nu invatarea.
"""

import json
import random

import pytest

from pathlib import Path

from app.training.data import (
    UnknownTokens,
    describe_writers,
    encode_target,
    load_pairs,
    noisy_decoder_input,
    split_by_fraction,
    split_by_writer,
    usable_pairs,
    writers,
)

VOCABULAR = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "4c": 3, "4d": 4, "=": 5}


def scrie_set(tmp_path, randuri):
    path = tmp_path / "date.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in randuri:
            handle.write(json.dumps(row) + "\n")
    return path


class TestCitireaPerechilor:
    def test_caile_sunt_relative_la_fisier(self, tmp_path):
        path = scrie_set(tmp_path, [
            {"image": "s01.png", "transcription": "4c 4d", "staves": 2, "writer": "ana"},
        ])
        pair = load_pairs(path)[0]
        # Setul trebuie sa se poata muta intreg, deci calea se rezolva langa
        # `date.jsonl`, nu fata de directorul curent.
        assert pair.image == tmp_path / "s01.png"
        assert pair.tokens == ["4c", "4d"]
        assert pair.staves == 2
        assert pair.writer == "ana"

    def test_randurile_goale_se_sar(self, tmp_path):
        path = tmp_path / "date.jsonl"
        row = {"image": "a.png", "transcription": "4c", "staves": 2, "writer": "ana"}
        path.write_text(json.dumps(row) + chr(10) * 2, encoding="utf-8")
        assert len(load_pairs(path)) == 1

    def test_fara_scriitor_se_refuza(self, tmp_path):
        # Fara eticheta, validarea n-ar putea tine un scris deoparte, iar
        # cifra ar raporta memorare drept citire.
        path = scrie_set(tmp_path, [
            {"image": "s01.png", "transcription": "4c", "staves": 2},
        ])
        with pytest.raises(ValueError, match="writer"):
            load_pairs(path)

    def test_fara_numarul_de_portative_se_refuza(self, tmp_path):
        # Perechile scrise inainte de 2026-09-01 n-au campul, iar fara el
        # imaginea s-ar aduce la alta scara decat la inferenta — o diferenta
        # care n-ar da nicio eroare, doar o transcriere mai proasta.
        path = scrie_set(tmp_path, [
            {"image": "s01.png", "transcription": "4c", "writer": "ana"},
        ])
        with pytest.raises(ValueError, match="staves"):
            load_pairs(path)

    def test_fisierul_lipsa_se_spune_pe_nume(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pairs(tmp_path / "nu-exista.jsonl")


class TestConventiaDeTokeni:
    """Conventia e cea din `vendor/SMT/data.py`, nu una inventata de noi."""

    def test_capetele_sunt_bos_si_eos(self):
        ids = encode_target(["4c", "4d"], VOCABULAR)
        assert ids[0] == VOCABULAR["<bos>"]
        assert ids[-1] == VOCABULAR["<eos>"]
        assert ids == [1, 3, 4, 2]

    def test_tokenii_necunoscuti_se_refuza_nu_se_inlocuiesc(self):
        # O tinta pe care modelul n-o poate reprezenta e o problema de date;
        # inlocuirea tacuta ar antrena modelul sa scrie altceva decat pagina.
        with pytest.raises(UnknownTokens, match="16c"):
            encode_target(["4c", "16c"], VOCABULAR)

    def test_toti_tokenii_necunoscuti_se_raporteaza_o_data(self):
        with pytest.raises(UnknownTokens) as excinfo:
            encode_target(["x", "y", "x"], VOCABULAR)
        assert "2 tokeni" in str(excinfo.value)


class TestZgomotulDeTeacherForcing:
    def test_bos_ul_ramane_neatins(self):
        # Bucla lor incepe de la 1: prima pozitie nu se strica niciodata.
        ids = [1] + [3] * 200
        noised = noisy_decoder_input(ids, len(VOCABULAR), 1.0, random.Random(0))
        assert noised[0] == 1

    def test_rata_zero_nu_schimba_nimic(self):
        ids = [1, 3, 4, 2]
        assert noisy_decoder_input(ids, len(VOCABULAR), 0.0, random.Random(0)) == ids

    def test_rata_unu_schimba_tot_ce_urmeaza(self):
        ids = [1] + [3] * 100
        noised = noisy_decoder_input(ids, 6, 1.0, random.Random(1))
        # Toate pozitiile au fost rescrise; unele pot nimeri aceeasi valoare.
        assert noised != ids

    def test_intrarea_nu_se_modifica_pe_loc(self):
        ids = [1, 3, 4, 2]
        noisy_decoder_input(ids, 6, 1.0, random.Random(0))
        assert ids == [1, 3, 4, 2]


class TestProgres:
    def test_orele_ramase_se_calculeaza_din_pasii_ramasi(self):
        from app.training.loop import Progress

        progress = Progress(step=100, steps=500, seconds_per_step=3.6)
        assert progress.remaining_hours == pytest.approx(0.4)

    def test_la_final_nu_mai_ramane_nimic(self):
        from app.training.loop import Progress

        assert Progress(step=500, steps=500, seconds_per_step=3.6).remaining_hours == 0


class TestInghetarea:
    def test_configuratia_necunoscuta_se_refuza(self):
        from app.training.loop import trainable_for

        with pytest.raises(ValueError, match="configuratie necunoscuta"):
            trainable_for(object(), "altceva")


def perechi(*specs):
    """Perechi de jucarie: `("ana", 3)` inseamna trei pagini scrise de Ana."""
    from app.training.data import Pair

    out = []
    for writer, count in specs:
        for index in range(count):
            out.append(Pair(Path(f"{writer}{index}.png"), ["4c"], 2, writer))
    return out


class TestImpartireaPeScriitor:
    """Piesa de care atarna orice cifra: fiecare din cele ~500 de pagini are alt scris."""

    def test_niciun_scriitor_nu_apare_in_amandoua_partile(self):
        pairs = perechi(("ana", 3), ("bob", 2), ("cezar", 4))
        train, validation = split_by_writer(pairs, "bob")
        assert not ({p.writer for p in train} & {p.writer for p in validation})
        assert {p.writer for p in validation} == {"bob"}
        assert (len(train), len(validation)) == (7, 2)

    def test_se_pot_tine_deoparte_mai_multi(self):
        pairs = perechi(("ana", 2), ("bob", 2), ("cezar", 2))
        train, validation = split_by_writer(pairs, {"ana", "bob"})
        assert {p.writer for p in train} == {"cezar"}
        assert len(validation) == 4

    def test_un_scriitor_care_nu_exista_se_refuza(self):
        # Aproape sigur o greseala de scriere; altfel validarea ar iesi goala si
        # n-ar da nicio cifra, fara ca nimeni sa observe.
        with pytest.raises(ValueError, match="scriitor necunoscut"):
            split_by_writer(perechi(("ana", 2)), "anna")

    def test_o_impartire_care_goleste_antrenarea_se_refuza(self):
        with pytest.raises(ValueError, match="cel putin doua scrisuri"):
            split_by_writer(perechi(("ana", 3)), "ana")

    def test_numaratoarea_pe_scriitori(self):
        assert writers(perechi(("bob", 1), ("ana", 2))) == {"ana": 2, "bob": 1}


@pytest.mark.slow
class TestAugmentarea:
    """Cere torch si torchvision, deci sta la testele lente."""

    def test_schimba_imaginea_dar_nu_tinta(self, tmp_path):
        """Doua proprietati, si numai una e sigura la fiecare extragere.

        In lantul lor fiecare transformare se aplica cu `p=0.2`, deci o
        extragere care nu schimba nimic e perfect normala — se intampla in
        ~26% din cazuri. Un test care cere „imaginea difera" o singura data ar
        pica din cand in cand, fara ca ceva sa fie stricat.

        Deci: **tinta nu se atinge niciodata** se cere la fiecare extragere, iar
        „imaginea se schimba" se cere pe zece, unde sansa unui fals negativ e
        sub o milionime.
        """
        import numpy as np
        from PIL import Image

        from app.training.data import Pair, encode_pair

        rng = np.random.default_rng(0)
        Image.fromarray((rng.random((300, 800)) * 255).astype("uint8")).save(
            tmp_path / "s01.png"
        )
        pair = Pair(tmp_path / "s01.png", ["4c", "4d"], 2, "ana")

        fara = encode_pair(pair, VOCABULAR, 256, 1024, rate=0.0, rng=random.Random(0))
        schimbari = 0
        for _ in range(10):
            cu = encode_pair(
                pair, VOCABULAR, 256, 1024, rate=0.0, rng=random.Random(0), augment=True
            )
            assert cu[0].shape == fara[0].shape
            assert bool((cu[2] == fara[2]).all())  # tinta, de fiecare data
            if not bool((cu[0] == fara[0]).all()):
                schimbari += 1
        assert schimbari > 0


class TestRezumatulScriitorilor:
    """Eticheta nu e un om, ci o grupare — deci pot fi sute."""

    def test_putini_se_insira(self):
        assert describe_writers(perechi(("ana", 1), ("bob", 2))) == "ana: 1, bob: 2"

    def test_multi_se_rezuma(self):
        rezumat = describe_writers(perechi(*[(f"p{i:03d}", 1) for i in range(137)]))
        assert "137 maini" in rezumat
        assert "p001" not in rezumat  # nu se varsa lista pe ecran

    def test_setul_gol(self):
        assert describe_writers([]) == "niciun scriitor"


class TestImpartireaPeCota:
    """Cazul real: fiecare pagina e alt scris, deci eticheta e pagina."""

    def _multe(self, n=100):
        return perechi(*[(f"pagina-{i:03d}", 1) for i in range(n)])

    def test_cota_se_ia_din_maini(self):
        train, validation = split_by_fraction(self._multe(), 0.2)
        assert len(writers(validation)) == 20
        assert len(writers(train)) == 80

    def test_impartirea_e_stabila(self):
        # Esential: o rulare reluata din checkpoint trebuie sa gaseasca aceeasi
        # impartire, altfel ar antrena pe pagini care erau validare inainte.
        pairs = self._multe()
        _, prima = split_by_fraction(pairs, 0.2)
        _, doua = split_by_fraction(pairs, 0.2)
        assert sorted(writers(prima)) == sorted(writers(doua))

    def test_samanta_schimba_alegerea(self):
        pairs = self._multe()
        _, cu_zero = split_by_fraction(pairs, 0.2, seed=0)
        _, cu_unu = split_by_fraction(pairs, 0.2, seed=1)
        assert sorted(writers(cu_zero)) != sorted(writers(cu_unu))

    def test_o_cota_minuscula_tot_tine_o_mana(self):
        _, validation = split_by_fraction(self._multe(4), 0.01)
        assert len(writers(validation)) == 1

    def test_o_cota_uriasa_lasa_o_mana_de_antrenat(self):
        train, _ = split_by_fraction(self._multe(4), 0.99)
        assert len(writers(train)) == 1

    def test_cota_in_afara_intervalului_se_refuza(self):
        for cota in (0, 1, 1.5, -0.2):
            with pytest.raises(ValueError, match="intre 0 si 1"):
                split_by_fraction(self._multe(), cota)

    def test_o_singura_mana_nu_se_poate_imparti(self):
        with pytest.raises(ValueError, match="cel putin doua"):
            split_by_fraction(perechi(("ana", 5)), 0.5)


class TestPerechileFolosibile:
    """Vocabularul se verifica inainte de primul pas, nu la nimereala in timpul lui."""

    def _pereche(self, writer, *tokens):
        from app.training.data import Pair

        return Pair(Path(f"{writer}.png"), list(tokens), 2, writer)

    def test_cele_bune_trec_si_cele_rele_se_explica(self):
        pairs = [
            self._pereche("ana", "4c", "4d"),
            self._pereche("bob", "4c", "[2e"),  # inceput de legato, lipsa din vocabular
        ]
        good, bad = usable_pairs(pairs, VOCABULAR)
        assert [p.writer for p in good] == ["ana"]
        assert len(bad) == 1
        assert "[2e" in bad[0][1]

    def test_un_set_intreg_bun_nu_lasa_nimic_pe_dinafara(self):
        good, bad = usable_pairs([self._pereche("ana", "4c")], VOCABULAR)
        assert len(good) == 1 and not bad

    def test_un_set_intreg_rau_se_vede_ca_atare(self):
        # `finetune.py` iese cu mesaj in loc sa porneasca o rulare fara date.
        good, bad = usable_pairs([self._pereche("ana", "16c")], VOCABULAR)
        assert not good and len(bad) == 1


FARA_VENDOR = not (Path(__file__).resolve().parents[1] / "vendor" / "SMT" / "data_augmentation").exists()


@pytest.mark.skipif(FARA_VENDOR, reason="augmentarea vine din vendor/SMT, care se cloneaza separat")
class TestRepetabilitatea:
    """`--seminte` trebuie sa insemne „aceeasi rulare", nu „aproape" (§15.27).

    Parea sa functioneze si nu functiona: alegerea perechilor mergea pe
    `random.Random(seed)`, dar **augmentarea** foloseste generatorul global al lui
    numpy, iar dropoutul pe al lui torch — amandoua nesamantate.

    Masurat: doua rulari cu exact aceeasi comanda au dat, la pasul 150, **15/182
    note** si **0/182**. Adica orice comparatie intre doua rulari masura zgomotul,
    nu schimbarea.
    """

    def pereche(self, tmp_path):
        import numpy as np
        from PIL import Image

        from app.training.data import Pair

        rng = np.random.default_rng(0)
        Image.fromarray((rng.random((300, 800)) * 255).astype("uint8")).save(
            tmp_path / "s01.png"
        )
        return Pair(tmp_path / "s01.png", ["4c", "4d"], 2, "ana")

    def augmentata(self, pair, seed):
        from app.training.data import encode_pair
        from app.training.loop import _seed_everything

        _seed_everything(seed)
        return encode_pair(
            pair, VOCABULAR, 256, 1024, rate=0.0, rng=random.Random(seed), augment=True
        )[0]

    def test_aceeasi_saminta_da_aceeasi_imagine(self, tmp_path):
        pair = self.pereche(tmp_path)
        assert bool((self.augmentata(pair, 0) == self.augmentata(pair, 0)).all())

    def test_alta_saminta_da_alta_imagine(self, tmp_path):
        """Altfel testul de sus ar trece si daca augmentarea n-ar face nimic."""
        pair = self.pereche(tmp_path)
        diferite = sum(
            not bool((self.augmentata(pair, 0) == self.augmentata(pair, s)).all())
            for s in range(1, 11)
        )
        assert diferite > 0


class TestCallbackurileNuMutaGeneratoarele:
    """Observatia nu mai schimba ce observa (§15.29).

    Gasit comparand doua rulari cu aceeasi saminta: identice pana la pasul 25,
    divergente exact la 26 — imediat dupa prima masuratoare. Aceeasi pereche
    trasa, alta pierdere, fiindca masuratoarea consumase din generatoarele
    globale si pasul urmator a primit alta augmentare si alt dropout.
    """

    def trage(self):
        import numpy as np
        import torch

        return (random.random(), float(np.random.rand()), float(torch.rand(1)))

    def seteaza(self):
        """Cele trei generatoare **globale**.

        Nu `_seed_everything`: acela lasa `random`-ul global in pace, fiindca
        alegerea perechilor merge pe o instanta privata. Aici ne trebuie toate
        trei pornite din acelasi loc, ca sa se vada ce pune paza la loc.
        """
        import numpy as np
        import torch

        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)

    def consuma(self):
        """Ce face, ca efect secundar, o masuratoare la checkpoint."""
        import numpy as np
        import torch

        random.random()
        np.random.rand()
        torch.rand(1)

    def test_ce_consuma_un_callback_se_pune_la_loc(self):
        from app.training.loop import rng_neatins

        self.seteaza()
        asteptat = self.trage()

        self.seteaza()
        with rng_neatins():
            self.consuma()
        assert self.trage() == asteptat

    def test_fara_paza_chiar_se_schimba(self):
        """Altfel testul de sus ar trece si daca n-ar exista nicio problema."""
        self.seteaza()
        asteptat = self.trage()

        self.seteaza()
        self.consuma()
        assert self.trage() != asteptat


class TestScorulCareAlegeCheckpointul:
    """Ce inseamna „cel mai bun" cand se pastreaza un singur fisier (§15.31)."""

    def test_un_model_mut_nu_castiga(self):
        """Cele doua candidate evidente pun amandoua muta prima (§15.30).

        Un model care scrie o nota si o nimereste are SER sub 1 si precizie
        100%. Pe note corecte iese ultimul, fiindca acolo tacerea nu ajuta.
        """
        from app.training.loop import scor_checkpoint

        mut = scor_checkpoint(matched_notes=1, hyp_events=1)
        citeste = scor_checkpoint(matched_notes=16, hyp_events=37)
        assert citeste > mut

    def test_o_pagina_de_tacere_nu_mai_castiga(self):
        """Corectia din §15.42: `matched` se putea castiga tacand, `matched_notes` nu.

        Un model care umple pagina cu pauze nimereste pauzele referintei — 11
        din cele 16 „corecte" ale reperului erau asa. Numarate ca note
        adevarate, alea sunt zero.
        """
        from app.training.loop import scor_checkpoint

        tacere = scor_checkpoint(matched_notes=0, hyp_events=180)
        citeste_putin = scor_checkpoint(matched_notes=4, hyp_events=40)
        assert citeste_putin > tacere

    def test_la_egalitate_castiga_cine_scrie_mai_putin(self):
        """Cazul real: reperul si pasul 25 nimeresc amandoua 16 note din 182.

        Reperul le scoate din 202 de note, antrenarea din 37. Aceeasi muzica
        regasita, de cinci ori si jumatate mai putin de sters.
        """
        from app.training.loop import scor_checkpoint

        assert scor_checkpoint(16, 37) > scor_checkpoint(16, 202)

    def test_mai_multe_note_corecte_bat_orice_economie(self):
        """Ordinea e lexicografica, nu un compromis intre cele doua."""
        from app.training.loop import scor_checkpoint

        assert scor_checkpoint(17, 500) > scor_checkpoint(16, 1)
