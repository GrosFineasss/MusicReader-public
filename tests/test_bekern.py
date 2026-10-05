"""Puntea dintre tokenii lui `smt-grandstaff` si parserul de `**kern`.

Modelul scoate un sir plat, cu `<t>` pentru tab si `<b>` pentru linie noua, si
cu exponentul `**ekern_1.0` in loc de `**kern`. Fara traducere, `split_voices`
gaseste zero voci si pierde toata partitura fara sa crape — de asta testele de
aici verifica intai ca traducerea produce un document valid, si abia apoi ce se
intelege din el.

Iesirea reala din `data/validation/grandstaff_system02.txt` e fisier de
regresie: e prima transcriere de sistem intreg pe care am avut-o, si tot ce am
descoperit pe ea trebuie sa ramana descoperit.
"""

from pathlib import Path

import pytest

from app.recognition.bekern import COLUMN, LINE, SPLIT, Grammar, to_kern
from app.recognition.kern import parse_open_system

ROOT = Path(__file__).resolve().parents[1]
REAL_OUTPUT = ROOT / "data" / "validation" / "grandstaff_system02.txt"


def _iesirea_reala() -> str:
    """Iesirea reala a modelului; fisier local, deci testul se sare intr-o copie fara date."""
    if not REAL_OUTPUT.exists():
        pytest.skip(f"lipseste {REAL_OUTPUT.name} (date locale, nu se publica)")
    return REAL_OUTPUT.read_text(encoding="utf-8")

# Un sistem minimal, scris ca tokeni: doua coloane, a doua se imparte in doua
# voci. Aceeasi forma ca iesirea reala, dar destul de scurt cat sa se citeasca.
TOKENS = [
    "**ekern_1.0", "<t>", "**ekern_1.0", "<b>",
    "*", "<t>", "*^", "<b>",
    "*clefF4", "<t>", "*clefG2", "<t>", "*clefG2", "<b>",
    "*k[]", "<t>", "*k[]", "<t>", "*k[]", "<b>",
    "*M4/4", "<t>", "*M4/4", "<t>", "*M4/4", "<b>",
    "=-", "<t>", "=-", "<t>", "=-", "<b>",
    "1G", "<t>", "2g", "<t>", "2e", "<b>",
    ".", "<t>", "2f", "<t>", "2d", "<b>",
]


class TestTraducere:
    def test_separatorii_devin_tab_si_linie_noua(self):
        assert to_kern(["4g", "<t>", "4d", "<b>", "4a"]).kern == "4g\t4d\n4a"

    def test_exponentul_ekern_devine_kern(self):
        for spine in ("**ekern_1.0", "**ekern", "**bekern", "**bekern_2.0"):
            assert to_kern([spine]).kern == "**kern"

    def test_alte_exponente_raman_neatinse(self):
        assert to_kern(["**mxhm"]).kern == "**mxhm"

    def test_subtokenii_ekern_se_lipesc_la_loc(self):
        # `4@g@#` e o nota, nu durata `4`: pastrarea primului camp ar arunca
        # inaltimea. Separatorii se sterg, subtokenii se lipesc.
        assert to_kern(["2.@ee@[", "<t>", "4@g@#"]).kern == "2.ee[\t4g#"

    def test_marcajele_necunoscute_se_raporteaza_nu_se_inghit(self):
        conversion = to_kern(["4g", "<pad>", "4a"])
        assert conversion.unknown == ["<pad>"]
        assert conversion.kern == "4g4a"

    def test_accepta_si_textul_salvat_pe_disc(self):
        assert to_kern("4g\n<t>\n4d").kern == to_kern(["4g", "<t>", "4d"]).kern

    def test_linia_taiata_la_final_se_pastreaza(self):
        # `--max-tokens` taie iesirea la mijlocul unei linii. Linia aceea e
        # ultima muzica citita, deci exact ce vrem sa vedem.
        assert to_kern(["4g", "<b>", "4a", "<t>"]).kern == "4g\n4a\t"

    def test_iesirea_reala_nu_contine_marcaje_necunoscute(self):
        conversion = to_kern(_iesirea_reala())
        assert conversion.unknown == []
        assert conversion.kern.startswith("**kern\t**kern")


class TestSistem:
    @pytest.fixture
    def parsed(self):
        return parse_open_system(to_kern(TOKENS).kern)

    def test_cheia_decide_portativul_nu_ordinea_coloanelor(self, parsed):
        # Humdrum pune vocea cea mai joasa in stanga, deci prima coloana e
        # portativul de jos. Repartizarea dupa pozitie ar da S&A basului.
        assert [part.id for part in parsed.score.parts] == ["S", "A", "T", "B"]
        assert parsed.score.part("T").measures[0].clef.sign == "F"
        assert parsed.score.part("S").measures[0].clef.sign == "G"

    def test_portativul_cu_o_singura_coloana_e_unison(self, parsed):
        # Coloana de bas nu se imparte niciodata in exemplu: tokenii ei trebuie
        # sa ajunga in amandoua vocile, altfel basul ar ramane cu masuri goale
        # si validatorul ar raporta erori inexistente.
        assert parsed.score.part("T").measures[0].events == parsed.score.part("B").measures[0].events
        assert any("unison" in problem for problem in parsed.problems)

    def test_nu_ramane_niciun_token_nerecunoscut(self, parsed):
        assert parsed.unrecognised == []

    def test_documentul_fara_coloane_kern_e_respins(self):
        # Fara traducere, `split_voices` ar gasi zero voci si ar produce o
        # partitura goala, in tacere. Preferam sa crape.
        with pytest.raises(ValueError, match="bekern"):
            parse_open_system("**ekern_1.0\n*clefG2\n4g\n")


class TestIesireaReala:
    """Sistemul al doilea din „Iata-L pe Hristos", citit de `smt-grandstaff`."""

    @pytest.fixture
    def parsed(self):
        return parse_open_system(to_kern(_iesirea_reala()).kern)

    def test_da_patru_voci(self, parsed):
        assert [part.id for part in parsed.score.parts] == ["S", "A", "T", "B"]

    def test_tokenul_null_nu_e_raportat_ca_nerecunoscut(self, parsed):
        # Iesirea are 28 de `.` — tokenul null Humdrum. Pana la sisteme intregi
        # coloana `**mxhm` era aruncata si cazul nu aparea.
        assert parsed.unrecognised == []

    def test_vocea_aparuta_la_mijloc_ramane_sincronizata(self, parsed):
        # Basul se desparte de tenor abia la masura 4. Fara mostenirea barelor,
        # ar incepe sa numere de la 1 si s-ar desincroniza de restul sistemului.
        assert {part.id: len(part.measures) for part in parsed.score.parts} == {
            "S": 4, "A": 4, "T": 4, "B": 4,
        }
        assert parsed.score.part("B").measures[-1].number == 4


class TestAcorduri:
    """`<s>` e un spatiu, iar un spatiu in interiorul coloanei e un acord."""

    def test_marcajul_de_spatiu_se_traduce_in_spatiu(self):
        assert to_kern(["4g", "<s>", "4b", "<t>", "4c"]).kern == "4g 4b\t4c"
        assert to_kern(["4g", "<s>", "4b"]).unknown == []

    def test_acordul_pastreaza_prima_nota_si_se_raporteaza(self):
        # Acordurile sunt in afara subsetului v1. `_NOTE` ar potrivi tacut doar
        # prima nota; vrem sa se vada ca am pierdut ceva.
        parsed = parse_open_system("**kern\n*clefG2\n*M4/4\n=\n2g 2b\n2a\n")
        assert any("acord" in item for item in parsed.unrecognised)
        assert len(parsed.score.part("S").measures[0].events) == 2


class TestDocumentStricat:
    """Cand modelul nu scrie un tabel valid, trebuie sa se vada.

    Pe primul sistem al paginii reale — un singur portativ, dus din greseala la
    inaltimea unui sistem intreg — modelul a scos randuri cu patru campuri acolo
    unde structura avea trei coloane. `zip` le taia in tacere.
    """

    DOCUMENT = "**kern\t**kern\n*clefF4\t*clefG2\n*M4/4\t*M4/4\n=-\t=-\n4G\t4g\t4e\n4A\t4a\n"

    def test_randurile_cu_prea_multe_campuri_se_raporteaza(self):
        parsed = parse_open_system(self.DOCUMENT)
        assert any("campuri" in problem for problem in parsed.problems)

    def test_raportul_e_un_rand_nu_o_lista(self):
        # 67 de randuri stricate pe un singur sistem ar ineca restul raportului.
        parsed = parse_open_system(self.DOCUMENT)
        assert len([p for p in parsed.problems if "campuri" in p]) == 1

    def test_documentul_valid_nu_raporteaza_nimic(self):
        parsed = parse_open_system(to_kern(TOKENS).kern)
        assert not [p for p in parsed.problems if "campuri" in p]


# --------------------------------------------------------------------------
# Gramatica: structura de tabel, urmarita token cu token (§15, experimentul 2)
# --------------------------------------------------------------------------


def feed(grammar, words):
    """Trece tokenii prin gramatica si intoarce ce era interzis la fiecare pas."""
    banned = []
    for word in words:
        banned.append(grammar.forbidden())
        grammar.accept(word)
    return banned


class TestGramatica:
    def test_antetul_hotaraste_cate_coloane_are_documentul(self):
        grammar = Grammar()
        feed(grammar, ["**ekern_1.0", COLUMN, "**ekern_1.0", LINE])
        assert grammar.columns == 2

    def test_un_rand_neterminat_nu_se_poate_inchide(self):
        """`<b>` pe primul camp dintr-un document cu doua coloane e imposibil."""
        grammar = Grammar(spines=2)
        assert LINE in grammar.forbidden()

    def test_ultimul_camp_nu_mai_poate_primi_tab(self):
        grammar = Grammar(spines=2)
        feed(grammar, ["**ekern_1.0", COLUMN, "**ekern_1.0"])
        assert COLUMN in grammar.forbidden()
        assert LINE not in grammar.forbidden()

    def test_randurile_de_dupa_urmeaza_numarul_de_coloane_curent(self):
        grammar = Grammar()
        feed(grammar, ["**ekern_1.0", COLUMN, "**ekern_1.0", LINE, "4c"])
        assert COLUMN not in grammar.forbidden()
        grammar.accept(COLUMN)
        grammar.accept("4e")
        assert COLUMN in grammar.forbidden()

    def test_o_desfacere_de_voce_mareste_numarul_de_coloane(self):
        grammar = Grammar()
        feed(grammar, ["**ekern_1.0", COLUMN, "**ekern_1.0", LINE])
        feed(grammar, ["*", COLUMN, "*^", LINE])
        assert grammar.columns == 3

    def test_lipirea_vocilor_il_scade_la_loc(self):
        grammar = Grammar(columns=3)
        feed(grammar, ["*v", COLUMN, "*v", COLUMN, "*", LINE])
        assert grammar.columns == 2

    def test_plafonul_de_voci_face_desfacerea_imposibila(self):
        """Declarat, ca `--portative-pe-sistem`: doua portative, doua voci."""
        grammar = Grammar(columns=4, max_columns=4)
        assert SPLIT in grammar.forbidden()

    def test_fara_plafon_desfacerea_ramane_permisa(self):
        """Pe „Iata-L pe Hristos" numarul de voci chiar variaza."""
        grammar = Grammar(columns=9)
        assert SPLIT not in grammar.forbidden()

    def test_plafonul_numara_si_desfacerile_din_randul_curent(self):
        grammar = Grammar(columns=2, max_columns=3)
        grammar.accept("*^")
        assert SPLIT in grammar.forbidden()

    def test_documentul_produs_asa_are_randuri_egale(self):
        """Proba care conteaza: ce iese trebuie sa fie un tabel care se inchide.

        Pe pagina 3, fara gramatica, 59 de randuri aveau alt numar de campuri
        decat structura, iar `split_voices` le taia tacut cu `zip`.
        """
        grammar = Grammar(spines=2)
        words = ["**ekern_1.0", COLUMN, "**ekern_1.0", LINE, "4c", COLUMN, "4e", LINE]
        for word in words:
            assert word not in grammar.forbidden()
            grammar.accept(word)
        document = to_kern(words).kern
        assert {len(line.split("\t")) for line in document.splitlines()} == {2}
