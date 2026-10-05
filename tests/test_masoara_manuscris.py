"""`masoara_manuscris.py`: notele se numara pe portativ, nu pe voce."""

from scripts.masoara_manuscris import compare_system, levenshtein

TINTA = "\n".join([
    "**kern\t**kern",
    "*clefF4\t*clefG2",
    "*M4/4\t*M4/4",
    "=\t=",
    "4C\t4c",
    "4D\t4d",
    "4E\t4e",
    "4F\t4f",
    "==\t==",
    "*-\t*-",
])


def test_o_voce_despartita_altfel_nu_costa_nimic():
    """Basul scris pe doua voci la unison: aceleasi note pe acelasi portativ."""
    despartit = "\n".join([
        "**kern\t**kern",
        "*clefF4\t*clefG2",
        "*M4/4\t*M4/4",
        "*^\t*",
        "=\t=\t=",
        "4C\t4C\t4c",
        "4D\t4D\t4d",
        "4E\t4E\t4e",
        "4F\t4F\t4f",
        "==\t==\t==",
        "*v\t*v\t*",
        "*-\t*-",
    ])
    rezultat = compare_system(TINTA, despartit, 2)
    assert rezultat["note_tinta"] == 8
    assert rezultat["corecte"] == 8


def test_inaltimea_gresita_se_numara():
    gresit = TINTA.replace("4e", "4g")
    rezultat = compare_system(TINTA, gresit, 2)
    assert rezultat["corecte"] == 7
    assert rezultat["distanta"] > 0


def test_iesirea_goala_e_zero_nu_eroare():
    rezultat = compare_system(TINTA, "", 2)
    assert rezultat["corecte"] == 0
    assert rezultat["problema"] == "gol"


def test_levenshtein():
    assert levenshtein(list("abc"), list("abd")) == 1
    assert levenshtein([], list("ab")) == 2
