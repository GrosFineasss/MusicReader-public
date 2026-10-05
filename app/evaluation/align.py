"""Aliniere de siruri — Needleman-Wunsch cu cost injectat.

De ce nu comparam pozitional: daca modelul pierde o masura, toate masurile de
dupa se decaleaza cu una. O potrivire dupa index ar raporta atunci sute de
greseli pentru o singura eroare reala, iar cifra n-ar mai spune nimic despre
model. Alinierea plateste masura lipsa o data si continua potrivit.

Modulul e generic intentionat: acelasi cod aliniaza sirul de masuri al unei
voci (cu cost = distanta dintre continuturile a doua masuri) si sirul de
evenimente din interiorul unei perechi de masuri (cu cost = cat difera doua
note). Cine cheama decide ce inseamna „cost"; aici nu se stie nimic despre
muzica.

Costurile sunt `Fraction`, nu `float`, din acelasi motiv pentru care duratele
sunt `Fraction`: programarea dinamica alege intre drumuri comparand sume de
costuri, iar doua drumuri egale trebuie sa iasa **egale**, ca departajarea sa
fie facuta de regula noastra de determinism, nu de eroarea de rotunjire.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import TypeVar

R = TypeVar("R")
H = TypeVar("H")

# Costul unui gol. Costul de substituire e normalizat in [0, 1] de catre cel
# care cheama, deci o substituire (cel mult 1) e intotdeauna preferata unei
# perechi stergere+inserare (2). Asta e si intentia: cand referinta si
# transcrierea au fiecare cate un element in acelasi loc, vrem sa aflam *cu
# cat* difera, nu sa raportam ca unul lipseste si altul e in plus.
GAP = Fraction(1)


class OpKind(str, Enum):
    """Ce s-a intamplat cu un element al referintei."""

    MATCH = "match"           # identic
    SUBSTITUTE = "substitute"  # prezent, dar diferit
    MISSING = "missing"        # in referinta, absent din transcriere
    EXTRA = "extra"            # in transcriere, absent din referinta


@dataclass(frozen=True)
class Op:
    """O operatie de aliniere, cu indicii de ambele parti.

    `ref` lipseste la EXTRA, `hyp` lipseste la MISSING. Indicii se pastreaza
    pentru ca raportul trebuie sa poata arata **unde** — o cifra fara
    localizare nu ajuta la nicio corectura.
    """

    kind: OpKind
    ref: int | None
    hyp: int | None
    cost: Fraction = Fraction(0)


def align(
    ref: Sequence[R],
    hyp: Sequence[H],
    cost: Callable[[R, H], Fraction],
    gap: Fraction = GAP,
) -> list[Op]:
    """Alinierea de cost minim intre `ref` si `hyp`.

    `cost` intoarce cat de mult difera doua elemente, normalizat in [0, 1];
    zero inseamna identice si produce un MATCH.

    La costuri egale departajarea e fixa — diagonala, apoi lipsa, apoi in plus.
    Fara o regula explicita, doua rulari pe aceleasi date ar putea raporta
    aceleasi greseli in feluri diferite, si n-am mai putea compara doua
    masuratori intre ele.
    """
    n, m = len(ref), len(hyp)

    # d[i][j] = costul minim de aliniere a primelor i elemente din referinta
    # cu primele j din transcriere. back[i][j] retine din ce directie s-a venit.
    d: list[list[Fraction]] = [[Fraction(0)] * (m + 1) for _ in range(n + 1)]
    back: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    subs: dict[tuple[int, int], Fraction] = {}

    for i in range(1, n + 1):
        d[i][0] = gap * i
        back[i][0] = "missing"
    for j in range(1, m + 1):
        d[0][j] = gap * j
        back[0][j] = "extra"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = cost(ref[i - 1], hyp[j - 1])
            subs[(i - 1, j - 1)] = substitution
            diagonal = d[i - 1][j - 1] + substitution
            missing = d[i - 1][j] + gap
            extra = d[i][j - 1] + gap
            best = min(diagonal, missing, extra)
            d[i][j] = best
            # Ordinea de mai jos *este* regula de departajare.
            back[i][j] = (
                "diagonal" if best == diagonal else "missing" if best == missing else "extra"
            )

    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diagonal":
            substitution = subs[(i - 1, j - 1)]
            kind = OpKind.MATCH if substitution == 0 else OpKind.SUBSTITUTE
            ops.append(Op(kind, i - 1, j - 1, substitution))
            i, j = i - 1, j - 1
        elif move == "missing":
            ops.append(Op(OpKind.MISSING, i - 1, None, gap))
            i -= 1
        else:
            ops.append(Op(OpKind.EXTRA, None, j - 1, gap))
            j -= 1

    ops.reverse()
    return ops


def total_cost(ops: Sequence[Op]) -> Fraction:
    """Costul intregii alinieri — distanta de editare."""
    return sum((op.cost for op in ops), Fraction(0))


def normalized_cost(ops: Sequence[Op], ref_length: int, hyp_length: int) -> Fraction:
    """Distanta adusa in [0, 1], ca sa poata fi folosita ea insasi drept cost.

    Impartirea se face la lungimea celui mai lung dintre siruri: e cel mai mare
    numar de operatii pe care le poate cere alinierea, deci raportul nu poate
    depasi 1. Doua siruri goale sunt identice, deci costa 0.
    """
    longest = max(ref_length, hyp_length)
    if longest == 0:
        return Fraction(0)
    return min(Fraction(1), total_cost(ops) / longest)
