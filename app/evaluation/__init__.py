"""Masurarea acuratetei: transcriere vs partitura de referinta.

Pachetul primeste doua obiecte `Score` si spune cat de mult seamana. Nu importa
nimic din `recognition/` — regula de separare din PROJECT.md §4.2 — deci merge
la fel de bine pe o partitura venita din model si pe una scrisa de mana in cod.

`align` si `metrics` nu depind decat de biblioteca standard. `reference`, care
citeste MusicXML prin `music21`, isi face importul lenes, ca testele de metrica
sa ramana rapide.
"""

from app.evaluation.align import Op, OpKind, align, total_cost
from app.evaluation.metrics import Comparison, VoiceComparison, compare

__all__ = [
    "Comparison",
    "Op",
    "OpKind",
    "VoiceComparison",
    "align",
    "compare",
    "total_cost",
]
