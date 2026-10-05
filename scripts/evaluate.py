"""Cat de bine a citit modelul o pagina, fata de o partitura de referinta.

Pana acum am comparat modelele intre ele (`try_grandstaff.py`) si am numarat ce
a iesit — masuri, note, tokeni nerecunoscuti. Niciuna dintre cifrele acelea nu
spune **cate note sunt corecte**, fiindca pentru asta e nevoie de ground truth.
Scriptul asta il foloseste.

    python scripts/evaluate.py --ref referinta.musicxml --hyp transcris.musicxml
    python scripts/evaluate.py --ref referinta.musicxml --imagine pagina.png
    python scripts/evaluate.py --ref r.musicxml --imagine p.png --portative-pe-sistem 2 --curata

Pe scris de mana, ultima forma e singura care da o cifra comparabila cu ce
foloseste conducta: gruparea in sisteme se declara (§14.6), iar pagina se
curata de versuri inainte de model (§14.7).

Referinta o scrii in MuseScore si o exporti ca MusicXML. `--imagine` ruleaza
intai motorul si compara ce a iesit; e o comoditate, nu o a doua conducta —
scriptul nu contine nicio logica de recunoastere.

`--json` scrie aceleasi cifre intr-un fisier, ca doua rulari despartite de o
schimbare de model sa poata fi puse una langa alta. Fara asta, „mi se pare mai
bine" ramane singura evaluare disponibila, si e cea pe care §9 o refuza.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

# Numele de note si mesajele contin diacritice pe care codificarea implicita a
# consolei Windows nu le poate scrie; fara asta, procesul moare fara mesaj.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from app.evaluation.metrics import Comparison, VoiceComparison, compare  # noqa: E402
from app.music_model.model import Score, measures_slice  # noqa: E402
from app.staff_detection.staves import parse_grouping, parse_layout  # noqa: E402


def percent(value: Fraction) -> str:
    return f"{float(value) * 100:.1f}%"


def _rows(comparison: Comparison) -> list[VoiceComparison]:
    return [*comparison.voices, comparison.total]


COLUMNS = (
    ("voce", lambda v: v.part_id),
    # Peste tot, `note+pauze`. Fara despartirea asta, cineva care citeste
    # „corecte 16" langa „note ref 182" calculeaza singur 8,8% — si cifra
    # adevarata e 3,2%, fiindca 11 din cele 16 sunt perechi de pauze (§15.42).
    ("note ref", lambda v: f"{v.ref_notes}+{v.ref_rests}p"),
    ("scrise", lambda v: f"{v.hyp_notes}+{v.hyp_rests}p"),
    ("corecte", lambda v: f"{v.matched_notes}+{v.matched_rests}p"),
    ("pitch gr.", lambda v: str(v.pitch_wrong)),
    ("ritm gr.", lambda v: str(v.duration_wrong)),
    ("ambele", lambda v: str(v.both_wrong)),
    ("lipsa", lambda v: str(v.missing)),
    ("in plus", lambda v: str(v.extra)),
    ("masuri", lambda v: f"{v.measures_exact}/{v.ref_measures}"),
    ("note%", lambda v: percent(v.note_accuracy)),
    ("precizie", lambda v: percent(v.precision)),
    ("pitch%", lambda v: percent(v.pitch_accuracy)),
    ("ritm%", lambda v: percent(v.rhythm_accuracy)),
    ("SER", lambda v: f"{float(v.ser):.3f}"),
)


def report(comparison: Comparison) -> None:
    rows = _rows(comparison)
    widths = [max(len(name), *(len(get(row)) for row in rows)) for name, get in COLUMNS]

    print()
    print("  ".join(name.ljust(w) for (name, _), w in zip(COLUMNS, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        if row.part_id == "TOTAL":
            print("  ".join("-" * w for w in widths))
        print("  ".join(get(row).ljust(w) for (_, get), w in zip(COLUMNS, widths)))

    if comparison.structure:
        print("\nstructura:")
        for item in comparison.structure:
            print(f"  ! {item}")

    for voice in comparison.voices:
        for item in voice.context:
            # Un singur rand din astea explica de obicei zeci de note gresite.
            print(f"  ! {voice.part_id}: {item}")

    # Masuri care nu se inchid la metrica — regula exista deja in validator, si
    # e utila pe amandoua partiturile: daca referinta insasi are masuri
    # incomplete, cifrele de mai sus judeca modelul pentru greseala noastra.
    print(f"\nmasuri care nu se inchid — referinta: {len(comparison.ref_findings)}, "
          f"transcriere: {len(comparison.hyp_findings)}")

    _report_signature(comparison)
    _report_quiet(comparison)

    print("\ncorect = inaltime SI durata (§9). SER = (substituiri+lipsa+in plus)/note referinta.")
    print("`note+pauze`: pauzele se numara separat fiindca doua pauze de aceeasi")
    print("  durata ies „identice\" — niciuna n-are inaltime — si ar umfla cifrele (§15.42).")
    print("note% = note corecte / note din referinta (cat din MUZICA s-a citit).")
    print("precizie = note corecte / note scrise. SER ramane pe evenimente, cu pauze")
    print("  cu tot, fiindca asa il raporteaza literatura si doar asa e comparabil.")


def _report_signature(comparison: Comparison) -> None:
    """*Cum* greseste, nu doar cat — vezi `metrics.Signature`.

    Un decalaj constant de inaltime ar fi un bug determinist (cheie sau octava),
    nu ceva ce repara antrenarea; o confuzie de durate care se repeta spune ce
    anume nu recunoaste modelul pe pagina.
    """
    signature = comparison.signature
    if signature.durations:
        print("\ndurate confundate (referinta -> transcriere), cele mai dese:")
        for (was, became), count in signature.durations.most_common(5):
            print(f"  {was} patrimi citit ca {became}: {count}")
    if signature.pitches:
        total = sum(signature.pitches.values())
        exact = signature.pitches.get(0, 0)
        offsets = sorted(signature.pitches.items(), key=lambda item: -item[1])[:5]
        spread = ", ".join(f"{step:+d}: {count}" for step, count in offsets)
        print(f"\ndecalaj de inaltime, in trepte (din {total}): {spread}")
        print(f"  aceeasi treapta, alta durata: {exact}")


def _report_quiet(comparison: Comparison) -> None:
    """Spune raspicat cand cifrele de deasupra sunt ale unui model care tace.

    SER-ul scade cand modelul nu mai inventeaza la fel de bine ca atunci cand
    citeste mai bine, si nu le deosebeste: un model care nu scrie **nimic**
    iese fix la SER 1,000, iar unul care scrie 14 note din 182 iese sub 1 fara
    sa fi citit pagina (§15.28). Coloana `scrise` arata asta oricui se uita la
    ea; randul de aici e pentru cine se uita doar la SER.

    Pragul e la jumatate din referinta fiindca sub el nu mai e vorba de o
    citire slaba, ci de o transcriere din care lipseste muzica.
    """
    total = comparison.total
    if total.ref_events == 0 or 2 * total.hyp_events >= total.ref_events:
        return
    print()
    print(f"  ! note scrise: {total.hyp_events}, fata de {total.ref_events} in "
          "referinta — din pagina lipseste muzica.")
    print("    SER-ul de mai sus e mic fiindca modelul tace, nu fiindca citeste "
          "bine (§15.28);")
    print(f"    cifra de comparat aici e `note%` ({percent(total.note_accuracy)}).")


def taie_pagina(reference: Score, pages: list[int], pagina: int) -> tuple[Score, int, int]:
    """Bucata de referinta care e pe pagina ceruta, plus marginile ei.

    `pages` sunt numerele primelor masuri ale fiecarei pagini, asa cum le da
    `reference.load_with_layout`. Ultima pagina se termina la ultima masura.

    Exista fiindca o referinta de mai multe pagini masurata intreaga fata de o
    singura poza da o cifra fara sens: tot ce e pe celelalte pagini iese
    `lipsa`. Aceeasi taiere o face si `dataset.py`, cu aceleasi margini.
    """
    if not 1 <= pagina <= len(pages):
        raise SystemExit(f"referinta are {len(pages)} pagini, ceruta e {pagina}")
    first = pages[pagina - 1]
    last = pages[pagina] - 1 if pagina < len(pages) else reference.measure_count
    return measures_slice(reference, first, last), first, last


def as_dict(comparison: Comparison, ref: str, hyp: str) -> dict:
    def voice_dict(voice: VoiceComparison) -> dict:
        return {
            "voce": voice.part_id,
            "note_referinta": voice.ref_events,
            "note_transcriere": voice.hyp_events,
            "corecte": voice.matched,
            "corecte_note": voice.matched_notes,
            "corecte_pauze": voice.matched_rests,
            "pitch_gresit": voice.pitch_wrong,
            "ritm_gresit": voice.duration_wrong,
            "ambele_gresite": voice.both_wrong,
            "lipsa": voice.missing,
            "in_plus": voice.extra,
            "masuri_referinta": voice.ref_measures,
            "masuri_transcriere": voice.hyp_measures,
            "masuri_exacte": voice.measures_exact,
            "masuri_lipsa": voice.measures_missing,
            "masuri_in_plus": voice.measures_extra,
            "acuratete_note": float(voice.note_accuracy),
            "acuratete_pitch": float(voice.pitch_accuracy),
            "acuratete_ritm": float(voice.rhythm_accuracy),
            "ser": float(voice.ser),
            "context": list(voice.context),
        }

    return {
        "referinta": ref,
        "transcriere": hyp,
        "voci": [voice_dict(v) for v in comparison.voices],
        "total": voice_dict(comparison.total),
        "structura": list(comparison.structure),
        "masuri_neinchise_referinta": [str(f) for f in comparison.ref_findings],
        "masuri_neinchise_transcriere": [str(f) for f in comparison.hyp_findings],
        "identice": comparison.exact,
        "durate_confundate": {f"{a}->{b}": n for (a, b), n in comparison.signature.durations.items()},
        "decalaj_inaltime": {str(k): v for k, v in sorted(comparison.signature.pitches.items())},
    }


def transcribe(
    image: Path,
    kind_name: str,
    max_tokens: int | None,
    staves_per_system: int | None = None,
    layout: list[list[str]] | None = None,
    weights: Path | None = None,
    clean: bool = False,
    repo: str | None = None,
    voices_per_staff: int | None = None,
    strict_structure: bool = False,
    margin_spaces: float | None = None,
    beam: int = 1,
) -> Score:
    """Ruleaza motorul pe o imagine si intoarce partitura obtinuta.

    Importurile sunt inauntru: fara `--imagine`, scriptul nu are nevoie de
    torch, si nu are de ce sa astepte incarcarea lui.

    `staves_per_system` si `clean` sunt aceleasi optiuni ca in `transcribe.py`
    si `batch.py`. Fara ele, scriptul de masurat nu putea reproduce singura
    configuratie care merge pe manuscris (§14.6 si §14.7) — deci masura o cifra
    obtinuta altfel decat cea pe care o folosim.
    """
    from app.recognition.engine import Kind, PageImage
    from app.recognition.grandstaff_engine import GrandstaffEngine

    kwargs = {"repo": repo} if repo else {}
    if voices_per_staff:
        kwargs["voices_per_staff"] = voices_per_staff
    if strict_structure:
        kwargs["strict_structure"] = True
    if margin_spaces is not None:
        kwargs["margin_spaces"] = margin_spaces
    if layout is not None:
        kwargs["layout"] = layout
    if weights is not None:
        kwargs["weights"] = weights
    if beam > 1:
        kwargs["beam"] = beam
    engine = GrandstaffEngine(
        max_tokens=max_tokens, staves_per_system=staves_per_system, clean=clean, **kwargs
    )
    engine.load()
    result = engine.transcribe(PageImage(path=image, kind=Kind(kind_name)))
    for problem in result.problems:
        print(f"  ! {problem}")
    for error in result.errors:
        print(f"  ! {error}")
    if result.score is None:
        raise SystemExit("asamblarea a esuat: nu exista partitura de comparat")
    return result.score


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ref", type=Path, required=True, help="MusicXML de referinta")
    parser.add_argument("--hyp", type=Path, help="MusicXML produs de conducta")
    parser.add_argument("--imagine", type=Path, help="poza de transcris inainte de comparatie")
    parser.add_argument("--kind", choices=("printed", "handwritten"), default="handwritten")
    parser.add_argument("--max-tokens", type=int, default=None)
    # Aceleasi nume ca in `transcribe.py`, `batch.py` si `try_grandstaff.py`.
    parser.add_argument(
        "--portative-pe-sistem", type=parse_grouping, default=None,
        metavar="N|N,N,...",
        help="cate portative are un sistem: un numar (`2`) sau cate unul pe "
             "sistem (`2,2,2,3,3`)",
    )
    parser.add_argument(
        "--asezare", type=parse_layout, default=None, metavar="S,A/T,B",
        help="ce voci stau pe fiecare portativ, de sus in jos; pentru o cantare "
             "cu solo pe portativ separat: `X1/S,A/T,B`",
    )
    parser.add_argument(
        "--greutati", type=Path, default=None, metavar="CHECKPOINT",
        help="checkpoint de la `scripts/finetune.py`, aplicat peste greutatile publicate",
    )
    parser.add_argument(
        "--pagina", type=int, default=None, metavar="N",
        help="a cata pagina a referintei e in poza (1 = prima). O referinta de "
             "mai multe pagini masurata intreaga fata de o singura poza da o "
             "cifra fara sens: tot ce e pe celelalte pagini iese `lipsa`",
    )
    parser.add_argument("--curata", action="store_true")
    parser.add_argument("--margine", type=float, default=None,
                        help="cat se pastreaza in jurul portativului, in spatii (implicit 1.0)")
    parser.add_argument("--structura", action="store_true",
                        help="decodarea nu mai poate scrie randuri cu alt numar de campuri")
    parser.add_argument("--voci-pe-portativ", type=int, default=None,
                        help="plafon de voci pe portativ; decodarea nu mai deschide peste el")
    parser.add_argument("--model", default=None,
                        help="alte greutati SMT, ex. antoniorv6/smt-camera-grandstaff")
    parser.add_argument("--out", type=Path,
                        help="unde se scrie MusicXML-ul transcrierii, ca sa poata fi deschis in MuseScore")
    parser.add_argument("--json", type=Path, help="unde se scriu cifrele, pentru comparatii viitoare")
    args = parser.parse_args()

    if args.out is not None and args.imagine is None:
        parser.error("--out are sens numai cu --imagine: cu --hyp, fisierul exista deja")

    if (args.hyp is None) == (args.imagine is None):
        parser.error("da exact una dintre --hyp si --imagine")

    from app.evaluation.reference import load, load_with_layout

    reference = load(args.ref)
    if args.pagina is not None:
        _, _, pages = load_with_layout(args.ref)
        reference, first, last = taie_pagina(reference, pages, args.pagina)
        print(f"pagina {args.pagina} din referinta: masurile {first}-{last}")
    if args.imagine is not None:
        hypothesis = transcribe(
            args.imagine,
            args.kind,
            args.max_tokens,
            staves_per_system=args.portative_pe_sistem,
            layout=args.asezare,
            weights=args.greutati,
            clean=args.curata,
            repo=args.model,
            voices_per_staff=args.voci_pe_portativ,
            strict_structure=args.structura,
            margin_spaces=args.margine,
        )
        source = str(args.imagine)
        if args.out is not None:
            from app.musicxml import export

            args.out.parent.mkdir(parents=True, exist_ok=True)
            export.write(hypothesis, args.out)
            print(f"transcriere scrisa in {args.out}")
    else:
        # O masura scurta intr-o **transcriere** e o greseala pana la proba
        # contrarie, nu o anacruza: MusicXML nu pastreaza diferenta.
        hypothesis = load(args.hyp, trust_short_measures=False)
        source = str(args.hyp)

    comparison = compare(reference, hypothesis)
    report(comparison)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(as_dict(comparison, str(args.ref), source), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\ncifre scrise in {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
