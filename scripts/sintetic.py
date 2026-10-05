"""Referinta + glife -> perechi de antrenare sintetice.

Piesa care transforma randarea din §15.17 in date (PROJECT.md §15.18). Scrie
acelasi `date.jsonl` ca `scripts/dataset.py`, deci `finetune.py` si impartirea pe
scriitor merg nemodificate, iar perechile reale si sintetice se pot amesteca in
acelasi set.

    python scripts/sintetic.py --ref data/validation/pagina3.musicxml \\
        --glife models/glife.pkl --out data/antrenare-sintetic --per-stil 5

**Se randeaza cate un sistem odata.** Randand o pagina intreaga si afland pe
urma ce masuri au cazut pe ce sistem, ne-am baza pe layoutul intern al lui
Smashcima, care nu e obligat sa respecte `<print new-system="yes">`. O
imperechere gresita invata modelul alte note decat cele din imagine, si n-ar
avea cine sa observe (§15.7). Asa, imaginea si tinta ies din **aceleasi masuri**,
deci nu se pot contrazice.

**Felierea imaginii se face cu `music21`, nu cu exportul nostru.** Verificat pe
2026-09-03: Smashcima citeste MusicXML-ul scris de `app/musicxml/export.py`, dar
acela are patru parti separate si niciun `<stem>` — alegerea deliberata de la
§11.5. Ar da pagini cu patru portative, pe cand manuscrisele sunt partitura
stransa pe doua. `music21` felieaza pastrand structura originalului, deci pagina
sintetica seamana cu manuscrisul. Formatul de iesire al produsului ramane
neschimbat: nu se strica ce foloseste utilizatorul ca sa fie comod generatorul.

Ruleaza in mediul cu Smashcima, nu in `.venv`-ul proiectului (§15.14).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from app.evaluation.metrics import compare  # noqa: E402
from app.evaluation.reference import load_with_layout  # noqa: E402
from app.music_model.model import measures_slice  # noqa: E402
from app.recognition.bekern import from_kern  # noqa: E402
from app.recognition.kern import parse_open_system  # noqa: E402
from app.recognition.kern_writer import to_kern_text  # noqa: E402
from scripts.dataset import system_bounds  # noqa: E402


def target_for(felie) -> tuple[str, str]:
    """Tinta unui sistem, **si proba ca nu minte**.

    Se scrie in Humdrum si se citeste inapoi cu asezarea feliei; daca ce se
    intoarce nu e identic cu ce s-a trimis, tinta nu descrie muzica din imagine
    si perechea nu are ce cauta in set. E aceeasi regula ca la `dataset.py`,
    care refuza sa scrie cand numarul de sisteme nu se potriveste: mai bine
    niciun set decat un set care minte.
    """
    # Fara marcaje de legato: modelul nu le are in vocabular, deci o tinta care
    # le poarta face sistemul sa cada. Vezi `write_event` pentru cifre.
    text = to_kern_text(felie, ties=False)
    inapoi = parse_open_system(text, layout=felie.staves)
    rezultat = compare(felie, inapoi.score)
    if not rezultat.exact:
        raise ValueError(
            f"tinta nu se inchide: {rezultat.total.matched} din "
            f"{rezultat.total.ref_events} note se regasesc"
        )
    return text, " ".join(from_kern(text))


def slice_musicxml(parsed, first: int, last: int, out: Path) -> Path:
    """Masurile `first..last` dintr-o partitura deja parsata.

    Parsarea se face **o data pe fisier**, de apelant: `music21` citeste o
    referinta de 14 masuri in cateva secunde, iar reluand-o pentru fiecare
    sistem am plati-o de cate sisteme are pagina, degeaba.
    """
    parsed.measures(first, last).write("musicxml", fp=str(out))
    return out


def pin_style(model, style: str) -> None:
    """Fixeaza mana cu care se deseneaza.

    Se restrange **domeniul**, nu doar stilul curent: `Styler` cheama
    `pick_style()` pentru fiecare exemplar sintetizat, si ar rescrie o simpla
    atribuire. Cu domeniul de un singur element, orice alegere cade pe el.
    """
    domain = model.glife_style_domain
    if style not in domain.all_styles:
        raise ValueError(
            f"stil necunoscut: {style} — in repository sunt "
            f"{', '.join(sorted(domain.all_styles))}"
        )
    domain.all_styles = [style]
    domain.current_style = style


# De cate ori se reincearca o randare care a esuat. Nu e o precautie de forma:
# §15.13 a masurat ca Smart App Control blocheaza DLL-uri **la intamplare**, iar
# aceeasi comanda reincercata trece. Cu ~4 ore de generare (§15.18), o rulare de
# noapte fara reincercari are sanse mari sa moara pe la mijloc.
RETRIES = 3
RETRY_WAIT = 2.0

# Cum se recunoaste un simbol pe care **nicio mana nu l-a scris**. Mesajele sunt
# ale lui Smashcima: `RepositoryGlyphSynthesizer` ridica „Unsupported glyph
# label: X" cand nu are eticheta si n-are rezerva, iar `pick` ridica „The glyph
# class X is not present in the symbol repository".
LIPSA_GLIFA = ("Unsupported glyph label", "is not present in the symbol repository")


class GlifaLipsa(Exception):
    """Un simbol care nu exista in repository. **Nu se reincearca.**

    Reincercarea are rost la blocajul intermitent din §15.13, care trece de la
    sine. O eticheta lipsa nu trece niciodata: nimeni n-a scris-o pe foaie.
    Reincercand-o de trei ori s-ar pierde doar timp, iar la ~60 s pe randare
    (§15.18) asta se aduna.

    Cu doi oameni si 108 casute fiecare, cineva **va** sari o casuta. Daca amandoi
    sar aceeasi, simbolul lipseste din tot repository-ul — si e mult mai util sa
    iasa 90% din perechi si sa se spuna ce lipseste, decat sa moara rularea la 3%.
    """

    def __init__(self, eticheta: str):
        self.eticheta = eticheta
        super().__init__(f"niciun scris nu are simbolul {eticheta}")


def eticheta_lipsa(problem: Exception) -> str | None:
    """Eticheta din mesajul de eroare, daca e vorba de o glifa lipsa."""
    mesaj = str(problem)
    for tipar in LIPSA_GLIFA:
        if tipar in mesaj:
            if mesaj.startswith("Unsupported glyph label:"):
                return mesaj.split(":", 1)[1].strip()
            cuvinte = mesaj.split()
            return cuvinte[cuvinte.index("class") + 1] if "class" in cuvinte else "?"
    return None


def render_with_retries(render, model, xml, staves, dpi, note: list[str]):
    """Randeaza, reincercand la esec **trecator**.

    Se separa doua feluri de esec, fiindca cer raspunsuri opuse: blocajul din
    §15.13 trece la a doua incercare, iar o glifa lipsa nu trece niciodata.
    Prima se reincearca, a doua se ridica imediat ca `GlifaLipsa`.

    Nu se ascunde nimic: fiecare reincercare se scrie in `note`, iar la final se
    spune de cate ori a fost nevoie.
    """
    import time

    for attempt in range(1, RETRIES + 1):
        try:
            return render(model, xml, staves=staves, dpi=dpi)
        except Exception as problem:
            eticheta = eticheta_lipsa(problem)
            if eticheta is not None:
                raise GlifaLipsa(eticheta) from problem
            if attempt == RETRIES:
                raise
            note.append(f"{Path(xml).stem}, incercarea {attempt}: {type(problem).__name__}")
            time.sleep(RETRY_WAIT)


def merge_rows(path: Path, produced: list[dict]) -> list[dict]:
    """Randurile vechi si cele noi, **imbinate dupa numele imaginii**.

    Nu se adauga orb, cum face `dataset.py`. Acolo fiecare rulare aduce alta
    pagina, deci adaugarea e corecta; aici o rulare reluata trece prin aceleasi
    imagini si le-ar scrie a doua oara. Masurat inainte de reparatie: 36 de
    randuri pentru 18 imagini — un set in care jumatate din exemple sunt
    duplicate, deci antrenarea le-ar alege de doua ori mai des, tacut.
    """
    by_name = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                by_name[row["image"]] = row
    for row in produced:
        by_name[row["image"]] = row
    return [by_name[name] for name in sorted(by_name)]


def write_rows(path: Path, rows: list[dict]) -> Path:
    """Scrie setul **atomic**: intai alaturi, apoi se muta peste.

    O intrerupere in mijlocul scrierii ar lasa un `date.jsonl` trunchiat, iar
    rularea urmatoare l-ar citi ca pe un set mai mic, fara sa se planga nimeni.
    Aceeasi hotarare ca la checkpoint-urile de antrenare (§15.11).
    """
    temporary = path.with_suffix(".jsonl.nou")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ref", type=Path, nargs="+", required=True,
                        help="referinte MusicXML din care se genereaza")
    parser.add_argument("--glife", type=Path, required=True, help="repository-ul de glife")
    parser.add_argument("--out", type=Path, required=True, help="unde se scriu perechile")
    parser.add_argument("--per-stil", type=int, default=3,
                        help="cate variante se randeaza din fiecare sistem, per mana")
    parser.add_argument("--stiluri", default=None,
                        help="mainile de folosit, despartite prin virgula; implicit toate")
    parser.add_argument("--dpi", type=float, default=300.0)
    parser.add_argument(
        "--de-la-capat", action="store_true",
        help="reface si imaginile care exista deja; implicit se sar",
    )
    args = parser.parse_args()

    from app.training.sinteza import build_model, load_repository, render_system

    repository = load_repository(args.glife)
    toate = sorted(repository.get_all_styles())
    stiluri = (
        [s.strip() for s in args.stiluri.split(",") if s.strip()]
        if args.stiluri
        else toate
    )
    print(f"glife: {len(toate)} maini ({', '.join(toate)}) | se folosesc {len(stiluri)}")

    # Un model per mana, construit o data. `pin_style` restrange domeniul, deci
    # modelul nu mai poate desena cu alta.
    models = {}
    for style in stiluri:
        model = build_model(repository, dpi=args.dpi)
        pin_style(model, style)
        models[style] = model

    args.out.mkdir(parents=True, exist_ok=True)
    temp = args.out / "_felii"
    temp.mkdir(exist_ok=True)

    rows: list[dict] = []
    refuzate: list[str] = []
    reincercari: list[str] = []
    lipsuri: dict[str, list[str]] = {}
    sarite = 0

    import music21

    for ref in args.ref:
        parsed = music21.converter.parse(str(ref))
        score, starts, pages = load_with_layout(ref)
        if len(pages) > 1:
            print(f"{ref.name}: {len(pages)} pagini — se genereaza din toate sistemele")
        bounds = system_bounds(starts, score.measure_count)
        print(f"{ref.name}: {score.measure_count} masuri, {len(bounds)} sisteme")

        for index, (first, last) in enumerate(bounds, 1):
            felie = measures_slice(score, first, last)
            try:
                _, tokens = target_for(felie)
            except ValueError as problema:
                refuzate.append(f"{ref.stem} sistem {index} (m{first}-{last}): {problema}")
                continue

            xml = slice_musicxml(parsed, first, last, temp / f"{ref.stem}-s{index:02d}.musicxml")
            staves = len(felie.staves) or 2

            lipsa_aici = None
            for style in stiluri:
                if lipsa_aici:
                    break
                model = models[style]
                for varianta in range(args.per_stil):
                    name = f"{ref.stem}-s{index:02d}-{style}-{varianta}"
                    row = {
                        "image": f"{name}.png",
                        "transcription": tokens,
                        "staves": staves,
                        "writer": style,
                    }

                    # **Reluabil.** O randare costa ~1 minut si nu depinde de
                    # rezolutie — costul e in sinteza, nu in rasterizare
                    # (masurat: 64 s la 300 dpi, 48 s la 150). Un set de cateva
                    # mii de perechi e deci o rulare de noapte, care se poate
                    # intrerupe: fie de tine, fie de blocajul intermitent din
                    # §15.13. Ce s-a desenat deja nu se mai deseneaza.
                    if (args.out / f"{name}.png").exists() and not args.de_la_capat:
                        rows.append(row)
                        sarite += 1
                        continue

                    # Varianta vine din samanta, nu dintr-un model nou: aceeasi
                    # muzica, alte exemplare de glife din aceeasi mana. Un model
                    # per varianta ar plati constructia de zeci de ori degeaba.
                    model.rng.seed(hash((ref.stem, index, style, varianta)) % (2**31))
                    try:
                        imagine, _ = render_with_retries(
                            render_system, model, xml, staves, args.dpi, reincercari
                        )
                    except GlifaLipsa as problema:
                        # Se sare **sistemul intreg**, nu doar varianta: rezerva
                        # intre maini e globala (`glyphs_by_label`), deci ce
                        # lipseste lipseste pentru toti. Incercandu-l pentru
                        # fiecare stil s-ar pierde zeci de minute degeaba.
                        lipsa_aici = problema.eticheta
                        lipsuri.setdefault(problema.eticheta, []).append(
                            f"{ref.stem} sistem {index}"
                        )
                        break

                    imagine.save(args.out / f"{name}.png")
                    rows.append(row)
            if lipsa_aici:
                print(f"  sistem {index}: masurile {first}-{last} — SARIT, "
                      f"niciun scris n-are {lipsa_aici}")
            else:
                print(f"  sistem {index}: masurile {first}-{last}, "
                      f"{len(tokens.split())} tokeni, "
                      f"{len(stiluri) * args.per_stil} imagini")

    if lipsuri:
        print()
        print(f"{len(lipsuri)} simboluri lipsesc din toate scrisurile:")
        for eticheta, unde in sorted(lipsuri.items()):
            print(f"  {eticheta}: a oprit {len(unde)} sisteme")
        print("  Completeaza casutele alea pe foaie si ruleaza din nou —")
        print("  perechile deja desenate se sar, deci nu se pierde nimic.")

    if reincercari:
        print()
        print(f"{len(reincercari)} randari au cerut o reincercare (§15.13):")
        for nota in reincercari[:5]:
            print(f"  {nota}")

    if sarite:
        print()
        print(f"{sarite} imagini existau deja si s-au sarit "
              "(`--de-la-capat` le reface)")

    if refuzate:
        print(f"\n{len(refuzate)} sisteme refuzate:")
        for problema in refuzate[:5]:
            print(f"  {problema}")

    if not rows:
        raise SystemExit("nicio pereche produsa")

    rows = merge_rows(args.out / "date.jsonl", rows)
    lines = write_rows(args.out / "date.jsonl", rows)

    lungimi = sorted(len(row["transcription"].split()) for row in rows)
    per_stil: dict[str, int] = {}
    for row in rows:
        per_stil[row["writer"]] = per_stil.get(row["writer"], 0) + 1
    print(
        f"\n{len(rows)} perechi in {lines}"
        f"\n  per mana: {per_stil}"
        f"\n  tokeni: mediana {lungimi[len(lungimi) // 2]}, "
        f"min {lungimi[0]}, max {lungimi[-1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
