"""Referinta -> aceeasi muzica, urcata sau coborata. Mai multe tinte, gratis.

Problema pe care o rezolva: pentru antrenare avem **13 sisteme** din doua
referinte, si atat. Randand fiecare in mai multe variante de stil obtii mai multe
imagini, dar **aceeasi muzica** — modelul vede a opta oara aceleasi note. Urcand
partitura cu doua semitonuri obtii alte note, deci **alta tinta**, la fel de
valida muzical.

    python scripts/transpune.py --ref data/validation/pagina3.musicxml \
        --semitonuri -3 -2 2 3 --out data/validation/transpuse

Iesirea se da apoi lui `sintetic.py`, care primeste mai multe referinte:

    python scripts/sintetic.py --ref data/validation/transpuse/*.musicxml ...

**Ambitusul e o straja adevarata, nu decor.** Muzica de cor sta intre Mi2 si La5
(§ memoria de proiect). O linie de bas coborata cu cinci semitonuri iese sub ce
poate canta un om, deci nu seamana cu nimic din manuscrise — iar modelul ar
invata inaltimi care nu apar niciodata acolo. Ce iese din ambitus se **refuza si
se spune**, nu se scrie tacut.

**A doua straja: scrierea rezultata.** Transpozitia cromatica poate produce o
partitura corecta ca sunet si stricata ca notatie. Masurat pe cele doua referinte
reale, mutarea cu **un singur semiton** a dat, la una, o armura de 8 diezi (nu
exista; maximul e 7), iar la cealalta note a caror alteratie scrisa nu se mai
potriveste cu armura. Cu doua sau trei semitonuri, amandoua ies curate.

Deci nu se tine o lista de semitonuri „bune" — se **verifica rezultatul**: se
citeste inapoi si se trece prin validator, iar ce a capatat erori pe care
originalul nu le avea se sterge si se raporteaza. Asa straja tine si pentru
referintele care vor veni, nu doar pentru astea doua.

**A treia straja: ritmul.** `music21` isi „repara" notatia cand scrie MusicXML,
si repararea nu e neutra — atinge si ce n-a fost cerut. Masurat pe corpusul din
03.09: ritmul a fost rescris la **90 din 195 de perechi transpuse**, prin pauze
alaturate contopite (`4r 8r 16r` -> `4..r`) si note punctate care in original nu
erau. S-au vazut doar 15, cele care au iesit din vocabularul modelului si au fost
aruncate la antrenare; celelalte 75 au intrat tacut, invatand modelul alt ritm
decat arata pagina. Scrierea se face acum cu `makeNotation=False`, iar rezultatul
se verifica: duratele fiecarei voci trebuie sa iasa identice cu ale originalului.

Ce ramane pentru `scripts/verifica_referinta.py`: dus-intorsul pe fiecare sistem.
Ruleaz-o pe ce iese de aici inainte sa generezi mii de pagini.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from app.evaluation.reference import load_with_layout  # noqa: E402
from app.validation.rules import Severity, validate  # noqa: E402

# Mi2 - La5, ambitusul de cor confirmat de utilizator.
MIN_MIDI = 40
MAX_MIDI = 81


def ambitus(score) -> tuple[int, int]:
    """Cea mai joasa si cea mai inalta nota, in numere MIDI."""
    inaltimi = [n.pitch.midi for n in score.recurse().notes if hasattr(n, "pitch")]
    if not inaltimi:
        raise ValueError("partitura n-are nicio nota")
    return min(inaltimi), max(inaltimi)


def transpune(cale: Path, semitonuri: int):
    """Partitura urcata/coborata, si ambitusul rezultat."""
    from music21 import converter, interval

    with warnings.catch_warnings():
        # `music21` avertizeaza despre elemente pe care nu le foloseste; ce
        # conteaza cu adevarat prinde `verifica_referinta.py`.
        warnings.simplefilter("ignore")
        score = converter.parse(str(cale))
        mutat = score.transpose(interval.Interval(semitonuri))
    return mutat, ambitus(mutat)


def scrie(mutat, cale: Path) -> None:
    """Scrie transpozitia fara sa lase music21 sa rescrie notatia.

    `music21` isi „repara" notatia la export, si repararea vine la pachet:
    lipeste pauze alaturate, puncteaza note — deci schimba **ritmul** unei
    transpozitii care ar fi trebuit sa atinga doar inaltimile (§15.29).
    `makeNotation=False` o opreste.

    Dar oprita de tot, se pierde si partea care chiar trebuia: transpozitia
    cromatica lasa note a caror alteratie scrisa nu se mai potriveste cu armura,
    iar straja de notatie le refuza pe drept. Masurat pe pagina 3 urcata cu un
    semiton: **56 de erori** fara nimic, **2** cu `makeAccidentals` chemata pe
    fiecare parte — si ritmul neatins in amandoua cazurile.

    Deci nu „scriere implicita" sau „scriere goala", ci exact pasul care
    trebuie, chemat singur.
    """
    for parte in mutat.parts:
        parte.makeAccidentals(inPlace=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mutat.write("musicxml", fp=str(cale), makeNotation=False)


def ritm(cale: Path) -> dict[str, list[str]]:
    """Duratele fiecarei voci, in ordine — semnatura ritmica a unei partituri.

    Transpozitia muta inaltimi si nimic altceva. Orice diferenta aici inseamna
    ca pe drum a rescris altcineva notatia, iar tinta nu mai descrie ce arata
    imaginea randata din ea.

    Se citeste cu acelasi incarcator ca restul conductei, din acelasi motiv ca
    la `erori_de_notatie`: doua citiri diferite s-ar putea contrazice.
    """
    score, _, _ = load_with_layout(cale)
    return {
        part.id: [
            str(event.duration) for measure in part.measures for event in measure.events
        ]
        for part in score.parts
    }


def erori_de_notatie(cale: Path) -> tuple[int, str]:
    """Cate erori are notatia din fisier, si prima dintre ele.

    Se citeste cu **acelasi** incarcator ca restul conductei si se trece prin
    **acelasi** validator. O a doua verificare, scrisa separat, ar putea sa nu fie
    de acord cu prima — si atunci n-am mai sti care are dreptate.
    """
    try:
        score, _, _ = load_with_layout(cale)
    except Exception as eroare:
        return -1, f"{type(eroare).__name__}: {eroare}"
    erori = [f for f in validate(score) if f.severity is Severity.ERROR]
    return len(erori), str(erori[0]) if erori else ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ref", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--semitonuri", type=int, nargs="+", default=[-3, -2, -1, 1, 2, 3],
        help="cu cate semitonuri; negativ = mai jos (implicit: -3..3 fara 0)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scrise, refuzate = [], []

    for cale in args.ref:
        original, (jos, sus) = transpune(cale, 0)
        erori_originale, _ = erori_de_notatie(cale)
        ritm_original = ritm(cale)
        print(f"{cale.name}: ambitus {jos}-{sus}, {erori_originale} erori de notatie")

        for n in args.semitonuri:
            if n == 0:
                continue
            mutat, (jos, sus) = transpune(cale, n)
            iesire = args.out / f"{cale.stem}-t{n:+d}.musicxml"

            if jos < MIN_MIDI or sus > MAX_MIDI:
                incotro = "prea jos" if jos < MIN_MIDI else "prea sus"
                print(f"  {n:+d}: REFUZAT — {incotro} ({jos}-{sus}, cor: {MIN_MIDI}-{MAX_MIDI})")
                refuzate.append((cale.name, n))
                continue

            scrie(mutat, iesire)
            # Scris, apoi verificat, apoi eventual sters: validatorul lucreaza pe
            # fisier, prin acelasi drum ca restul conductei.
            erori, prima = erori_de_notatie(iesire)
            if erori == -1:
                print(f"  {n:+d}: REFUZAT — nu se mai citeste ({prima})")
                iesire.unlink()
                refuzate.append((cale.name, n))
                continue
            if erori > erori_originale:
                print(f"  {n:+d}: REFUZAT — {erori} erori de notatie fata de "
                      f"{erori_originale} in original ({prima})")
                iesire.unlink()
                refuzate.append((cale.name, n))
                continue

            ritm_nou = ritm(iesire)
            if ritm_nou != ritm_original:
                unde = next(
                    (v for v in ritm_original if ritm_nou.get(v) != ritm_original[v]),
                    "?",
                )
                print(f"  {n:+d}: REFUZAT — ritmul s-a schimbat la scriere "
                      f"(vocea {unde}: {len(ritm_original.get(unde, []))} durate "
                      f"-> {len(ritm_nou.get(unde, []))})")
                iesire.unlink()
                refuzate.append((cale.name, n))
                continue

            print(f"  {n:+d}: {iesire.name} (ambitus {jos}-{sus})")
            scrise.append(iesire)

    print(f"\n{len(scrise)} scrise, {len(refuzate)} refuzate")
    if scrise:
        print("Verifica-le inainte de generare:")
        print(f"  python scripts/verifica_referinta.py {args.out}/*.musicxml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
