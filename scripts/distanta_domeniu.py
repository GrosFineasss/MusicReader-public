"""Cat de departe e randarea de fotografie, pentru un set de greutati anume.

Intrebarea a aparut in §15.30 si a ramas fara unitate de masura: acelasi sistem,
aceeasi tinta, doua imagini — una randata din glifele tale, alta decupata din
poza. Modelul de la pasul 50 a scris **638** de tokeni pe randare si **43** pe
fotografie, pentru o tinta de 103. Nu invatase sa se opreasca devreme; se
oprea devreme *numai pe hartie*.

Cifra pe care o da scriptul e raportul dintre cele doua. E utila fiindca e
**mai aproape de cauza** decat acuratetea pe pagina: acuratetea amesteca ce nu
citeste modelul cu ce nu poate scrie cititorul, pe cand aici se numara exact ce
a emis modelul, in doua domenii care difera printr-un singur lucru.

    python scripts/distanta_domeniu.py --greutati models/curat-25/checkpoint.pt \
        --sintetic data/synthetic/corpus/pagina3-s01-fineas-0.png \
        --real data/synthetic/antrenare/pagina3_sistem01.png \
        --tinta data/synthetic/antrenare/pagina3_sistem01.krn

Se pot da mai multe din fiecare; se imperecheaza in ordine. Fara `--greutati`
se masoara modelul publicat, adica reperul dinainte de orice antrenare.

Ce **nu** spune: de ce difera. §15.33 a exclus deja metrica, scara si aspectul
cernelii — raman formele. Scriptul da doar cifra care se urmareste in timp.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def tokeni_emisi(motor, cale: Path, plafon: int | None = None) -> tuple[int, bool]:
    """Cati tokeni scrie modelul pe imaginea asta, si daca a fost taiat.

    Se numara `**kern`-ul brut, nu notele din partitura. Diferenta conteaza: la
    pasul 300 modelul emitea 281 de tokeni din care cititorul salva cateva note
    (§15.30), iar „note scrise" ar fi spus ca modelul tace cand de fapt scria
    mult si prost.

    A doua valoare e mai importanta decat pare. Decodarea are un **plafon**
    (`maxlen - 1`, 1276 pe `smt-grandstaff`), iar un model care o ia razna il
    atinge si e taiat acolo. Cifra care iese atunci nu masoara cat a scris
    modelul, ci unde s-a oprit unealta — si arata exact ca o masuratoare
    (§15.36). Cine o citeste fara steag compara un plafon cu o valoare
    adevarata si crede ca a masurat o diferenta.
    """
    from app.recognition.engine import Kind, PageImage

    rezultat = motor.transcribe(PageImage(path=cale, kind=Kind.HANDWRITTEN))
    tokeni = sum(len(staff.kern.split()) for staff in rezultat.staves)
    taiat = plafon is not None and any(staff.tokens >= plafon for staff in rezultat.staves)
    return tokeni, taiat


def raport(perechi: list[tuple[Path, Path, Path | None]], greutati: Path | None) -> None:
    from app.recognition.grandstaff_engine import GrandstaffEngine

    kwargs = {"weights": greutati} if greutati else {}
    motor = GrandstaffEngine(staves_per_system=2, **kwargs)
    motor.load()

    # Acelasi numar ca bucla care taie, nu o formula scrisa a doua oara: daca
    # `--max-tokens` coboara plafonul, si marcajul trebuie sa coboare cu el.
    plafon = motor._decoder.plafon(motor.max_tokens)

    print()
    print(f"{'sistem':28s} {'tinta':>7} {'randare':>10} {'fotografie':>12} {'raport':>8}")
    print("-" * 70)
    rapoarte = []
    taiate = 0
    for sintetic, real, tinta in perechi:
        cate = len(tinta.read_text(encoding="utf-8").split()) if tinta else 0
        pe_randare, taiat_r = tokeni_emisi(motor, sintetic, plafon)
        pe_foto, taiat_f = tokeni_emisi(motor, real, plafon)
        taiate += taiat_r + taiat_f
        # Impartirea se face cu fotografia jos: cifra creste cand distanta
        # creste, ceea ce e sensul in care o citim.
        raport_lor = pe_randare / pe_foto if pe_foto else float("inf")
        # Un raport in care o parte e plafonata nu e o distanta, ci o margine.
        semn = "!" if (taiat_r or taiat_f) else " "
        rapoarte.append(None if (taiat_r or taiat_f) else raport_lor)
        print(f"{real.stem:28s} {cate or '-':>7} "
              f"{str(pe_randare) + ('*' if taiat_r else ''):>10} "
              f"{str(pe_foto) + ('*' if taiat_f else ''):>12} "
              f"{raport_lor:>6.1f}x{semn}")

    bune = [r for r in rapoarte if r is not None]
    print("-" * 70)
    if bune:
        print(f"{'mediu (fara cele taiate)':28s} {'':>7} {'':>10} {'':>12} "
              f"{sum(bune) / len(bune):>6.1f}x")
    else:
        print("niciun raport nefalsificat de plafon")
    print()
    if taiate:
        print(f"* = decodarea a atins plafonul de {plafon} tokeni de model si a fost taiata:")
        print("  cifra spune unde s-a oprit unealta, nu cat a scris modelul (§15.36).")
        print(f"  {taiate} din {2 * len(perechi)} masuratori.")
    print("1,0x = modelul scrie la fel de mult in amandoua domeniile.")
    print("Mai mult = scrie pe randare si tace pe hartie (§15.30).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sintetic", type=Path, nargs="+", required=True)
    parser.add_argument("--real", type=Path, nargs="+", required=True)
    parser.add_argument("--tinta", type=Path, nargs="*", default=[],
                        help="fisierele `.krn` cu tinta; doar pentru context")
    parser.add_argument("--greutati", type=Path, default=None,
                        help="fara ele se masoara modelul publicat")
    args = parser.parse_args()

    if len(args.sintetic) != len(args.real):
        raise SystemExit(
            f"{len(args.sintetic)} imagini sintetice fata de {len(args.real)} reale: "
            "se imperecheaza in ordine, deci trebuie sa fie la fel de multe"
        )
    if args.tinta and len(args.tinta) != len(args.real):
        raise SystemExit(
            f"{len(args.tinta)} tinte fata de {len(args.real)} perechi"
        )

    lipsa = [c for c in (*args.sintetic, *args.real, *args.tinta) if not c.exists()]
    if lipsa:
        raise SystemExit("nu exista: " + ", ".join(str(c) for c in lipsa))

    tinte = list(args.tinta) or [None] * len(args.real)
    raport(list(zip(args.sintetic, args.real, tinte)), args.greutati)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
