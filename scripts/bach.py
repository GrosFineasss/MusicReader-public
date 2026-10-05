"""Coralele lui Bach -> perechi de antrenare tiparite + un set de validare.

    python scripts/bach.py --out data/bach
    python scripts/bach.py --out data/bach --limita 5     # proba, pe cinci corale

Ce iese in `--out`:

- `xml/<coral>-p<k>.musicxml` — fiecare pagina, in partitura stransa, cu
  rupturile de sistem scrise. E si sursa randarii, si referinta paginii.
- `png/<coral>-p<k>-1.png` — pagina randata de MuseScore, 200 dpi.
- `antrenare/date.jsonl` + decupajele — perechile, in forma din `dataset.py`.
  Eticheta `writer` e coralul, ca impartirea sa se faca pe piesa.
- `set-validare.json` — paginile coralelor **tinute deoparte**, in formatul lui
  `masoara.py`. Nu ajung niciodata in `date.jsonl`.
- `raport.json` — ce s-a refuzat si de ce.

**E reluabil.** O pagina deja trecuta in `facute.txt` se sare, iar o randare
existenta nu se reface — aceeasi hotarare ca lotul (§12.2).

Tinta si imaginea vin din aceeasi partitura (vezi `app/training/bach.py`), deci
singurul loc unde se pot desparti e decuparea: daca detectorul gaseste alt numar
de sisteme decat s-au scris, pagina se refuza intreaga, ca in `dataset.py`.

Al doilea loc e curatarea: decupajele se iau din pagina **doar indreptata**, nu
si curatata de text (vezi `fara_stergere`). `--decupaj-curatat` reface felul
primelor loturi (`data/bach`, `data/bach2`).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from app.music_model.model import Score  # noqa: E402
from app.training.bach import (  # noqa: E402
    CU_SOLO,
    SISTEME_PE_PAGINA,
    SISTEME_PE_PAGINA_CU_SOLO,
    coral_satb,
    cu_solo,
    masuri_cu_acorduri,
    musicxml_strans,
    pagini,
    repetitii,
)

# Stilul cu care se randeaza: ultimul sistem intins pe toata latimea, altfel
# detectorul nu-l vede (vezi comentariul din fisier).
STIL = ROOT / "scripts" / "bach-stil.mss"

# Cate fisiere primeste MuseScore intr-o singura pornire. O pornire costa
# secunde; un lot prea mare pierde tot, daca pica la mijloc.
RANDARI_PE_LOT = 60


def corale() -> list[Path]:
    """Fisierele corpusului, in ordine stabila."""
    from music21 import corpus

    return sorted(
        (path for path in corpus.getComposer("bach") if path.suffix == ".mxl"),
        key=lambda path: path.name,
    )


def cheie_solo(validare: bool, score: Score) -> tuple:
    """Cine poate da solo cui: aceeasi parte (antrenare/validare), armura si metrica."""
    _, key, time = score.part("S").context_at(1)
    return (validare, key.fifths, time.beats, time.beat_type)


def tinute_deoparte(nume: list[str], procent: int, samanta: int = 0) -> set[str]:
    """Ce corale nu se antreneaza niciodata. Stabil: acelasi set, aceeasi alegere."""
    alese = random.Random(samanta).sample(sorted(nume), len(nume))
    return set(alese[: max(1, round(len(nume) * procent / 100))])


def randeaza(
    musescore: Path, lucrari: list[tuple[Path, Path]], dpi: int, stil: Path | None = STIL
) -> None:
    """MusicXML -> PNG, cate un lot de MuseScore pe rand."""
    for start in range(0, len(lucrari), RANDARI_PE_LOT):
        lot = lucrari[start : start + RANDARI_PE_LOT]
        job = lot[0][1].parent / "job.json"
        job.write_text(
            json.dumps([{"in": str(xml), "out": str(png)} for xml, png in lot]),
            encoding="utf-8",
        )
        comanda = [str(musescore), "-r", str(dpi), "-j", str(job)]
        if stil is not None:
            comanda[1:1] = ["-S", str(stil)]
        subprocess.run(
            comanda,
            check=True, capture_output=True, timeout=1800,
        )
        print(f"  randate {min(start + RANDARI_PE_LOT, len(lucrari))}/{len(lucrari)}")


def fara_stergere(png: Path, unghi: float, crops: list) -> list:
    """Aceleasi chenare, taiate din pagina indreptata, dar **nestearsa** de text.

    Paginile de aici n-au versuri, deci curatarea n-are text de sters; sterge in
    schimb simboluri pe care tinta le are: nota intreaga pe a doua linie
    suplimentara si alteratia ei, pauza vocii a doua coborata sub portativ, cifra
    unui triolet (§15.60 — una din zece pagini PDMX din esantion). Detectia ramane pe
    pagina curatata, ca numarul de sisteme sa se verifice la fel; doar pixelii se
    iau de dinainte de stergere.

    Pagina se reface pe acelasi drum ca in `clean_page` (Otsu, apoi rotirea cu
    unghiul gasit acolo), deci curatarea poate doar sa **lipseasca** cerneala
    fata de ea. Daca vreun pixel de cerneala din decupajul curatat lipseste aici,
    drumurile s-au despartit, iar chenarele n-ar mai cadea peste aceeasi muzica.
    """
    import cv2
    import numpy as np
    from PIL import Image

    from app.preprocessing import binarize, rotate

    pagina = np.array(Image.open(png).convert("L"))
    dreapta = cv2.bitwise_not(rotate(binarize(pagina), unghi))
    brute = []
    for (x1, y1, x2, y2), curatat, portative in crops:
        brut = dreapta[y1:y2, x1:x2]
        if np.any((curatat < 128) & (brut >= 128)):
            raise ValueError(f"{png.name}: decupajul nesters nu contine cerneala celui curatat")
        brute.append(((x1, y1, x2, y2), brut, portative))
    return brute


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--procent-validare", type=int, default=10)
    parser.add_argument("--limita", type=int, default=None, help="doar primele N corale")
    parser.add_argument(
        "--lista", type=Path, default=None,
        help="piesele de convertit, in locul corpusului Bach: JSON cu `cale` si `id` "
             "(de exemplu `acceptate.json` de la `scripts/pdmx_satb.py`). Se convertesc "
             "cu `satb.cor_satb`, care stie si partitura stransa si imnul scris pe pian",
    )
    parser.add_argument(
        "--dpi", default="200", metavar="N[,N...]",
        help="rezolutiile la care se randeaza fiecare pagina. Mai multe (`150,200,300`) "
             "dau aceeasi muzica la alta grosime de linie: antrenat doar la 200, modelul "
             "pierdea pe `pagina3` scanata la 150 (55 -> 48 de note, §15.59)",
    )
    parser.add_argument(
        "--impartire-veche", action="store_true",
        help="impartirea si randarea primului lot (`data/bach`, 2026-10-02): fara "
             "plafonul de coloane si fara stilul care intinde ultimul sistem",
    )
    parser.add_argument(
        "--stil", choices=("voci", "acorduri"), default="voci",
        help="`acorduri`: S+A si T+B pe aceeasi codita unde merg in acelasi ritm, ca in "
             "imnare si in `pagina3` (`musicxml_strans(acorduri=True)`). Paginile fara "
             "nicio masura de acord se sar — ar fi identice cu cele din stilul `voci`",
    )
    parser.add_argument(
        "--doar-repetitii", action="store_true",
        help="doar paginile cu cel putin un semn de repetitie. Pana pe 2026-10-04 "
             "generatorul le arunca; restul paginilor ar iesi identice cu loturile vechi",
    )
    parser.add_argument(
        "--decupaj-curatat", action="store_true",
        help="decupajele din pagina curatata de text, ca in `data/bach` si `data/bach2`. "
             "Implicit se iau din pagina doar indreptata: curatarea stergea note intregi "
             "si pauze pe care tinta le avea (§15.60)",
    )
    parser.add_argument(
        "--solo", action="store_true",
        help="un portativ de solo deasupra corului (`cu_solo`): sisteme de trei portative, ca "
             "pe cantarile cu solo si pian. Linia de solo vine din alta piesa cu aceeasi armura "
             "si metrica, aleasa din aceeasi parte (antrenare / validare), ca sa nu se scurga",
    )
    parser.add_argument(
        "--stil-randare", type=Path, default=STIL, metavar="MSS",
        help="stilul MuseScore cu care se randeaza (implicit `scripts/bach-stil.mss`). "
             "`scripts/bach-stil-mana.mss` deseneaza cu fontul Petaluma, cu aspect de mana",
    )
    args = parser.parse_args()

    from music21 import converter

    from app.evaluation.reference import UnsupportedNotation
    from app.recognition.kern_writer import UnwritableScore
    from app.ui.musescore import find

    musescore = find()
    if musescore is None:
        raise SystemExit("MuseScore nu e instalat — randarea il cere")

    dpis = [int(value) for value in args.dpi.split(",") if value.strip()]
    # Validarea se masoara la o singura rezolutie, ca cifra sa ramana comparabila
    # de la un lot la altul: 200, cand e printre ele.
    dpi_validare = 200 if 200 in dpis else dpis[0]

    def baza(xml: Path, dpi: int) -> str:
        """Numele paginii randate; cu o singura rezolutie, ca in primul lot."""
        return xml.stem if len(dpis) == 1 else f"{xml.stem}@{dpi}"

    out = args.out
    for folder in ("xml", "png", "antrenare"):
        (out / folder).mkdir(parents=True, exist_ok=True)

    if args.lista is not None:
        from app.training.satb import cor_satb

        intrari = json.loads(args.lista.read_text(encoding="utf-8"))
        surse = [(intrare["id"], Path(intrare["cale"])) for intrare in intrari]
        converteste = cor_satb
    else:
        surse = [(fisier.stem, fisier) for fisier in corale()]
        converteste = coral_satb
    surse = surse[: args.limita]
    fisiere = [cale for _, cale in surse]
    validare = tinute_deoparte([nume for nume, _ in surse], args.procent_validare)
    raport: dict[str, list] = {
        "refuzate": [], "curg_pe_doua_pagini": [], "decupaj_gresit": [], "fara_acorduri": [],
        "fara_repetitii": [],
    }
    acorduri = args.stil == "acorduri"

    # 1. Corale -> pagini MusicXML.
    convertite: list[tuple[str, Score | None]] = []
    for nume, fisier in surse:
        try:
            convertite.append((nume, converteste(converter.parse(str(fisier)))))
        except (UnsupportedNotation, UnwritableScore, ValueError) as problema:
            raport["refuzate"].append([nume, str(problema)])
    donori: dict[tuple, list[tuple[str, Score]]] = {}
    if args.solo:
        # Donorii, pe (parte, armura, metrica): o piesa de validare nu da solo pieselor de
        # antrenare si invers, ca melodia ei sa nu ajunga in antrenare pe alt drum.
        for nume, score in convertite:
            donori.setdefault(cheie_solo(nume in validare, score), []).append((nume, score))
    pagini_scrise: list[tuple[str, Path]] = []
    for nume, score in convertite:
        rng = random.Random(nume)
        try:
            if args.solo:
                candidati = [d for d in donori[cheie_solo(nume in validare, score)] if d[0] != nume]
                if not candidati:
                    raport.setdefault("fara_donor", []).append(nume)
                    continue
                score = cu_solo(score, rng.choice(candidati)[1], rng)
            toate = pagini(
                score, rng, dupa_densitate=not args.impartire_veche,
                sisteme_pe_pagina=SISTEME_PE_PAGINA_CU_SOLO if args.solo else SISTEME_PE_PAGINA,
            )
        except (UnsupportedNotation, UnwritableScore, ValueError) as problema:
            raport["refuzate"].append([nume, str(problema)])
            continue
        for k, (pagina, limite) in enumerate(toate, 1):
            xml = out / "xml" / f"{nume}-p{k}.musicxml"
            if acorduri and not masuri_cu_acorduri(pagina):
                raport["fara_acorduri"].append(xml.stem)
                continue
            if args.doar_repetitii and not any(repetitii(pagina)):
                raport["fara_repetitii"].append(xml.stem)
                continue
            if not xml.exists():
                xml.write_text(
                    musicxml_strans(pagina, limite, k == len(toate), acorduri), encoding="utf-8"
                )
            pagini_scrise.append((nume, xml))
    print(
        f"{len(fisiere)} corale: {len(fisiere) - len(raport['refuzate'])} convertite, "
        f"{len(raport['refuzate'])} refuzate, {len(pagini_scrise)} pagini"
    )

    # 2. Randarea, doar ce lipseste.
    for dpi in dpis:
        lucrari = [
            (xml, out / "png" / f"{baza(xml, dpi)}.png")
            for _, xml in pagini_scrise
            if not (out / "png" / f"{baza(xml, dpi)}-1.png").exists()
        ]
        if lucrari:
            print(f"se randeaza {len(lucrari)} pagini la {dpi} dpi...")
            randeaza(musescore, lucrari, dpi, None if args.impartire_veche else args.stil_randare)

    # 3. Decupaje + tinte, pentru ce nu s-a facut deja.
    facute_cale = out / "facute.txt"
    facute = set(facute_cale.read_text(encoding="utf-8").split()) if facute_cale.exists() else set()

    from app.evaluation.reference import load_with_layout
    from scripts.dataset import system_bounds, targets_for

    engine = None
    set_validare = []
    perechi_noi = 0
    lucru = [(nume, xml, dpi) for nume, xml in pagini_scrise for dpi in dpis]
    for nume, xml, dpi in lucru:
        pagina = baza(xml, dpi)
        png = out / "png" / f"{pagina}-1.png"
        if (out / "png" / f"{pagina}-2.png").exists():
            raport["curg_pe_doua_pagini"].append(pagina)
            continue
        if not png.exists():
            raport["refuzate"].append([pagina, "MuseScore n-a scris pagina"])
            continue

        if nume in validare:
            if dpi != dpi_validare:
                continue
            intrare = {
                "imagine": png.as_posix(),
                "referinta": xml.as_posix(),
                "scriitor": nume,
                "portative": 3 if args.solo else 2,
            }
            if args.solo:
                # Fara asezare, `masoara.py` ar imperechea vocile dupa pozitie si o transcriere
                # perfecta ar iesi 92/160 (gasit de code review, 2026-10-04).
                intrare["asezare"] = CU_SOLO
            set_validare.append(intrare)
            continue
        if pagina in facute:
            continue

        # Referinta se **reciteste** de pe disc, nu se ia din memorie: asa tinta
        # trece prin exact drumul unei referinte scrise in MuseScore, iar o
        # greseala a scriitorului MusicXML ar iesi aici, nu in antrenare.
        # Doua voci pe portativ, mereu: o pagina scrisa numai cu acorduri n-ar
        # declara nicio voce, si s-ar citi ca doua voci in loc de patru.
        # Cu solo: o singura voce pe portativul lui, altfel ar primi o voce de pauze in plus.
        score, starts, _ = load_with_layout(xml, min_voices=[1, 2, 2] if args.solo else 2)
        limite = system_bounds(starts, score.measure_count)

        if engine is None:
            from app.recognition.grandstaff_engine import GrandstaffEngine

            engine = GrandstaffEngine(staves_per_system=3 if args.solo else 2, clean=True)
            engine.load()
        crops = engine.detect_systems(png)
        asteptate = 3 if args.solo else 2
        if len(crops) != len(limite) or any(portative != asteptate for _, _, portative in crops):
            # Si portativele din fiecare decupaj: un portativ scapat de detector da [2, 2, 2, 1]
            # pe patru sisteme — numarul de decupaje se potriveste, dar 2-4 amesteca sisteme.
            raport["decupaj_gresit"].append(
                [pagina, len(crops), len(limite), [portative for _, _, portative in crops]]
            )
            continue
        if not args.decupaj_curatat:
            crops = fara_stergere(png, engine.cleaned.angle, crops)

        import cv2

        try:
            tinte = targets_for(score, limite)
        except UnwritableScore as problema:
            raport["refuzate"].append([pagina, str(problema)])
            continue
        randuri = []
        for index, ((_, imagine, portative), (text, tokeni)) in enumerate(zip(crops, tinte), 1):
            decupaj = f"{pagina}_sistem{index:02d}"
            cv2.imwrite(str(out / "antrenare" / f"{decupaj}.png"), imagine)
            (out / "antrenare" / f"{decupaj}.krn").write_text(text, encoding="utf-8")
            randuri.append({
                "image": f"{decupaj}.png", "transcription": tokeni,
                "staves": portative, "writer": nume,
            })
        with (out / "antrenare" / "date.jsonl").open("a", encoding="utf-8") as handle:
            for rand in randuri:
                handle.write(json.dumps(rand, ensure_ascii=False) + "\n")
        with facute_cale.open("a", encoding="utf-8") as handle:
            handle.write(pagina + "\n")
        perechi_noi += len(randuri)

    (out / "set-validare.json").write_text(
        json.dumps(set_validare, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "raport.json").write_text(
        json.dumps(raport, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"\nperechi noi: {perechi_noi} | validare: {len(set_validare)} pagini din "
        f"{len(validare)} corale | pe doua pagini: {len(raport['curg_pe_doua_pagini'])} | "
        f"decupaj gresit: {len(raport['decupaj_gresit'])} | refuzate: {len(raport['refuzate'])}"
        + (f" | fara acorduri, sarite: {len(raport['fara_acorduri'])}" if acorduri else "")
        + (f" | fara repetitii, sarite: {len(raport['fara_repetitii'])}" if args.doar_repetitii else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
