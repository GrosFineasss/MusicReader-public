"""Verifica fiecare pagina generata de `scripts/bach.py`, si o arata.

    python scripts/verifica_bach.py --date data/bach

Dus-intorsul (`tests/test_bach.py`) garanteaza ca **tinta** e muzica din fisier.
Nu garanteaza ca **decupajul** contine toata muzica aceea. Doua feluri in care
imaginea poate pierde note pe care tinta le are, si pe care nu le vede nimeni:

1. **chenarul taie.** O nota cu linii suplimentare iese din sistem; daca
   chenarul detectorului trece prin ea, jumatate ramane afara. Se vede ca
   cerneala pe **marginea** chenarului: o codita sau o linie suplimentara
   taiata lasa pixeli negri exact pe randul de sus sau de jos.
2. **curatarea sterge.** `suppress_text` sterge orice componenta care nu atinge
   banda portativului largita cu un spatiu (§14.7). Pe pagini fara versuri,
   singurul lucru legitim de sters sunt numerele de masura; orice altceva
   sters dintr-un sistem e o nota pe care tinta o are si imaginea nu.

Pentru fiecare pagina se scrie o imagine cu chenarele (rosu) si ce a sters
curatarea (albastru), iar paginile se strang cate sase pe o plansa, ca sa poata
fi privite toate. `verificare.json` tine cifrele si ce e suspect.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# Cate pagini pe plansa, si cat de lata e fiecare pe ea.
PE_PLANSA = (3, 2)
LATIME_PAGINA = 560


def sterse_suspecte(straight, cleaned, cutii, bands) -> list[tuple[int, int, int, int]]:
    """Componentele sterse de curatare care nu sunt numere de masura sau arce.

    Doua lucruri se sterg pe drept: **numerele de masura** — mici, in capul
    chenarului, deasupra portativului de sus — si **legaturile**, arce late si
    joase care nu intra in tinta (vocabularul n-are legato, §15.19). Orice
    altceva sters din interiorul unui chenar se raporteaza.
    """
    import cv2
    import numpy as np

    sters = cv2.bitwise_and(straight, cv2.bitwise_not(cleaned))
    count, _, stats, _ = cv2.connectedComponentsWithStats(sters, connectivity=8)
    spatiu = max((band.spacing for band in bands), default=10.0) or 10.0
    suspecte = []
    for label in range(1, count):
        x, y, w, h, arie = stats[label]
        if arie < 4:
            continue
        for x1, y1, x2, y2 in cutii:
            if not (x1 <= x + w / 2 <= x2 and y1 <= y + h / 2 <= y2):
                continue
            numar = h < 1.6 * spatiu and w < 1.2 * spatiu and y < y1 + 2.5 * spatiu
            arc = w >= 3 * spatiu and h <= 2 * spatiu
            if not (numar or arc):
                suspecte.append((int(x), int(y), int(w), int(h)))
            break
    return suspecte


def taieturi(cleaned_ink, cutii) -> list[int]:
    """Cati pixeli de cerneala stau pe randul de sus si de jos al fiecarui chenar."""
    rezultat = []
    for x1, y1, x2, y2 in cutii:
        sus = int(cleaned_ink[y1, x1:x2].sum())
        jos = int(cleaned_ink[min(y2 - 1, cleaned_ink.shape[0] - 1), x1:x2].sum())
        rezultat.append(sus + jos)
    return rezultat


def imagine_verificare(straight, cleaned, cutii, suspecte, nume: str):
    """Pagina, micsorata: cerneala pastrata neagra, cea stearsa albastra, chenare rosii."""
    import cv2
    import numpy as np

    h, w = straight.shape
    pagina = np.full((h, w, 3), 255, dtype=np.uint8)
    pagina[straight > 0] = (255, 160, 60)  # sters: albastru (BGR)
    pagina[cleaned > 0] = (0, 0, 0)
    for x1, y1, x2, y2 in cutii:
        cv2.rectangle(pagina, (x1, y1), (x2, y2), (0, 0, 255), 4)
    for x, y, cw, ch in suspecte:
        cv2.rectangle(pagina, (x - 12, y - 12), (x + cw + 12, y + ch + 12), (255, 0, 255), 6)
    scara = LATIME_PAGINA / w
    mica = cv2.resize(pagina, (LATIME_PAGINA, int(h * scara)), interpolation=cv2.INTER_AREA)
    cv2.putText(mica, nume, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 0), 2)
    return mica


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=Path, required=True, help="folderul lui `scripts/bach.py`")
    args = parser.parse_args()

    import cv2
    import numpy as np
    from PIL import Image

    from app.preprocessing import clean_page
    from app.preprocessing.deskew import binarize, deskew
    from app.recognition.grandstaff_engine import GrandstaffEngine

    pagini = sorted((args.date / "png").glob("*-1.png"))
    out = args.date / "verificare"
    (out / "planse").mkdir(parents=True, exist_ok=True)

    engine = GrandstaffEngine(staves_per_system=2, clean=True)
    engine.load()

    rezultate = []
    imagini = []
    for png in pagini:
        nume = png.stem[:-2]
        # `@150` din numele paginii e rezolutia (`bach.py --dpi`); referinta e una.
        xml = args.date / "xml" / f"{nume.split('@')[0]}.musicxml"
        crops = engine.detect_systems(png)
        cutii = [box for box, _, _ in crops]

        grey = np.array(Image.open(png).convert("L"))
        straight, _ = deskew(binarize(grey))
        curat = clean_page(grey)
        cleaned = cv2.bitwise_not(curat.image)  # cerneala alba pe negru, ca `straight`

        from app.evaluation.reference import load_with_layout

        _, starts, _ = load_with_layout(xml)
        suspecte = sterse_suspecte(straight, cleaned, cutii, curat.bands)
        pe_margine = taieturi(cleaned > 0, cutii)
        rezultate.append({
            "pagina": nume,
            "sisteme_scrise": len(starts),
            "sisteme_decupate": len(cutii),
            "pixeli_pe_marginea_chenarului": pe_margine,
            "sterse_suspecte": suspecte,
        })
        imagini.append(imagine_verificare(straight, cleaned, cutii, suspecte, nume))

    coloane, randuri = PE_PLANSA
    for start in range(0, len(imagini), coloane * randuri):
        grup = imagini[start : start + coloane * randuri]
        inaltime = max(im.shape[0] for im in grup)
        plansa = np.full((inaltime * randuri, LATIME_PAGINA * coloane, 3), 230, dtype=np.uint8)
        for i, im in enumerate(grup):
            r, c = divmod(i, coloane)
            plansa[r * inaltime : r * inaltime + im.shape[0],
                   c * LATIME_PAGINA : (c + 1) * LATIME_PAGINA] = im
        cv2.imwrite(str(out / "planse" / f"plansa-{start // (coloane * randuri) + 1:03d}.png"), plansa)

    nepotrivite = [r["pagina"] for r in rezultate if r["sisteme_scrise"] != r["sisteme_decupate"]]
    taiate = [r["pagina"] for r in rezultate if any(r["pixeli_pe_marginea_chenarului"])]
    sterse = [r["pagina"] for r in rezultate if r["sterse_suspecte"]]
    (out / "verificare.json").write_text(
        json.dumps({
            "pagini": len(rezultate),
            "decupaj_nepotrivit": nepotrivite,
            "chenar_taie_cerneala": taiate,
            "curatarea_sterge_din_sistem": sterse,
            "detalii": rezultate,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"{len(rezultate)} pagini | decupaj nepotrivit: {len(nepotrivite)} | "
        f"chenar care taie: {len(taiate)} | curatare suspecta: {len(sterse)}\n"
        f"planse in {out / 'planse'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
