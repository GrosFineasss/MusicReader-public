"""Foaie de glife completata -> glife decupate, cu eticheta si stil.

Etapa 11, scriitor-independenta (PROJECT.md §15.12, §15.15). `foaie_glife.py` a
desenat grila si i-a scris coordonatele intr-un JSON; aici se face drumul invers,
pe foaia scanata.

**Nu se ghiceste nimic despre unde sunt casutele** — noi le-am desenat. Ce
trebuie aflat din scan e doar cum s-a asezat foaia in scaner: rotita, mutata,
marita. Asta se citeste din cele patru repere din colturi, iar restul e o
transformare aplicata coordonatelor pe care le stim deja.

E aceeasi hotarare ca la `--portative-pe-sistem` si la eticheta de scriitor:
informatia care se stie la producere se scrie, si nu se mai deduce dupa.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Cat se strange decupajul fata de caseta desenata, in pixeli la scara hartii
# (casuta are 250). Chenarul nu trebuie sa ramana in decupaj: fiecare glifa ar
# capata un dreptunghi in jur, pe care Smashcima l-ar desena apoi pe pagina ca
# si cum ar fi cerneala.
#
# **40, nu 10.** Zece ajungea pe scanuri drepte, dar nu pe fotografii: hartia nu
# e perfect plana, iar transformarea din patru repere lasa o eroare de ~1% din
# latimea paginii — destul cat chenarul sa cada in decupaj. Masurat pe prima
# foaie reala (§15.22), pe punctul de prelungire, care e cel mai mic simbol si
# deci cel mai sensibil:
#
# | inset | marimea punctului decupat |
# |---|---|
# | 10 | 223 px — adica toata casuta, chenar cu tot |
# | 20 | 204 px |
# | **30** | **19 px — punctul adevarat** |
# | 40 | 19 px, si capul de nota se strange si el la marimea lui |
#
# 40 pe o casuta de 250 inseamna 16% margine; instructiunea de pe foaie cere
# oricum sa nu se atinga marginile, deci nu se pierde scris.
BORDER_INSET = 40

# Sub atata cerneala, caseta se considera **goala** — omul a sarit-o, cum ii
# spune instructiunea de pe foaie. O caseta goala returnata ca glifa ar baga in
# repository un simbol inexistent, care ar fi apoi desenat pe pagini sintetice.
MIN_INK_PIXELS = 40


@dataclass(frozen=True)
class GlyphCrop:
    """O glifa decupata: masca ei, eticheta si al cui e scrisul."""

    label: str
    index: int
    writer: str
    mask: object  # np.ndarray uint8, 0/1 — forma ceruta de Smashcima

    @property
    def ink(self) -> int:
        return int(self.mask.sum())


def la_scara_hartii(grey, plan_page: dict):
    """Aduce poza la rezolutia hartii, daca e mai mare.

    Doua motive, amandoua masurate pe prima foaie reala (§15.22):

    * **Marimea glifelor.** Harta e desenata la 300 dpi si o casuta are 250 px.
      Un telefon da 8000 px pe latura lunga, deci aceeasi casuta iese de ~660 px,
      iar glifele intra in repository de doua ori si jumatate mai mari. Doua
      persoane cu telefoane diferite ar da acelasi simbol la scari diferite, iar
      Smashcima le-ar desena diferit fara sa se stie de ce.
    * **Viteza.** Pragul local isi ia fereastra ca fractiune din imagine; pe 80
      de megapixeli fereastra trece de 1000 px si binarizarea nu se mai termina.

    Nu se mareste niciodata o poza mai mica: dintr-o foaie fotografiata prost
    n-avem de unde scoate detaliu care nu exista.
    """
    import cv2

    latura_hartii = max(plan_page["size"])
    latura = max(grey.shape)
    if latura <= latura_hartii:
        return grey
    factor = latura_hartii / latura
    return cv2.resize(
        grey, (round(grey.shape[1] * factor), round(grey.shape[0] * factor)),
        interpolation=cv2.INTER_AREA,
    )


def binarize_markers(grey):
    """Alb-negru pentru **reperele din colturi**, cu prag local si fereastra mare.

    Reperele sunt patrate pline de ~60 px. Fereastra trebuie sa fie **mult mai
    mare decat ele**: una mai mica le-ar vedea interiorul uniform si l-ar
    declara fond, deci patratul plin ar deveni un contur gol si detectia l-ar
    pierde. Cu fereastra mare, interiorul ramane mult sub media din jur.

    De ce nu Otsu, pe care il foloseste `deskew.py` pentru fotocopii: el alege
    **un singur** prag pentru toata imaginea, deci o umbra puternica ii scoate
    reperele din partea luminata. Masurat pe o foaie cu gradient de lumina:

    | tarie umbra | Otsu | prag local, fereastra mare |
    |---|---|---|
    | 0,55 | gaseste 4 | gaseste 4 |
    | 0,70 | **gaseste 2** | gaseste 4 |
    | 0,90 | **gaseste 2** | gaseste 4 |

    Adica pe o poza de telefon cu umbra, Otsu pierde jumatate din repere si
    segmentarea se opreste, pe cand pragul local tine.
    """
    import cv2

    fereastra = max(101, (min(grey.shape) // 6) | 1)  # impar, mult peste reper
    return cv2.adaptiveThreshold(
        grey, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, fereastra, 10
    ).astype("uint8")


def binarize_cells(grey):
    """Alb-negru pentru **continutul casutelor**, cu prag local.

    Aici situatia e inversa fata de repere. Un prag global merge pe un scan,
    unde hartia e uniform alba; pe o poza de telefon nu, fiindca o umbra peste
    pagina face jumatate din foaie mai intunecata decat cerneala din cealalta
    jumatate. Partea umbrita ar iesi toata cerneala, partea luminata si-ar
    pierde glifele — si nimic n-ar semnala-o, fiindca segmentarea ar gasi tot
    casutele, doar cu continut gresit.

    Fereastra se alege mare fata de o trasatura de pix, dar mica fata de o
    umbra. Doua binarizari pentru doua treburi diferite, fiecare cu motivul ei.
    """
    import cv2

    fereastra = max(31, (min(grey.shape) // 20) | 1)  # impar
    return cv2.adaptiveThreshold(
        grey, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, fereastra, 10
    ).astype("uint8")


def load_plan(path: str | Path) -> dict:
    """Harta casutelor scrisa de `scripts/foaie_glife.py`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"nu exista harta casutelor: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_markers(binary) -> list[tuple[float, float]]:
    """Centrele celor patru repere din colturi, in ordinea din harta.

    Reperele sunt patrate plin negre, singurele componente mari si aproape
    patrate de pe foaie. Se cauta printre componente, nu la coordonate fixe:
    tocmai pozitia lor reala spune cum s-a asezat foaia in scaner.
    """
    import cv2
    import numpy as np

    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = []
    for label in range(1, count):
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if w < 20 or h < 20:
            continue
        if not 0.7 < w / h < 1.4:  # aproape patrat
            continue
        if area < 0.7 * w * h:  # aproape plin
            continue
        candidates.append((float(centroids[label][0]), float(centroids[label][1])))

    if len(candidates) < 4:
        raise ValueError(
            f"s-au gasit doar {len(candidates)} repere din 4 — foaia e taiata la "
            "scanare, sau reperele au iesit prea slabe"
        )

    height, width = binary.shape[:2]
    corners = [(0, 0), (width, 0), (0, height), (width, height)]
    chosen = []
    for cx, cy in corners:
        nearest = min(candidates, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        chosen.append(nearest)
    if len({(round(x), round(y)) for x, y in chosen}) != 4:
        raise ValueError(
            "acelasi reper a fost ales pentru doua colturi — foaia pare rotita "
            "mult sau taiata"
        )
    return chosen


def sheet_transform(plan_page: dict, binary):
    """Matricea care duce coordonatele din harta in coordonatele scanului."""
    import cv2
    import numpy as np

    expected = [
        ((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in plan_page["markers"]
    ]
    found = find_markers(binary)
    return cv2.getPerspectiveTransform(
        np.array(expected, dtype="float32"), np.array(found, dtype="float32")
    )


def _cell_mask(binary, box, transform, inset: int = BORDER_INSET):
    """Decupajul unei casute, cu chenarul lasat pe dinafara."""
    import cv2
    import numpy as np

    x1, y1, x2, y2 = box
    corners = np.array(
        [
            [x1 + inset, y1 + inset],
            [x2 - inset, y1 + inset],
            [x2 - inset, y2 - inset],
            [x1 + inset, y2 - inset],
        ],
        dtype="float32",
    ).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, transform).reshape(-1, 2)

    width = int(round(max(np.linalg.norm(mapped[1] - mapped[0]), 1)))
    height = int(round(max(np.linalg.norm(mapped[3] - mapped[0]), 1)))
    target = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype="float32"
    )
    unwarp = cv2.getPerspectiveTransform(mapped.astype("float32"), target)
    return cv2.warpPerspective(binary, unwarp, (width, height))


def _trim(mask):
    """Strange masca pe conturul cernelii. `None` daca nu e cerneala."""
    import numpy as np

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def segment(
    image_path: str | Path,
    plan_page: dict,
    writer: str,
    min_ink: int = MIN_INK_PIXELS,
    inset: int = BORDER_INSET,
) -> tuple[list[GlyphCrop], list[str]]:
    """Foaia scanata -> (glife decupate, casute sarite).

    Casutele goale se **raporteaza**, nu se umplu cu nimic: instructiunea de pe
    foaie ii spune omului sa sara caseta daca greseste, deci golul e un raspuns
    valid, iar o glifa inventata din el ar ajunge desenata pe pagini sintetice.
    """
    import cv2
    import numpy as np
    from PIL import Image

    grey = la_scara_hartii(np.array(Image.open(image_path).convert("L")), plan_page)
    transform = sheet_transform(plan_page, binarize_markers(grey))
    binary = binarize_cells(grey)

    crops: list[GlyphCrop] = []
    skipped: list[str] = []
    for cell in plan_page["cells"]:
        mask = _cell_mask(binary, cell["box"], transform, inset)
        trimmed = _trim(mask)
        if trimmed is None or int(trimmed.sum()) < min_ink:
            skipped.append(f"{cell['label']}#{cell['index']}")
            continue
        crops.append(
            GlyphCrop(
                label=cell["label"],
                index=int(cell["index"]),
                writer=writer,
                mask=trimmed.astype(np.uint8),
            )
        )
    return crops, skipped


def coverage(crops: list[GlyphCrop], symbols: list[str]) -> dict[str, int]:
    """Cate exemplare s-au obtinut din fiecare simbol cerut.

    Se raporteaza inclusiv zerourile: un simbol fara niciun exemplar inseamna ca
    stilul acela nu-l poate desena, si `RepositoryGlyphSynthesizer` va cadea pe
    sintetizatorul de rezerva — bine de stiut inainte, nu dupa.
    """
    counts = {label: 0 for label in symbols}
    for crop in crops:
        counts[crop.label] = counts.get(crop.label, 0) + 1
    return counts
