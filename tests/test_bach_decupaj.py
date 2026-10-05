"""`scripts/bach.py`: decupajele de antrenare se iau din pagina nestearsa (§15.60)."""

import cv2
import numpy as np
import pytest
from PIL import Image

from app.preprocessing import clean_page

# Un semn departe sub portativul de jos (8,5 spatii): fara codita si fara vecini,
# curatarea il ia drept text — cum facea cu pauza vocii a doua din PDMX.
SEMN = (400, 600, 414, 625)
TOT_SISTEMUL = (40, 130, 860, 450)


def _pagina(cale):
    pagina = np.full((700, 900), 255, np.uint8)
    for sus in (150, 350):
        for k in range(5):
            cv2.line(pagina, (50, sus + 20 * k), (850, sus + 20 * k), 0, 2)
        for x in range(120, 800, 90):
            cv2.ellipse(pagina, (x, sus + 50), (11, 8), -20, 0, 360, 0, -1)
            cv2.line(pagina, (x + 10, sus + 50), (x + 10, sus - 10), 0, 2)
    x1, y1, x2, y2 = SEMN
    cv2.rectangle(pagina, (x1, y1), (x2, y2), 0, -1)
    cv2.imwrite(str(cale), pagina)


def _curatata(cale):
    return clean_page(np.array(Image.open(cale).convert("L")))


def _decupaj(imagine, chenar, portative=2):
    x1, y1, x2, y2 = chenar
    return (chenar, imagine[y1:y2, x1:x2], portative)


def test_pastreaza_ce_sterge_curatarea(tmp_path):
    from scripts.bach import fara_stergere

    cale = tmp_path / "pagina.png"
    _pagina(cale)
    curatata = _curatata(cale)
    chenar = (SEMN[0] - 20, SEMN[1] - 20, SEMN[2] + 20, SEMN[3] + 20)
    decupaj = _decupaj(curatata.image, chenar)
    assert not (decupaj[1] < 128).any(), "premisa: curatarea sterge semnul"

    ((acelasi, brut, portative),) = fara_stergere(cale, curatata.angle, [decupaj])

    assert acelasi == chenar and portative == 2
    assert (brut < 128).sum() >= (SEMN[2] - SEMN[0]) * (SEMN[3] - SEMN[1])


def test_restul_sistemului_e_neschimbat(tmp_path):
    from scripts.bach import fara_stergere

    cale = tmp_path / "pagina.png"
    _pagina(cale)
    curatata = _curatata(cale)
    decupaj = _decupaj(curatata.image, TOT_SISTEMUL)

    ((_, brut, _),) = fara_stergere(cale, curatata.angle, [decupaj])

    assert np.array_equal(brut, decupaj[1])


def test_drumuri_despartite_se_opresc(tmp_path):
    from scripts.bach import fara_stergere

    cale = tmp_path / "pagina.png"
    _pagina(cale)
    curatata = _curatata(cale)
    decupaj = _decupaj(curatata.image, TOT_SISTEMUL)

    with pytest.raises(ValueError, match="nu contine cerneala"):
        fara_stergere(cale, curatata.angle + 3.0, [decupaj])
