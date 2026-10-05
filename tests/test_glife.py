"""Foaie de glife completata -> glife decupate (etapa 11, §15.15).

Testele isi construiesc singure foaia si o „scaneaza" — o rotesc, o mut si o
scaleaza — fiindca proba care conteaza nu e ca segmentarea merge pe o imagine
perfecta, ci ca **se descurca pe una asezata strambe in scaner**. Pe una
perfecta ar merge si o taiere la coordonate fixe, care e exact ce nu vrem.
"""

import numpy as np
import pytest

from app.training.glife import coverage, find_markers, segment

MARKER = 60
CELL = 200


def foaie(cells=(("gClef", 0), ("gClef", 1), ("fClef", 0)), umplute=None, size=(1200, 900)):
    """O foaie mica: patru repere in colturi si niste casute desenate."""
    from PIL import Image, ImageDraw

    width, height = size
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    margin = 40
    corners = [
        (margin, margin),
        (width - margin - MARKER, margin),
        (margin, height - margin - MARKER),
        (width - margin - MARKER, height - margin - MARKER),
    ]
    for x, y in corners:
        draw.rectangle([x, y, x + MARKER, y + MARKER], fill=0)

    umplute = {(label, index) for label, index in (umplute or cells)}
    boxes = []
    x0, y0 = 150, 200
    for position, (label, index) in enumerate(cells):
        bx = x0 + position * (CELL + 30)
        draw.rectangle([bx, y0, bx + CELL, y0 + CELL], outline=0, width=3)
        boxes.append({"label": label, "index": index, "box": [bx, y0, bx + CELL, y0 + CELL]})
        if (label, index) in umplute:
            # „scris de mana": o elipsa in mijlocul casutei
            draw.ellipse(
                [bx + 60, y0 + 70, bx + CELL - 60, y0 + CELL - 70], outline=0, width=8
            )

    plan = {
        "page": 1,
        "size": [width, height],
        "markers": [[x, y, x + MARKER, y + MARKER] for x, y in corners],
        "cells": boxes,
    }
    return image, plan


def scaneaza(image, unghi=0.0, scara=1.0, dx=0, dy=0):
    """Cum arata foaia dupa ce a trecut printr-un scaner cu mana omului pe ea."""
    import cv2

    a = np.array(image)
    h, w = a.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), unghi, scara)
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(a, m, (w, h), borderValue=255)


def scrie(tmp_path, array, name="foaie.png"):
    from PIL import Image

    path = tmp_path / name
    Image.fromarray(array).save(path)
    return path


class TestRepere:
    def test_cele_patru_colturi_se_gasesc(self):
        image, plan = foaie()
        binary = (np.array(image) < 128).astype(np.uint8)
        assert len(find_markers(binary)) == 4

    def test_fara_repere_se_spune_de_ce(self):
        binary = np.zeros((400, 400), dtype=np.uint8)
        with pytest.raises(ValueError, match="repere"):
            find_markers(binary)


class TestSegmentare:
    def test_pe_o_foaie_dreapta(self, tmp_path):
        image, plan = foaie()
        path = scrie(tmp_path, np.array(image))
        crops, sarite = segment(path, plan, writer="ana")
        assert len(crops) == 3 and not sarite
        assert {c.label for c in crops} == {"gClef", "fClef"}
        assert {c.writer for c in crops} == {"ana"}

    def test_pe_o_foaie_asezata_strambe(self, tmp_path):
        # Proba care conteaza: rotita, micsorata si mutata, ca un scan real.
        image, plan = foaie()
        path = scrie(tmp_path, scaneaza(image, unghi=2.0, scara=0.95, dx=20, dy=-15))
        crops, sarite = segment(path, plan, writer="ana")
        assert len(crops) == 3 and not sarite

    def test_casutele_goale_se_raporteaza_nu_se_inventeaza(self, tmp_path):
        # Instructiunea de pe foaie spune „daca gresesti, lasa caseta goala",
        # deci golul e un raspuns valid — si nu are ce cauta in repository.
        image, plan = foaie(umplute=[("gClef", 0)])
        path = scrie(tmp_path, np.array(image))
        crops, sarite = segment(path, plan, writer="ana")
        assert [c.label for c in crops] == ["gClef"]
        assert sorted(sarite) == ["fClef#0", "gClef#1"]

    def test_chenarul_nu_intra_in_decupaj(self, tmp_path):
        image, plan = foaie()
        path = scrie(tmp_path, np.array(image))
        crops, _ = segment(path, plan, writer="ana")
        for crop in crops:
            mask = crop.mask
            # Un chenar prins in decupaj ar face toate cele patru margini pline.
            margini = [mask[0].mean(), mask[-1].mean(), mask[:, 0].mean(), mask[:, -1].mean()]
            assert not all(m > 0.8 for m in margini)

    def test_decupajul_e_strans_pe_cerneala(self, tmp_path):
        image, plan = foaie()
        path = scrie(tmp_path, np.array(image))
        crops, _ = segment(path, plan, writer="ana")
        for crop in crops:
            # Mai mic decat caseta, fiindca s-a strans pe ce a scris omul.
            assert crop.mask.shape[0] < CELL and crop.mask.shape[1] < CELL
            assert crop.ink > 0


def fotografiaza(image, umbra=0.7, perspectiva=True):
    """Cum arata foaia intr-o poza de telefon: oblica si cu o umbra peste ea."""
    import cv2

    a = np.array(image).astype(np.float32)
    h, w = a.shape
    if perspectiva:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * .04, h * .02], [w * .97, 0], [w, h * .98], [w * .02, h]])
        a = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h),
                                borderValue=255)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gradient = 1.0 - umbra * ((xx / w) * 0.6 + (yy / h) * 0.4)
    return np.clip(a * gradient, 0, 255).astype(np.uint8)


class TestFotografii:
    """Foile vin ca poze de telefon, nu doar ca scanuri drepte."""

    def test_perspectiva_si_umbra_puternica(self, tmp_path):
        # Fondul ajunge sub 80 in coltul umbrit — mult sub pragul fix de 128 pe
        # care il foloseam la inceput, si sub ce mai poate separa Otsu global.
        image, plan = foaie()
        foto = fotografiaza(image, umbra=0.7)
        assert foto[-5, -5] < 100
        path = scrie(tmp_path, foto)
        crops, sarite = segment(path, plan, writer="ana")
        assert len(crops) == 3 and not sarite

    def test_umbra_foarte_puternica(self, tmp_path):
        image, plan = foaie()
        path = scrie(tmp_path, fotografiaza(image, umbra=0.85))
        crops, _ = segment(path, plan, writer="ana")
        assert len(crops) == 3

    def test_reperele_supravietuiesc_umbrei(self, tmp_path):
        """Punctul care cedeaza primul, si de asta e testat separat.

        Masurat: cu prag global (Otsu) se gaseau doar 2 repere din 4 la umbra
        de 0,70, iar segmentarea se oprea. Fereastra locala mare le tine.
        """
        from app.training.glife import binarize_markers, find_markers

        image, _ = foaie()
        assert len(find_markers(binarize_markers(fotografiaza(image, umbra=0.85)))) == 4


class TestAcoperire:
    def test_simbolurile_fara_exemplare_se_vad(self, tmp_path):
        image, plan = foaie(umplute=[("gClef", 0)])
        path = scrie(tmp_path, np.array(image))
        crops, _ = segment(path, plan, writer="ana")
        counts = coverage(crops, ["gClef", "fClef", "restQuarter"])
        # Zerourile se raporteaza: stilul acela nu poate desena simbolul, iar
        # Smashcima va cadea pe sintetizatorul de rezerva.
        assert counts == {"gClef": 1, "fClef": 0, "restQuarter": 0}
