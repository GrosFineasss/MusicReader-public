"""Glife decupate -> `MungSymbolRepository`, cate un stil de persoana.

Ultima piesa dintre foaia scrisa de om si paginile sintetice (PROJECT.md §15.14,
§15.15). Smashcima alege glife dupa **eticheta si stil**, iar stilul e documentat
in `MungGlyphMetadata` ca „writer number, book ID, etc." — deci fiecare om care
completeaza o foaie devine un stil, si fiecare stil e o mana in plus in datele de
antrenare.

Constructia urmeaza `BaseSymbolExtractor.emit_glyph_from_mung_node` din
Smashcima: masca devine singurul sprite al glifei, interpretata alb-negru pe
fond transparent. Nu inventam un format — il producem pe cel pe care biblioteca
il citeste deja.

**Importurile de Smashcima sunt lenese.** `.venv`-ul proiectului nu-l are
instalat (dependinta `stringzilla` cere compilator C++ pe Python 3.14, §15.14),
si nici nu trebuie: nucleul, testele si conducta de transcriere nu depind de el.
"""

from __future__ import annotations

from pathlib import Path

from app.training.glife import GlyphCrop

# Rezolutia la care s-au scanat foile. `foaie_glife.py` deseneaza la 300 dpi si
# cere tiparire la marime reala, tocmai ca marimea glifei sa insemne ceva.
SHEET_DPI = 300


def build_glyph(crop: GlyphCrop, dpi: int = SHEET_DPI, node_id: int = 0):
    """O glifa decupata -> un `Glyph` cu metadatele de stil.

    `bitmap_origin` e centrul mastii: pentru glifele MUSCIMA++ el vine dintr-un
    nor de puncte adnotat manual, pe care noi nu-l avem. Centrul e alegerea
    neutra, si e vizibila aici ca sa nu para o masuratoare.
    """
    from smashcima.assets.glyphs.mung.extraction.mung_mask_to_smashcima_sprite_bitmap import (
        mung_mask_to_smashcima_sprite_bitmap,
    )
    from smashcima.assets.glyphs.mung.MungGlyphMetadata import MungGlyphMetadata
    from smashcima.geometry import Point
    from smashcima.scene import AffineSpace, Glyph, Sprite

    height, width = crop.mask.shape
    space = AffineSpace()
    sprite = Sprite(
        space=space,
        bitmap=mung_mask_to_smashcima_sprite_bitmap(crop.mask),
        bitmap_origin=Point(0.5, 0.5),
        dpi=dpi,
    )
    glyph = Glyph(
        space=space,
        region=Glyph.build_region_from_sprites_alpha_channel(
            label=crop.label, sprites=[sprite]
        ),
        sprites=[sprite],
    )
    MungGlyphMetadata(
        glyph=glyph,
        mung_style=crop.writer,
        mung_document=f"foaie-{crop.writer}",
        mung_node_id=node_id,
    )
    return glyph


# Ce deseneaza Smashcima ca **linie**, nu ca simbol asezat: se intinde intre
# doua capete, deci glifa trebuie sa stie unde ii sunt capetele. Lista e cea din
# `MuscimaPPLineSynthesizer.LABEL_MAP`, nu una ghicita de noi.
LINE_LABELS = frozenset({"legerLine", "stem", "beam", "beamHook"})


def line_endpoints(mask):
    """Cele doua capete ale unei trasaturi alungite.

    Smashcima face asta prin **scheletizare** (`cv2.ximgproc.thinning`), care e
    in `opencv-contrib`, nu in pachetul de baza pe care il avem. Nici n-ar fi
    potrivita: scheletizarea rezolva adnotari incalcite din MUSCIMA++, pe cand
    la noi fiecare glifa e o singura trasatura, scrisa singura intr-o casuta.

    Deci: axa principala a norului de puncte (PCA), si cele doua extreme pe ea.
    Pentru o bara de grup, o linie suplimentara sau o codita — adica exact ce
    intra aici — capetele sunt tocmai punctele cele mai departate de-a lungul
    trasaturii.
    """
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        raise ValueError("masca nu are destula cerneala pentru o linie")

    points = np.stack([xs, ys], axis=1).astype(np.float64)
    centred = points - points.mean(axis=0)
    # Vectorul propriu cu valoarea cea mai mare = directia in care se intinde.
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    axis = vectors[0]
    projections = centred @ axis
    return points[int(projections.argmin())], points[int(projections.argmax())]


def build_line_glyph(crop: GlyphCrop, dpi: int = SHEET_DPI, node_id: int = 0):
    """O glifa decupata -> `LineGlyph`, cu capetele detectate.

    Originea sprite-ului e la **mijloc**, ca la ei (`emit_line_glyph_from_mung_node`
    pune `bitmap_origin` in centru), iar capetele se dau in spatiul glifei.
    """
    from smashcima.assets.glyphs.mung.extraction.mung_mask_to_smashcima_sprite_bitmap import (
        mung_mask_to_smashcima_sprite_bitmap,
    )
    from smashcima.assets.glyphs.mung.MungGlyphMetadata import MungGlyphMetadata
    from smashcima.geometry import Point
    from smashcima.scene import AffineSpace, Glyph, LineGlyph, ScenePoint, Sprite

    start, end = line_endpoints(crop.mask)
    space = AffineSpace()
    sprite = Sprite(
        space=space,
        bitmap=mung_mask_to_smashcima_sprite_bitmap(crop.mask),
        bitmap_origin=Point(0.5, 0.5),
        dpi=dpi,
    )
    to_space = sprite.get_pixels_to_origin_space_transform()
    line = LineGlyph(
        space=space,
        region=Glyph.build_region_from_sprites_alpha_channel(
            label=crop.label, sprites=[sprite]
        ),
        sprites=[sprite],
        start_point=ScenePoint(
            point=to_space.apply_to(Point(float(start[0]), float(start[1]))), space=space
        ),
        end_point=ScenePoint(
            point=to_space.apply_to(Point(float(end[0]), float(end[1]))), space=space
        ),
    )
    MungGlyphMetadata(
        glyph=line,
        mung_style=crop.writer,
        mung_document=f"foaie-{crop.writer}",
        mung_node_id=node_id,
    )
    return line


def build_repository(crops: list[GlyphCrop], dpi: int = SHEET_DPI):
    """Toate glifele -> un repository gata de dat lui Smashcima.

    Glifele se impart in doua dupa eticheta: cele **de linie** (`LINE_LABELS`)
    merg in `line_glyphs_index` cu capetele detectate, restul in `glyphs_index`.
    Impartirea nu e a noastra — e cea din `MuscimaPPLineSynthesizer`, fiindca ea
    hotaraste ce cere Smashcima de la fiecare.

    O glifa de linie care n-are destula cerneala ca sa i se gaseasca doua capete
    se **raporteaza si se sare**, nu se pune cu capete inventate: o bara de grup
    cu capete gresite ar intinde muzica aiurea pe toata pagina.
    """
    from smashcima.assets.glyphs.mung.repository.GlyphsIndex import GlyphsIndex
    from smashcima.assets.glyphs.mung.repository.LineGlyphsIndex import LineGlyphsIndex
    from smashcima.assets.glyphs.mung.repository.MungSymbolRepository import (
        MungSymbolRepository,
    )

    glyphs = []
    lines = []
    skipped = []
    for index, crop in enumerate(crops):
        if crop.label in LINE_LABELS:
            try:
                lines.append(build_line_glyph(crop, dpi, node_id=index))
            except ValueError as problem:
                skipped.append(f"{crop.writer}/{crop.label}#{crop.index}: {problem}")
        else:
            glyphs.append(build_glyph(crop, dpi, node_id=index))

    if skipped:
        print(f"  {len(skipped)} glife de linie sarite:")
        for problem in skipped[:5]:
            print(f"    {problem}")

    return MungSymbolRepository(
        glyphs_index=GlyphsIndex.build(glyphs),
        line_glyphs_index=LineGlyphsIndex.build(lines),
    )


def save(repository, path: str | Path) -> Path:
    """Scrie repository-ul pe disc, ca sa nu se reconstruiasca la fiecare rulare.

    Formatul e pickle, fiindca asa il citeste Smashcima — `MungSymbolRepository`
    isi spune in docstring ca exista tocmai ca sa fie pus pe disc asa.
    """
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".nou")
    with temporary.open("wb") as handle:
        pickle.dump(repository, handle)
    temporary.replace(path)
    return path
