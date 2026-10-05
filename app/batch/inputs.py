"""Ce intra in lot: poze si PDF-uri scanate.

Partiturile tiparite vin aproape intotdeauna ca PDF scanat, nu ca fotografie.
Un PDF de 12 pagini e **12 unitati de lucru**, nu una: daca se intrerupe lotul
la pagina 9, vrem sa reluam de la 10, nu de la 1.

Randarea PDF-urilor foloseste `pypdfium2` (Apache-2.0), care e un singur wheel
fara dependente externe — spre deosebire de `pdf2image`, care cere poppler
instalat separat pe sistem.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.recognition.engine import Kind, PageImage

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED = IMAGE_SUFFIXES | PDF_SUFFIXES

# 200 dpi: destul ca portativele sa fie citibile, fara sa umflem inutil
# imaginile — oricum le reducem la 128 px inaltime per portativ.
PDF_DPI = 200


class PdfSupportMissing(RuntimeError):
    pass


def page_count(path: Path) -> int:
    """Cate pagini are fisierul. O imagine e o pagina."""
    if path.suffix.lower() not in PDF_SUFFIXES:
        return 1
    return _pdf_page_count(path)


def _pdf_page_count(path: Path) -> int:
    try:
        import pypdfium2
    except ImportError as error:  # pragma: no cover - depinde de mediu
        raise PdfSupportMissing(
            "citirea PDF-urilor cere `pypdfium2`: uv pip install pypdfium2"
        ) from error

    document = pypdfium2.PdfDocument(str(path))
    try:
        return len(document)
    finally:
        document.close()


def render_pdf_page(path: Path, page: int, out_dir: Path, dpi: int = PDF_DPI) -> Path:
    """Randeaza o pagina de PDF ca PNG si intoarce calea.

    Randarea se face o singura data si se pastreaza pe disc: e scumpa, si o
    vrem oricum salvata ca sa putem privi ce a vazut modelul.

    Numele include o amprenta a caii intregi. Fara ea, `arhiva/a/scan.pdf` si
    `arhiva/b/scan.pdf` ar scrie amandoua in `scan_p1.png`, iar al doilea PDF ar
    fi transcris din paginile primului — fara nicio eroare, doar cu rezultatul
    gresit.
    """
    try:
        import pypdfium2
    except ImportError as error:  # pragma: no cover - depinde de mediu
        raise PdfSupportMissing(
            "citirea PDF-urilor cere `pypdfium2`: uv pip install pypdfium2"
        ) from error

    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:6]
    target = out_dir / f"{path.stem}_p{page + 1}_{key}.png"
    if target.exists():
        return target

    document = pypdfium2.PdfDocument(str(path))
    try:
        rendered = document[page].render(scale=dpi / 72)
        rendered.to_pil().save(target)
    finally:
        document.close()
    return target


def collect(roots: list[Path], kind: Kind) -> list[PageImage]:
    """Enumereaza recursiv fisierele acceptate, in ordine stabila.

    Ordinea stabila nu e cosmetica: un lot reluat trebuie sa parcurga aceleasi
    unitati in aceeasi ordine, ca raportul sa fie comparabil intre rulari.
    """
    pages: list[PageImage] = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED
            )
        for path in candidates:
            if path.suffix.lower() not in SUPPORTED:
                continue
            for page in range(page_count(path)):
                pages.append(PageImage(path=path, kind=kind, page=page))
    return pages


def resolve(page: PageImage, work_dir: Path) -> Path:
    """Calea imaginii de dat modelului. PDF-urile se randeaza intai."""
    if page.path.suffix.lower() in PDF_SUFFIXES:
        return render_pdf_page(page.path, page.page, work_dir / "pdf_pages")
    return page.path
