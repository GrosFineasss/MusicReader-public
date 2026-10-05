"""Intrarile lotului: poze si PDF-uri scanate.

Partiturile tiparite vin ca PDF. Un PDF de N pagini trebuie sa fie N unitati de
lucru, altfel o intrerupere la pagina 9 din 12 ar cere reluarea de la 1.
"""

import pytest

from app.batch import inputs
from app.recognition.engine import Kind

pypdfium2 = pytest.importorskip("pypdfium2")


@pytest.fixture
def pdf(tmp_path):
    """Un PDF de trei pagini, construit pe loc."""
    document = pypdfium2.PdfDocument.new()
    for _ in range(3):
        document.new_page(300, 400)
    path = tmp_path / "scan.pdf"
    document.save(str(path))
    document.close()
    return path


@pytest.fixture
def folder(tmp_path, pdf):
    (tmp_path / "poza.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "notite.txt").write_text("nu e partitura")
    return tmp_path


class TestPageCount:
    def test_an_image_is_one_page(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(b"x")
        assert inputs.page_count(path) == 1

    def test_a_pdf_reports_its_real_length(self, pdf):
        assert inputs.page_count(pdf) == 3


class TestCollect:
    def test_a_pdf_becomes_one_unit_per_page(self, pdf):
        pages = inputs.collect([pdf], Kind.PRINTED)
        assert [p.page for p in pages] == [0, 1, 2]

    def test_images_and_pdfs_mix_in_one_batch(self, folder):
        pages = inputs.collect([folder], Kind.PRINTED)
        assert len(pages) == 4  # 1 poza + 3 pagini de PDF

    def test_unsupported_files_are_skipped(self, folder):
        names = {p.path.name for p in inputs.collect([folder], Kind.PRINTED)}
        assert "notite.txt" not in names

    def test_order_is_stable_between_runs(self, folder):
        first = [(p.path, p.page) for p in inputs.collect([folder], Kind.PRINTED)]
        second = [(p.path, p.page) for p in inputs.collect([folder], Kind.PRINTED)]
        assert first == second

    def test_kind_is_carried_onto_every_page(self, pdf):
        assert all(p.kind is Kind.PRINTED for p in inputs.collect([pdf], Kind.PRINTED))


class TestStem:
    def test_an_image_keeps_its_name_recognisable(self, tmp_path):
        from app.recognition.engine import PageImage

        assert PageImage(tmp_path / "cantec.png", Kind.HANDWRITTEN).stem.startswith("cantec")

    def test_pdf_pages_are_numbered_from_one(self, tmp_path):
        from app.recognition.engine import PageImage

        page = PageImage(tmp_path / "scan.pdf", Kind.PRINTED, page=4)
        assert page.stem.startswith("scan_p5")

    def test_pages_of_one_pdf_do_not_collide(self, pdf):
        stems = {p.stem for p in inputs.collect([pdf], Kind.PRINTED)}
        assert len(stems) == 3

    def test_same_name_in_different_folders_does_not_collide(self, tmp_path):
        """`arhiva/a/scan.pdf` si `arhiva/b/scan.pdf` sunt fisiere diferite.

        Cu nume derivat doar din `path.stem`, al doilea ar scrie peste iesirile
        primului si ar fi transcris din paginile lui — fara nicio eroare.
        """
        from app.recognition.engine import PageImage

        first = tmp_path / "a" / "scan.pdf"
        second = tmp_path / "b" / "scan.pdf"
        for path in (first, second):
            path.parent.mkdir(parents=True)
            path.write_bytes(b"x")

        left = PageImage(first, Kind.PRINTED, page=0)
        right = PageImage(second, Kind.PRINTED, page=0)
        assert left.stem != right.stem

    def test_the_stem_is_stable_between_runs(self, tmp_path):
        """Un lot reluat trebuie sa scrie in aceleasi fisiere ca prima data."""
        from app.recognition.engine import PageImage

        path = tmp_path / "cantec.png"
        path.write_bytes(b"x")
        assert PageImage(path, Kind.HANDWRITTEN).stem == PageImage(path, Kind.HANDWRITTEN).stem


class TestRender:
    def test_a_pdf_page_renders_to_png(self, pdf, tmp_path):
        path = inputs.render_pdf_page(pdf, 1, tmp_path / "out")
        assert path.exists() and path.suffix == ".png"

    def test_rendering_is_cached(self, pdf, tmp_path):
        """Randarea e scumpa; a doua oara trebuie sa refoloseasca fisierul."""
        first = inputs.render_pdf_page(pdf, 0, tmp_path / "out")
        stamp = first.stat().st_mtime_ns
        again = inputs.render_pdf_page(pdf, 0, tmp_path / "out")
        assert again == first and again.stat().st_mtime_ns == stamp

    def test_resolve_passes_images_through_untouched(self, tmp_path):
        from app.recognition.engine import PageImage

        path = tmp_path / "poza.png"
        path.write_bytes(b"x")
        page = PageImage(path, Kind.HANDWRITTEN)
        assert inputs.resolve(page, tmp_path / "work") == path

    def test_resolve_renders_pdf_pages(self, pdf, tmp_path):
        from app.recognition.engine import PageImage

        page = PageImage(pdf, Kind.PRINTED, page=2)
        resolved = inputs.resolve(page, tmp_path / "work")
        assert resolved.suffix == ".png" and resolved.exists()
