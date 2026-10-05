"""Interfata grafica: detectia MuseScore, firul de lucru, popularea tabelului.

Testam ce contine logica, nu aspectul. Rulam fara ecran (`QT_QPA_PLATFORM=offscreen`),
ca suita sa mearga si intr-un terminal fara sesiune grafica.
"""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from app.batch.store import Job, State  # noqa: E402
from app.ui import musescore  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def make_job(**kwargs) -> Job:
    from pathlib import Path

    defaults = dict(
        path=Path("a.png"), page=0, kind="handwritten", state=State.DONE,
        systems=4, staves=8, errors=0, warnings=1, seconds=9.4,
    )
    defaults.update(kwargs)
    return Job(**defaults)


class TestMuseScoreDetection:
    def test_environment_override_wins(self, monkeypatch, tmp_path):
        fake = tmp_path / "MuseScore4.exe"
        fake.write_bytes(b"")
        monkeypatch.setenv("MUSESCORE", str(fake))
        assert musescore.find() == fake

    def test_a_missing_override_is_ignored(self, monkeypatch, tmp_path):
        """O cale gresita in mediu nu trebuie sa ascunda instalarea reala."""
        monkeypatch.setenv("MUSESCORE", str(tmp_path / "nu_exista.exe"))
        monkeypatch.setattr(musescore, "CANDIDATES", ())
        monkeypatch.setattr(musescore.shutil, "which", lambda _: None)
        assert musescore.find() is None

    def test_returns_none_when_not_installed(self, monkeypatch):
        monkeypatch.delenv("MUSESCORE", raising=False)
        monkeypatch.setattr(musescore, "CANDIDATES", ())
        monkeypatch.setattr(musescore.shutil, "which", lambda _: None)
        assert musescore.find() is None

    def test_opening_without_musescore_raises_a_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(musescore, "find", lambda: None)
        with pytest.raises(FileNotFoundError, match="MuseScore nu a fost gasit"):
            musescore.open_score(tmp_path / "a.musicxml")

    def test_first_run_wizard_is_detected(self, monkeypatch, tmp_path):
        """Pana la prima pornire, MuseScore blocheaza si interfata si CLI-ul."""
        monkeypatch.setattr(musescore.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        assert musescore.is_configured() is False
        (tmp_path / "MuseScore").mkdir()
        assert musescore.is_configured() is True


class TestWorker:
    def test_stop_sets_the_cancel_flag(self, qt_app, tmp_path):
        from app.recognition.engine import Kind
        from app.ui.worker import BatchWorker

        worker = BatchWorker(tmp_path / "b.db", tmp_path / "out", Kind.HANDWRITTEN)
        assert not worker.cancel.is_set()
        worker.stop()
        assert worker.cancel.is_set()

    def test_a_crash_is_reported_not_raised(self, qt_app, tmp_path, monkeypatch):
        """Un esec trebuie sa ajunga in fereastra, nu sa omoare firul in tacere."""
        from app.recognition.engine import Kind
        from app.ui import worker as worker_module

        monkeypatch.setattr(
            worker_module, "engine_for",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model lipsa")),
        )
        worker = worker_module.BatchWorker(tmp_path / "b.db", tmp_path / "out", Kind.HANDWRITTEN)
        seen = []
        worker.failed.connect(seen.append)
        worker.run()
        assert seen and "model lipsa" in seen[0]

    def test_a_kind_flattened_to_a_plain_string_is_accepted(self, qt_app, tmp_path):
        """Qt aplatizeaza `Kind` la `str` prin `QComboBox.currentData()`.

        `Kind` mosteneste din `str`, deci trece prin QVariant ca sir simplu si
        pierde tipul. Lucratorii trebuie sa-l readuca la enum, altfel primul
        `kind.value` crapa si intreaga adaugare esueaza.
        """
        from app.recognition.engine import Kind
        from app.ui.worker import BatchWorker, EnqueueWorker

        assert EnqueueWorker(tmp_path / "b.db", "handwritten", []).kind is Kind.HANDWRITTEN
        assert BatchWorker(tmp_path / "b.db", tmp_path, "printed").kind is Kind.PRINTED

    def test_handwritten_runs_with_cleaning_on(self, qt_app, tmp_path, monkeypatch):
        """Fereastra n-are control pentru curatare, deci trebuie sa aleaga bine.

        §15.1 a masurat ca pe scris de mana curatarea dubleaza notele corecte
        (5,5% -> 11,0%). Fara asta, cine converteste din interfata primeste
        tocmai setarea care s-a dovedit mai proasta pe materialul lui.
        """
        from app.recognition.engine import Kind
        from app.ui import worker as worker_module
        from tests.test_batch import FakeEngine

        primite = {}

        def fals(kind, **kwargs):
            primite[kind] = kwargs
            return FakeEngine()

        monkeypatch.setattr(worker_module, "engine_for", fals)
        for kind in (Kind.HANDWRITTEN, Kind.PRINTED):
            worker_module.BatchWorker(tmp_path / f"{kind.value}.db", tmp_path, kind).run()

        assert primite[Kind.HANDWRITTEN]["clean"] is True
        # Pe tiparit curat n-are ce repara, iar orice pas care sterge cerneala
        # e mai bine sa fie cerut decat presupus.
        assert primite[Kind.PRINTED]["clean"] is False

    def test_a_clean_run_emits_finished(self, qt_app, tmp_path, monkeypatch):
        from app.recognition.engine import Kind
        from app.ui import worker as worker_module
        from tests.test_batch import FakeEngine

        monkeypatch.setattr(worker_module, "engine_for", lambda *a, **k: FakeEngine())
        source = tmp_path / "in"
        source.mkdir()
        (source / "a.png").write_bytes(b"a")

        worker = worker_module.BatchWorker(
            tmp_path / "b.db", tmp_path / "out", Kind.HANDWRITTEN, [source]
        )
        done = []
        worker.finished.connect(done.append)
        worker.run()
        assert len(done) == 1 and len(done[0]) == 1


class TestChoosingSources:
    """Se pot alege si fisiere individuale, nu doar foldere.

    Prima versiune avea un singur buton, legat la selectorul de *foldere*: nu
    exista nicio cale sa adaugi o singura poza.
    """

    @pytest.fixture
    def window(self, qt_app, tmp_path):
        from app.ui.main_window import MainWindow

        return MainWindow(out_dir=tmp_path / "out")

    def test_both_a_file_picker_and_a_folder_picker_exist(self, window):
        assert window.choose_files_button.isEnabled()
        assert window.choose_folder_button.isEnabled()

    def test_choosing_files_sends_them_straight_to_the_queue(
        self, window, monkeypatch, tmp_path
    ):
        """Fisierele intra in coada la alegere, nu abia la apasarea butonului."""
        from PySide6.QtWidgets import QFileDialog

        first, second = tmp_path / "a.png", tmp_path / "b.jpg"
        for path in (first, second):
            path.write_bytes(b"x")
        monkeypatch.setattr(
            QFileDialog, "getOpenFileNames",
            staticmethod(lambda *a, **k: ([str(first), str(second)], "")),
        )
        started = []
        monkeypatch.setattr(window, "_add_paths", started.extend)
        window._choose_files()
        assert started == [first, second]

    def test_cancelling_the_dialog_queues_nothing(self, window, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        monkeypatch.setattr(
            QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([], ""))
        )
        called = []
        monkeypatch.setattr(window, "_add_paths", called.append)
        window._choose_files()
        assert called == []

    def test_the_filter_offers_every_supported_format(self, window, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        from app.batch import inputs

        captured = {}

        def fake(parent, title, directory, filters):
            captured["filters"] = filters
            return ([], "")

        monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(fake))
        window._choose_files()
        for suffix in inputs.SUPPORTED:
            assert f"*{suffix}" in captured["filters"]

    def test_pdfs_can_be_chosen(self, window, monkeypatch):
        """Partiturile tiparite vin ca PDF scanat, nu ca poze."""
        from PySide6.QtWidgets import QFileDialog

        captured = {}
        monkeypatch.setattr(
            QFileDialog, "getOpenFileNames",
            staticmethod(lambda p, t, d, f: (captured.update(filters=f), ([], ""))[1]),
        )
        window._choose_files()
        assert "*.pdf" in captured["filters"]

    def test_adding_files_persists_them_and_shows_them(
        self, window, tmp_path, qt_app, monkeypatch
    ):
        """Fisierele adaugate trebuie sa apara in tabel si sa supravietuiasca
        inchiderii aplicatiei — nu doar sa stea intr-o lista din memorie."""
        from app.batch.store import State, Store

        # Dialogul modal de eroare ar bloca la nesfarsit fara ecran. Il inlocuim
        # cu o colectare, ca un esec sa iasa la iveala ca test cazut, nu ca
        # suita blocata — asa a fost ascuns bug-ul cu `Kind` aplatizat la `str`.
        failures: list[str] = []
        monkeypatch.setattr(window, "_on_failed", failures.append)

        poza = tmp_path / "poza.png"
        poza.write_bytes(b"x")

        window._add_paths([poza])
        assert window.enqueue_thread is not None

        # `QThread.wait()` ar bloca firul principal — iar `worker.finished ->
        # thread.quit` e o conexiune in coada, care are nevoie exact de bucla
        # firului principal ca sa fie livrata. Ar fi deadlock. Pompam bucla.
        deadline = time.monotonic() + 20
        while window.enqueue_thread is not None and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.01)
        assert not failures, "adaugarea a esuat:\n" + "\n".join(failures)
        assert window.enqueue_thread is None, "firul de adaugare nu s-a incheiat"

        with Store(window.db_path) as store:
            jobs = store.all()
        assert [j.path for j in jobs] == [poza]
        assert jobs[0].state is State.PENDING
        assert window.table.rowCount() == 1

    def test_adding_nothing_starts_no_worker(self, window):
        window._add_paths([])
        assert window.enqueue_worker is None


class TestTable:
    @pytest.fixture
    def window(self, qt_app, tmp_path):
        from app.ui.main_window import MainWindow

        return MainWindow(out_dir=tmp_path / "out")

    def test_starts_empty(self, window):
        assert window.table.rowCount() == 0

    def test_one_row_per_job(self, window):
        window._fill_table([make_job(), make_job(path=__import__("pathlib").Path("b.png"))])
        assert window.table.rowCount() == 2

    def test_missing_numbers_show_a_dash(self, window):
        window._fill_table([make_job(state=State.PENDING, systems=None, seconds=None)])
        assert window.table.item(0, 4).text() == "—"

    def test_state_is_shown_in_romanian(self, window):
        window._fill_table([make_job(state=State.FAILED)])
        assert window.table.item(0, 2).text() == "esuat"

    def test_a_failure_reason_becomes_a_tooltip(self, window):
        window._fill_table([make_job(state=State.FAILED, message="poza e ilizibila")])
        assert "ilizibila" in window.table.item(0, 0).toolTip()

    def test_a_finished_score_offers_musescore_and_correction(self, window):
        """Doua butoane, si fac lucruri diferite (§15.24).

        MuseScore e pentru editat partitura. Corectarea e pentru **legat corectia
        de poza din care a iesit**, adica pentru a face din ea material de invatat
        — ceva ce MuseScore n-are cum sa faca, fiindca nu stie de unde a venit
        partitura.
        """
        from PySide6.QtWidgets import QPushButton

        window._fill_table([make_job(musicxml="a.musicxml")])
        actions = window.table.cellWidget(0, window.table.columnCount() - 1)
        labels = [b.text() for b in actions.findChildren(QPushButton)]
        assert labels == ["MuseScore", "Corecteaza"]

    def test_a_page_without_a_score_offers_an_explanation(self, window):
        """Cand nu iese partitura, butonul trebuie sa spuna de ce, nu sa lipseasca."""
        from PySide6.QtWidgets import QPushButton

        window._fill_table([make_job(musicxml=None, detail='{"problems": ["lipseste *^"]}')])
        actions = window.table.cellWidget(0, window.table.columnCount() - 1)
        labels = [b.text() for b in actions.findChildren(QPushButton)]
        assert labels == ["De ce?"]
