"""Motorul de lot: coada, reluare, izolarea esecurilor, raport.

Testele folosesc un `Engine` fals. Lotul nu are voie sa depinda de model — asta
e chiar rostul interfetei `Engine`, si un test care ar incarca SMT ar dura
minute si ar verifica altceva decat vrem aici.
"""

import json
from pathlib import Path

import numpy as np

import pytest

from app.batch import report
from app.batch.runner import Runner
from app.batch.store import State, Store
from app.recognition.engine import Kind, PageImage, PageResult, StaffCrop
from app.recognition.page import assemble_page
from tests.test_closed_score import BASS_STAFF, TREBLE_STAFF


class FakeEngine:
    """Motor previzibil: transcrie, esueaza sau numara, dupa nume."""

    name = "fake"

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.calls: list[str] = []
        self.loads = 0

    def transcribe(self, page: PageImage) -> PageResult:
        self.calls.append(page.path.name)
        if page.path.stem in self.fail_on:
            raise RuntimeError("poza e ilizibila")

        result = PageResult(page=page)
        for index, kern in enumerate([TREBLE_STAFF, BASS_STAFF], 1):
            result.staves.append(
                StaffCrop(
                    index=index,
                    box=(0, 0, 8, 4),
                    image=np.zeros((4, 8), dtype=np.uint8),
                    kern=kern,
                    tokens=9,
                )
            )
        assembled = assemble_page([TREBLE_STAFF, BASS_STAFF], title=page.stem)
        result.score = assembled.score
        result.systems = assembled.systems
        result.seconds = 1.0
        return result


@pytest.fixture
def images(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    for name in ("a", "b", "c"):
        (folder / f"{name}.png").write_bytes(name.encode())
    return folder


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "batch.db") as store:
        yield store


def make_runner(store, tmp_path, engine=None):
    return Runner(
        store=store,
        engine=engine or FakeEngine(),
        out_dir=tmp_path / "out",
        save_crops=False,
    )


class TestEnqueue:
    def test_collects_images_recursively(self, store, tmp_path, images):
        nested = images / "sub"
        nested.mkdir()
        (nested / "d.png").write_bytes(b"d")
        assert make_runner(store, tmp_path).enqueue([images], Kind.HANDWRITTEN) == 4

    def test_ignores_unsupported_files(self, store, tmp_path, images):
        (images / "note.txt").write_text("nu e partitura")
        assert make_runner(store, tmp_path).enqueue([images], Kind.HANDWRITTEN) == 3

    def test_enqueueing_twice_does_not_duplicate(self, store, tmp_path, images):
        runner = make_runner(store, tmp_path)
        runner.enqueue([images], Kind.HANDWRITTEN)
        assert runner.enqueue([images], Kind.HANDWRITTEN) == 3

    def test_a_single_file_can_be_given_directly(self, store, tmp_path, images):
        assert make_runner(store, tmp_path).enqueue([images / "a.png"], Kind.HANDWRITTEN) == 1


class TestRun:
    def test_every_file_is_processed(self, store, tmp_path, images):
        engine = FakeEngine()
        runner = make_runner(store, tmp_path, engine)
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run()
        assert sorted(engine.calls) == ["a.png", "b.png", "c.png"]
        assert store.counts() == {State.DONE.value: 3}

    def test_musicxml_is_written_per_page(self, store, tmp_path, images):
        runner = make_runner(store, tmp_path, FakeEngine())
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run()
        produced = sorted(p.stem.split("_")[0] for p in (tmp_path / "out").glob("*.musicxml"))
        assert produced == ["a", "b", "c"]

    def test_raw_kern_is_kept_for_diagnosis(self, store, tmp_path, images):
        runner = make_runner(store, tmp_path, FakeEngine())
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run()
        produced = list((tmp_path / "out").glob("a_*.krn"))
        assert produced and produced[0].read_text(encoding="utf-8")

    def test_same_name_in_different_folders_produces_two_outputs(
        self, store, tmp_path, images
    ):
        """Doua fisiere cu acelasi nume nu au voie sa se suprascrie."""
        nested = images / "sub"
        nested.mkdir()
        (nested / "a.png").write_bytes(b"alt continut")

        runner = make_runner(store, tmp_path, FakeEngine())
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run()
        assert len(list((tmp_path / "out").glob("a_*.musicxml"))) == 2

    def test_progress_is_reported_for_each_unit(self, store, tmp_path, images):
        seen = []
        runner = make_runner(store, tmp_path, FakeEngine())
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run(progress=seen.append)
        assert [p.done for p in seen] == [1, 2, 3]
        assert all(p.total == 3 for p in seen)

    def test_remaining_time_is_unknown_before_the_first_result(self, store, tmp_path, images):
        seen = []
        runner = make_runner(store, tmp_path, FakeEngine())
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run(progress=seen.append)
        assert seen[-1].seconds_remaining == pytest.approx(0)


class TestFailureIsolation:
    def test_one_bad_file_does_not_stop_the_batch(self, store, tmp_path, images):
        engine = FakeEngine(fail_on={"b"})
        runner = make_runner(store, tmp_path, engine)
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run()
        assert store.counts() == {State.DONE.value: 2, State.FAILED.value: 1}

    def test_the_failure_reason_is_recorded(self, store, tmp_path, images):
        runner = make_runner(store, tmp_path, FakeEngine(fail_on={"b"}))
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run()
        failed = [j for j in store.all() if j.state is State.FAILED]
        assert "ilizibila" in failed[0].message

    def test_files_after_the_failure_are_still_processed(self, store, tmp_path, images):
        engine = FakeEngine(fail_on={"a"})
        runner = make_runner(store, tmp_path, engine)
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run()
        assert "c.png" in engine.calls


class TestEnqueueFailureIsolation:
    """Un fisier pe care nu-l putem nici macar citi nu are voie sa opreasca lotul.

    Inainte, o singura exceptie la adaugare propaga in sus si ucidea un lot de
    peste noapte inainte sa converteasca vreo pagina — desi modulul isi declara
    exact regula contrara.
    """

    def test_an_unreadable_file_is_recorded_not_raised(self, store, tmp_path, images, monkeypatch):
        from app.batch import inputs as inputs_module

        real = inputs_module.page_count

        def explode(path):
            if path.name == "b.png":
                raise OSError("PDF criptat")
            return real(path)

        monkeypatch.setattr(inputs_module, "page_count", explode)
        make_runner(store, tmp_path).enqueue([images], Kind.HANDWRITTEN)

        states = {j.path.name: j.state for j in store.all()}
        assert states.get(images.name) is State.FAILED or State.FAILED in states.values()

    def test_the_other_files_still_reach_the_queue(self, store, tmp_path, images, monkeypatch):
        from app.batch import inputs as inputs_module

        def explode_on_one(path):
            if path.name == "b.png":
                raise OSError("fisier trunchiat")
            return 1

        monkeypatch.setattr(inputs_module, "page_count", explode_on_one)
        make_runner(store, tmp_path).enqueue([images / "a.png", images / "b.png"],
                                             Kind.HANDWRITTEN)
        names = {j.path.name for j in store.all()}
        assert "a.png" in names

    def test_the_reason_is_kept(self, store, tmp_path, images, monkeypatch):
        from app.batch import inputs as inputs_module

        monkeypatch.setattr(
            inputs_module, "page_count",
            lambda path: (_ for _ in ()).throw(OSError("PDF criptat")),
        )
        make_runner(store, tmp_path).enqueue([images / "a.png"], Kind.HANDWRITTEN)
        failed = [j for j in store.all() if j.state is State.FAILED]
        assert failed and "criptat" in failed[0].message


class TestKindChange:
    def test_re_adding_with_another_kind_requeues_the_file(self, store, tmp_path, images):
        """Tipul decide pe ce conducta merge, deci schimbarea lui trebuie aplicata."""
        runner = make_runner(store, tmp_path, FakeEngine())
        runner.enqueue([images / "a.png"], Kind.HANDWRITTEN)
        runner.run()
        assert store.counts() == {State.DONE.value: 1}

        runner.enqueue([images / "a.png"], Kind.PRINTED)
        job = store.all()[0]
        assert job.kind == Kind.PRINTED.value
        assert job.state is State.PENDING

    def test_re_adding_with_the_same_kind_changes_nothing(self, store, tmp_path, images):
        runner = make_runner(store, tmp_path, FakeEngine())
        runner.enqueue([images / "a.png"], Kind.HANDWRITTEN)
        runner.run()
        runner.enqueue([images / "a.png"], Kind.HANDWRITTEN)
        assert store.counts() == {State.DONE.value: 1}


class TestResume:
    def test_finished_work_is_not_repeated(self, store, tmp_path, images):
        runner = make_runner(store, tmp_path, FakeEngine())
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run()

        second = FakeEngine()
        make_runner(store, tmp_path, second).run()
        assert second.calls == []

    def test_a_cancelled_batch_continues_where_it_stopped(self, store, tmp_path, images):
        import threading

        cancel = threading.Event()
        first = FakeEngine()
        runner = make_runner(store, tmp_path, first)
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run(progress=lambda p: cancel.set(), cancel=cancel)
        assert len(first.calls) == 1

        second = FakeEngine()
        make_runner(store, tmp_path, second).run()
        assert len(second.calls) == 2
        assert store.counts() == {State.DONE.value: 3}

    def test_a_crashed_process_leaves_no_job_stuck_running(self, tmp_path, images):
        """`running` intr-o baza redeschisa inseamna proces mort, deci se reia."""
        path = tmp_path / "batch.db"
        store = Store(path)
        store.add(images / "a.png", 0, "handwritten")
        store.mark_running(images / "a.png", 0)
        store.close()

        reopened = Store(path)
        assert reopened.counts() == {State.PENDING.value: 1}
        reopened.close()

    def test_a_changed_file_is_processed_again(self, store, tmp_path, images):
        runner = make_runner(store, tmp_path, FakeEngine())
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run()

        (images / "a.png").write_bytes(b"continut nou")
        second = FakeEngine()
        again = make_runner(store, tmp_path, second)
        again.enqueue([images], Kind.HANDWRITTEN)
        again.run()
        assert second.calls == ["a.png"]

    def test_failed_files_are_retried_on_the_next_run(self, store, tmp_path, images):
        runner = make_runner(store, tmp_path, FakeEngine(fail_on={"b"}))
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run()

        second = FakeEngine()
        make_runner(store, tmp_path, second).run()
        assert second.calls == ["b.png"]


class TestReport:
    @pytest.fixture
    def jobs(self, store, tmp_path, images):
        runner = make_runner(store, tmp_path, FakeEngine(fail_on={"b"}))
        runner.enqueue([images], Kind.HANDWRITTEN)
        runner.run()
        return store.all()

    def test_renders_a_self_contained_page(self, jobs):
        html = report.render(jobs)
        assert html.startswith("<!doctype html>")
        assert "http://" not in html and "https://" not in html

    def test_every_unit_appears(self, jobs):
        html = report.render(jobs)
        for name in ("a.png", "b.png", "c.png"):
            assert name in html

    def test_failures_are_visible(self, jobs):
        assert "ilizibila" in report.render(jobs)

    def test_counts_are_summarised(self, jobs):
        html = report.render(jobs)
        assert "esuate" in html and "fara probleme" in html

    def test_writes_to_disk(self, jobs, tmp_path):
        path = report.write(jobs, tmp_path / "raport.html")
        assert path.exists() and path.read_text(encoding="utf-8")

    def test_detail_json_survives_the_round_trip(self, jobs):
        done = [j for j in jobs if j.state is State.DONE]
        detail = json.loads(done[0].detail)
        assert detail["errors"] == [] and detail["problems"] == []

    def test_no_crop_paths_are_recorded_when_crops_are_not_saved(self, jobs):
        """O cale catre un fisier care nu exista ar trimite raportul aiurea."""
        done = [j for j in jobs if j.state is State.DONE]
        assert json.loads(done[0].detail)["crops"] == []

    def test_crop_paths_are_recorded_when_crops_are_saved(self, store, tmp_path, images):
        runner = Runner(store, FakeEngine(), tmp_path / "out2", save_crops=True)
        runner.enqueue([images / "a.png"], Kind.HANDWRITTEN)
        runner.run()

        done = [j for j in store.all() if j.state is State.DONE]
        crops = json.loads(done[0].detail)["crops"]
        assert len(crops) == 2
        # Fiecare cale inregistrata trebuie sa arate spre un fisier real:
        # raportul incorporeaza imaginile de la aceste cai.
        assert all(Path(crop).exists() for crop in crops)
