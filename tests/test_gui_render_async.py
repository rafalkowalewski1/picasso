"""The asynchronous render worker of ``picasso.gui.render.View``.

Full renders run on a worker thread with a latest-wins queue
(``RenderWorker``): a burst of pan/zoom events collapses into rendering
the newest state, stale results are discarded by request id, and the
result must be pixel-identical to the synchronous path. The rest of the
suite runs with ``async_rendering`` disabled (see conftest); these tests
re-enable it explicitly.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from picasso import render
from picasso.gui import render as gui_render


WIDTH = HEIGHT = 64.0
PIXELSIZE = 130.0


def _locs(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(0.0, WIDTH, size=n),
            "y": rng.uniform(0.0, HEIGHT, size=n),
            "lpx": rng.uniform(0.05, 0.3, size=n),
            "lpy": rng.uniform(0.05, 0.3, size=n),
            "photons": rng.uniform(500.0, 5000.0, size=n),
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
        }
    )


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 1000,
            "Pixelsize": PIXELSIZE,
        }
    ]


def _qimage_bytes(qimage) -> bytes:
    bits = qimage.bits()
    bits.setsize(qimage.sizeInBytes())
    return bytes(bits)


def _wait_until(qapp, condition, timeout: float = 15.0) -> None:
    start = time.monotonic()
    while not condition():
        qapp.processEvents()
        time.sleep(0.005)
        if time.monotonic() - start > timeout:
            raise TimeoutError("render worker result did not arrive")


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window with one channel and async rendering enabled."""
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(128, 128)
    window.view.async_rendering = True  # instance attr shadows the stub
    yield window
    window.view.stop_render_worker()


class TestAsyncRender:
    def test_async_matches_sync(self, window, qapp):
        view = window.view
        view.update_scene()
        request_id = view._render_request_id
        assert request_id > 0
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        async_qimage = view.qimage
        async_raw = view.image.copy()

        # synchronous reference of the same state
        view.async_rendering = False
        view.image = None
        view.update_scene()
        assert view.image is not None
        np.testing.assert_array_equal(async_raw, view.image)
        assert _qimage_bytes(async_qimage) == _qimage_bytes(view.qimage)

    def test_burst_coalesces(self, window, qapp, monkeypatch):
        view = window.view
        calls = []
        original = render.render_scene

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(render, "render_scene", counting)
        n_requests = 6
        for _ in range(n_requests):
            view.update_scene()
        final_id = view._render_request_id
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        # drain any second render picked up after the first finished
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert len(calls) <= 2 < n_requests
        assert view._render_request_id == final_id

    def test_stale_result_discarded(self, window, qapp):
        view = window.view
        zoomed = ((0.0, 0.0), (HEIGHT / 2, WIDTH / 2))
        view.update_scene()
        stale_id = view._render_request_id
        # supersede immediately with a different viewport; a result is
        # only applied when its id matches the newest request, so
        # view.image stays unset until the *zoomed* render lands
        view.update_scene(viewport=zoomed)
        assert view._render_request_id > stale_id
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        async_raw = view.image.copy()

        # the applied frame must equal a synchronous render of the
        # final (zoomed) viewport, not of the superseded overview
        view.async_rendering = False
        view.update_scene(viewport=zoomed)
        np.testing.assert_array_equal(async_raw, view.image)

    def test_worker_stops_cleanly(self, window, qapp):
        view = window.view
        view.update_scene()
        view.stop_render_worker()
        assert view._render_thread is None
        # stopping twice is a no-op
        view.stop_render_worker()

    def test_interactive_preview_and_refine(self, window, qapp, monkeypatch):
        view = window.view
        monkeypatch.setattr(
            gui_render.View, "_interaction_subsample_target", lambda self: 500
        )
        seen = []
        original = render.render_scene

        def recording(*args, **kwargs):
            locs = kwargs["locs"]
            channels = [locs] if isinstance(locs, pd.DataFrame) else locs
            seen.append((sum(len(c) for c in channels), kwargs["contrast"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(render, "render_scene", recording)
        view.update_scene(interactive=True)
        # the preview renders the strided subsample; the refine timer
        # follows up with the full-quality render, which is the one
        # allowed to update the caches and spinboxes. The dialog values
        # themselves shift with the dynamic display pixel size, so the
        # preview is checked against the refine render's contrast: it
        # must be the same limits scaled by the sampled fraction.
        _wait_until(qapp, lambda: len(seen) >= 2)
        n_preview, contrast_preview = seen[0]
        n_full, contrast_full = seen[1]
        assert n_preview == 500
        assert n_full == 3000
        fraction = n_preview / n_full
        assert contrast_preview == pytest.approx(
            (contrast_full[0] * fraction, contrast_full[1] * fraction)
        )
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)

    def test_interaction_subsample_setting(self, window, monkeypatch):
        view = window.view

        def with_value(value):
            settings = {"Render": {"interaction_subsample": value}}
            monkeypatch.setattr(
                gui_render.io, "load_user_settings", lambda: settings
            )
            return view._interaction_subsample_target()

        assert with_value("off") == 0
        assert with_value(0) == 0
        assert with_value(1234) == 1234
        assert with_value(True) == gui_render.INTERACTION_SUBSAMPLE_AUTO
        assert with_value("auto") == gui_render.INTERACTION_SUBSAMPLE_AUTO
        assert with_value(-5) == gui_render.INTERACTION_SUBSAMPLE_AUTO

    def test_instant_geometric_preview(self, window, qapp):
        view = window.view
        # establish a fully rendered frame first
        view.update_scene()
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        before = _qimage_bytes(view.qimage)
        shown_before = view._displayed_viewport
        # an interactive viewport change must update the displayed frame
        # synchronously (geometric blit of the stale frame), before any
        # worker result can possibly be applied
        shifted = ((10.0, 10.0), (10.0 + HEIGHT, 10.0 + WIDTH))
        view.update_scene(viewport=shifted, interactive=True)
        assert view._displayed_viewport != shown_before
        assert view._displayed_viewport == (
            tuple(view.viewport[0]),
            tuple(view.viewport[1]),
        )
        assert _qimage_bytes(view.qimage) != before

    def test_stale_result_adopted_for_display(self, window, qapp):
        view = window.view
        view.update_scene()
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        image_cache = view.image
        n_locs_before = view.n_locs
        shown_before = _qimage_bytes(view.qimage)
        # fabricate a superseded completion: older id, shifted viewport
        stale_viewport = ((5.0, 5.0), (5.0 + HEIGHT, 5.0 + WIDTH))
        view._render_request_id += 1  # pretend a newer request exists
        fake = view.qimage_no_picks.copy()
        view._on_render_finished(
            view._render_request_id - 1,
            stale_viewport,
            fake,
            123,
            (0.0, 1.0),
            np.zeros((4, 4), dtype=np.float32),
        )
        # adopted for display: repositioned into the current viewport
        assert view._displayed_viewport == view._viewport_key()
        assert _qimage_bytes(view.qimage) != shown_before
        # ...but caches and spinboxes stay untouched by stale results
        assert view.image is image_cache
        assert view.n_locs == n_locs_before

    def test_remove_locs_stops_worker(self, window, qapp):
        # "Remove all localizations" rebuilds the view; the old view's
        # worker thread must be stopped first — a running QThread being
        # destroyed aborts the whole process (field-reported crash)
        old_view = window.view
        window.remove_locs()
        assert old_view._render_thread is None
        assert window.view is not old_view
        assert window.view._render_thread is not None
        window.view.stop_render_worker()
