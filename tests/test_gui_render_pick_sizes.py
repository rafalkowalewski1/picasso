"""Picks of different sizes and shapes in ``picasso.gui.render``.

Every pick keeps the size it was drawn with; the size in the Tools
settings applies to the next picks and "Apply to all" resizes the
existing ones. Changing the pick shape keeps the picks drawn so far, so
shapes can be mixed. These tests drive the real mouse handlers of a
``View``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore, QtWidgets

from picasso import io, postprocess
from picasso.gui import render as gui_render

WIDTH = HEIGHT = 32.0
PIXELSIZE = 130.0
FOV = WIDTH * PIXELSIZE  # nm
VIEW_SIZE = 256  # display pixels, one per FOV / VIEW_SIZE nm


def _locs(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
            "x": rng.uniform(0.0, FOV, size=n),
            "y": rng.uniform(0.0, FOV, size=n),
            "lpx": np.full(n, 0.1 * PIXELSIZE),
            "lpy": np.full(n, 0.1 * PIXELSIZE),
            "photons": np.full(n, 1000.0),
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


class _Event:
    """The parts of a Qt mouse event the pick handlers read."""

    def __init__(self, x, y, button=QtCore.Qt.MouseButton.LeftButton):
        self._pos = QtCore.QPoint(int(x), int(y))
        self._button = button

    def pos(self):
        return self._pos

    def button(self):
        return self._button

    def buttons(self):
        return self._button

    def modifiers(self):
        return QtCore.Qt.KeyboardModifier.NoModifier

    def accept(self):
        pass

    def ignore(self):
        pass


def _screen(x_nm: float, y_nm: float) -> tuple[float, float]:
    """Display position of a point in nm (the view shows the FOV)."""
    return x_nm / FOV * VIEW_SIZE, y_nm / FOV * VIEW_SIZE


def _click(view, x_nm, y_nm, button=QtCore.Qt.MouseButton.LeftButton):
    x, y = _screen(x_nm, y_nm)
    view.mousePressEvent(_Event(x, y, button))
    view.mouseReleaseEvent(_Event(x, y, button))


def _drag(view, start_nm, end_nm):
    (x0, y0), (x1, y1) = _screen(*start_nm), _screen(*end_nm)
    view.mousePressEvent(_Event(x0, y0))
    view.mouseMoveEvent(_Event(x1, y1))
    view.mouseReleaseEvent(_Event(x1, y1))


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window holding one channel, in Pick mode with circles."""
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (FOV, FOV)]
    window.view.resize(VIEW_SIZE, VIEW_SIZE)
    window.view._mode = "Pick"
    window.tools_settings_dialog.pick_shape.setCurrentText("Circle")
    return window


def _set_shape(window, shape: str) -> None:
    window.tools_settings_dialog.pick_shape.setCurrentText(shape)
    assert window.view._pick_shape == shape


class TestPickSizes:
    def test_the_size_applies_to_new_picks_only(self, window):
        view = window.view
        diameter = window.tools_settings_dialog.pick_diameter
        diameter.setValue(100.0)
        _click(view, 1000.0, 1000.0)
        diameter.setValue(300.0)
        _click(view, 3000.0, 3000.0)
        assert view._pick_shapes == ["Circle", "Circle"]
        assert view._pick_sizes == [100.0, 300.0]

    def test_apply_to_all(self, window):
        view = window.view
        dialog = window.tools_settings_dialog
        dialog.pick_diameter.setValue(100.0)
        _click(view, 1000.0, 1000.0)
        _set_shape(window, "Square")
        _click(view, 3000.0, 1000.0)
        _set_shape(window, "Circle")
        dialog.pick_diameter.setValue(250.0)
        _click(view, 3000.0, 3000.0)
        dialog.pick_diameter.setValue(400.0)
        dialog.pick_circle_settings.apply_to_all.click()
        assert view._pick_sizes == [400.0, 100.0, 400.0]

    def test_picked_locs_use_each_pick_size(self, window):
        view = window.view
        diameter = window.tools_settings_dialog.pick_diameter
        diameter.setValue(100.0)
        _click(view, 1000.0, 1000.0)
        diameter.setValue(600.0)
        _click(view, 3000.0, 3000.0)
        picked = view.picked_locs(0)
        for pick, size, locs in zip(view._picks, [100.0, 600.0], picked):
            distance = np.hypot(locs["x"] - pick[0], locs["y"] - pick[1])
            assert distance.max() < size / 2
            assert distance.max() > 0.8 * size / 2

    def test_rectangle_widths(self, window):
        view = window.view
        _set_shape(window, "Rectangle")
        width = window.tools_settings_dialog.pick_width
        width.setValue(80.0)
        _drag(view, (500.0, 500.0), (2000.0, 500.0))
        width.setValue(200.0)
        _drag(view, (500.0, 2500.0), (2000.0, 2500.0))
        assert view._pick_sizes == [80.0, 200.0]
        areas = view.pick_areas() * 1e6  # nm^2
        lengths = [np.hypot(*np.subtract(*pick)) for pick in view._picks]
        np.testing.assert_allclose(
            areas, [80.0 * lengths[0], 200.0 * lengths[1]]
        )


class TestMixedShapes:
    def test_changing_the_shape_keeps_the_picks(self, window):
        view = window.view
        _click(view, 1000.0, 1000.0)
        _set_shape(window, "Box")
        _drag(view, (2000.0, 2000.0), (3000.0, 2600.0))
        assert view._pick_shapes == ["Circle", "Box"]
        assert len(view._picks) == 2
        assert view._pick_sizes[1] is None

    def test_right_click_removes_the_pick_under_the_cursor(self, window):
        view = window.view
        _click(view, 1000.0, 1000.0)
        _set_shape(window, "Box")
        _drag(view, (2000.0, 2000.0), (3000.0, 2600.0))
        _set_shape(window, "Square")
        _click(view, 3500.0, 500.0)
        # the box is removed with the circle tool active
        _set_shape(window, "Circle")
        _click(view, 2500.0, 2300.0, QtCore.Qt.MouseButton.RightButton)
        assert view._pick_shapes == ["Circle", "Square"]
        assert np.allclose(view._picks[1], (3500.0, 500.0), atol=FOV / 256)

    def test_unfinished_polygon_is_dropped(self, window):
        view = window.view
        _click(view, 1000.0, 1000.0)
        _set_shape(window, "Polygon")
        _click(view, 2000.0, 2000.0)
        _click(view, 3000.0, 2000.0)
        assert view._pick_shapes == ["Circle", "Polygon"]
        _set_shape(window, "Circle")
        assert view._pick_shapes == ["Circle"]

    def test_a_polygon_after_another_pick_is_a_new_pick(self, window):
        view = window.view
        _click(view, 1000.0, 1000.0)
        _set_shape(window, "Polygon")
        _click(view, 2000.0, 2000.0)
        assert view._pick_shapes == ["Circle", "Polygon"]
        assert len(view._picks[1]) == 1
        # a right click undoes polygon vertices only
        view.mouseReleaseEvent(_Event(0, 0, QtCore.Qt.MouseButton.RightButton))
        view.mouseReleaseEvent(_Event(0, 0, QtCore.Qt.MouseButton.RightButton))
        assert view._pick_shapes == ["Circle"]

    def test_brush_strokes_do_not_merge_into_other_shapes(self, window):
        view = window.view
        window.tools_settings_dialog.pick_diameter.setValue(400.0)
        _click(view, 2000.0, 2000.0)
        _set_shape(window, "Brush")
        window.tools_settings_dialog.brush_width.setValue(100.0)
        _drag(view, (1900.0, 2000.0), (2100.0, 2000.0))
        assert view._pick_shapes == ["Circle", "Brush"]

    def test_save_and_load_round_trip(self, window, tmp_path):
        view = window.view
        _click(view, 1000.0, 1000.0)
        _set_shape(window, "Rectangle")
        _drag(view, (500.0, 2500.0), (2000.0, 2500.0))
        _set_shape(window, "Square")
        _click(view, 3500.0, 500.0)
        shapes, sizes = list(view._pick_shapes), list(view._pick_sizes)
        counts = [len(_) for _ in view.picked_locs(0)]
        path = str(tmp_path / "picks.yaml")
        view.save_picks(path)
        assert io.load_yaml(path)["Shape"] == "Mixed"
        view.clear_picks()
        _set_shape(window, "Box")
        view.load_picks(path)
        assert view._pick_shapes == shapes
        assert view._pick_sizes == sizes
        assert [len(_) for _ in view.picked_locs(0)] == counts
        # the tool is not changed by picks of several shapes
        assert view._pick_shape == "Box"

    def test_a_file_with_a_size_of_zero_is_reported(
        self, window, tmp_path, monkeypatch
    ):
        view = window.view
        _click(view, 1000.0, 1000.0)
        path = str(tmp_path / "picks.yaml")
        io.save_yaml(
            path,
            {
                "Shape": "Circle",
                "Centers": [[1.0, 1.0]],
                "Diameter (nm)": 0.0,
                io.FORMAT_VERSION_ATTR: io.FORMAT_VERSION,
            },
        )
        shown = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "critical",
            staticmethod(lambda parent, title, text: shown.append(text)),
        )
        view.load_picks(path)
        assert len(shown) == 1 and "need a size" in shown[0]
        assert len(view._picks) == 1  # the current picks stay

    def test_saved_metadata_lists_each_shape(self, window, tmp_path):
        view = window.view
        _click(view, 1000.0, 1000.0)
        _set_shape(window, "Square")
        window.tools_settings_dialog.pick_side_length.setValue(150.0)
        _click(view, 3000.0, 3000.0)
        path = str(tmp_path / "picked.hdf5")
        view.save_picked_locs(path, 0)
        _, info = io.load_locs(path)
        assert info[-1]["Pick Shape"] == ["Circle", "Square"]
        assert info[-1]["Pick Size (nm)"] == [
            window.tools_settings_dialog.pick_diameter.value(),
            150.0,
        ]
        assert len(info[-1]["Pick Areas (um^2)"]) == 2

    def test_saved_picked_locs_keep_every_pick(self, window, tmp_path):
        view = window.view
        _click(view, 1000.0, 1000.0)
        _set_shape(window, "Rectangle")
        _drag(view, (500.0, 2500.0), (2000.0, 2500.0))
        counts = [len(_) for _ in view.picked_locs(0)]
        path = str(tmp_path / "picked.hdf5")
        view.save_picked_locs(path, 0)
        saved, _ = io.load_locs(path)
        assert saved.groupby("group").size().tolist() == counts
        assert "x_pick_rot" not in saved.columns

    def test_filter_picks_counts_every_shape(self, window):
        view = window.view
        _click(view, 1000.0, 1000.0)
        _set_shape(window, "Box")
        _drag(view, (2000.0, 2000.0), (3000.0, 2600.0))
        counts = view._count_locs_in_picks(0)
        expected = [
            len(_)
            for _ in postprocess.picked_locs(
                view.locs[0],
                view.infos[0],
                picks=view._picks,
                pick_shape=view._pick_shapes,
                pick_size=view._pick_sizes,
            )
        ]
        assert list(counts) == expected

    def test_pick_similar_needs_one_shape(self, window, monkeypatch):
        view = window.view
        _click(view, 1000.0, 1000.0)
        _set_shape(window, "Square")
        _click(view, 3000.0, 3000.0)
        warnings = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *args, **kwargs: warnings.append(args[2]),
        )
        view.pick_similar()
        assert warnings == ["Pick similar needs picks of one shape."]
        assert len(view._picks) == 2

    def test_assigned_picks_take_the_tool_shape(self, window):
        view = window.view
        window.tools_settings_dialog.pick_diameter.setValue(120.0)
        view._picks = [(1000.0, 1000.0), (2000.0, 2000.0)]
        assert view._pick_shapes == ["Circle", "Circle"]
        assert view._pick_sizes == [120.0, 120.0]
