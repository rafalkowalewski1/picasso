"""The single-channel colormap of Render's Display settings is grayed
out while the rendering ignores it: with several channels, a grouped
channel or rendering by property.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso.gui import render as gui_render
from picasso.gui import rotation as gui_rotation

WIDTH = HEIGHT = 32.0
PIXELSIZE = 130.0


def _locs(n: int = 200, group: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    locs = pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
            "x": rng.uniform(0.0, WIDTH, size=n) * PIXELSIZE,
            "y": rng.uniform(0.0, HEIGHT, size=n) * PIXELSIZE,
            "lpx": np.full(n, 13.0),
            "lpy": np.full(n, 13.0),
            "photons": np.full(n, 1000.0),
        }
    )
    if group:
        locs["group"] = rng.integers(0, 3, size=n).astype(np.int32)
    return locs


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 1000,
            "Pixelsize": PIXELSIZE,
        }
    ]


@pytest.fixture
def window(qt_offscreen):
    return gui_render.Window(plugins_loaded=True)


def _add(window, tmp_path, name: str, **kwargs) -> None:
    window.view.add(
        str(tmp_path / name), _locs(**kwargs), _info(), render_=False
    )


def _enabled(dialog) -> bool:
    assert dialog.colormap_label.isEnabled() == dialog.colormap.isEnabled()
    return dialog.colormap.isEnabled()


def test_enabled_without_data(window):
    assert _enabled(window.display_settings_dlg)


def test_enabled_for_one_channel(window, tmp_path):
    _add(window, tmp_path, "a.hdf5")
    window.view._prepare_locs_for_rendering()
    dialog = window.display_settings_dlg
    assert _enabled(dialog)
    assert "Not used" not in dialog.colormap.toolTip()


def test_grayed_out_for_several_channels(window, tmp_path):
    _add(window, tmp_path, "a.hdf5")
    _add(window, tmp_path, "b.hdf5")
    window.view._prepare_locs_for_rendering()
    dialog = window.display_settings_dlg
    assert not _enabled(dialog)
    assert (
        gui_rotation.COLORMAP_UNUSED["multichannel"]
        in dialog.colormap.toolTip()
    )


def test_grayed_out_for_a_grouped_channel(window, tmp_path):
    _add(window, tmp_path, "a.hdf5", group=True)
    window.view._prepare_locs_for_rendering()
    assert not _enabled(window.display_settings_dlg)


def test_enabled_again_when_reason_goes_away(qt_offscreen):
    label = gui_rotation.QtWidgets.QLabel()
    combo = gui_rotation.QtWidgets.QComboBox()
    gui_rotation.set_colormap_unused(label, combo, "reason")
    assert not combo.isEnabled() and not label.isEnabled()
    gui_rotation.set_colormap_unused(label, combo, None)
    assert combo.isEnabled() and label.isEnabled()
    assert combo.toolTip() == gui_rotation.COLORMAP_TOOLTIP
