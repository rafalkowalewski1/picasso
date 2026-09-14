"""Picasso: Render and the user settings file: the ``Render`` defaults
are written on start, and an unreadable file is reported instead of
silently replaced.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

import pytest
import yaml
from PyQt6 import QtWidgets

from picasso import io
from picasso.gui import render as gui_render


@pytest.fixture
def settings_path(tmp_path):
    return tmp_path / "settings.yaml"  # see ``isolated_user_settings``


def test_render_writes_missing_defaults_on_start(qt_offscreen, settings_path):
    io.save_user_settings({"Render": {"Colormap": "hot"}})
    window = gui_render.Window(plugins_loaded=True)
    saved = yaml.safe_load(settings_path.read_text())["Render"]
    assert saved["Colormap"] == "hot"
    assert saved["gpu"]["enabled"] == "auto"
    assert saved["max_blur_width"] == 100.0
    assert saved["interaction_subsample"] == "auto"
    window.close()


def test_render_reports_an_unreadable_settings_file(
    qt_offscreen, settings_path, monkeypatch
):
    settings_path.write_text("Render: [")
    shown = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: shown.append(args)),
    )
    window = gui_render.Window(plugins_loaded=True)
    assert len(shown) == 1
    assert "could not be read" in shown[0][2]
    kept = settings_path.with_name("settings.yaml.broken")
    assert kept.read_text() == "Render: ["
    assert io.settings_load_error() is None  # reported, hence dismissed
    window.close()
