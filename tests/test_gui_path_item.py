"""Test ``lib_qt.TailElidedComboBox`` and ``lib_qt.get_path_item``, the
input dialog that shows only the end of long file paths, and
``lib_qt.ElidedCheckBox``, which elides long names in the middle.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest
from PyQt6 import QtWidgets

from picasso import lib_qt

LONG_PATH = "/" + "/".join(["a_very_long_folder_name"] * 10) + "/locs.hdf5"


@pytest.fixture
def combo(qt_offscreen):
    combo = lib_qt.TailElidedComboBox()
    yield combo
    combo.deleteLater()


def _chars(combo, n):
    return n * combo.fontMetrics().averageCharWidth()


def test_short_items_are_shown_in_full(combo):
    combo.addItems(["first.hdf5", "second.hdf5"])
    combo.setCurrentIndex(1)
    combo.resize(combo.sizeHint())
    assert combo.displayed_text() == "second.hdf5"
    # no room is reserved for max_chars characters
    assert combo.sizeHint().width() < _chars(combo, combo.max_chars)


def test_long_item_shows_its_end(combo):
    combo.addItems(["short", LONG_PATH])
    combo.setCurrentIndex(1)
    combo.resize(combo.sizeHint())
    shown = combo.displayed_text()
    assert shown.startswith("\u2026")
    assert LONG_PATH.endswith(shown[1:])
    assert len(shown) < len(LONG_PATH)
    path_width = combo.fontMetrics().horizontalAdvance(LONG_PATH)
    assert combo.sizeHint().width() < path_width
    combo.grab()  # paints the end of the path


def test_max_chars_sets_the_width_of_long_items(qt_offscreen):
    widths = []
    for max_chars in (15, 30, 60):
        combo = lib_qt.TailElidedComboBox(max_chars)
        combo.addItems([LONG_PATH])
        widths.append(combo.sizeHint().width())
        # about max_chars average characters plus the arrow
        assert _chars(combo, max_chars) <= widths[-1]
        assert widths[-1] < _chars(combo, max_chars + 10)
        combo.deleteLater()
    assert widths == sorted(widths)


def test_a_wider_box_shows_more_of_the_item(combo):
    combo.addItems([LONG_PATH])
    combo.resize(combo.sizeHint())
    narrow = combo.displayed_text()
    combo.resize(2 * combo.sizeHint().width(), combo.sizeHint().height())
    wide = combo.displayed_text()
    assert len(wide) > len(narrow)
    assert LONG_PATH.endswith(wide[1:])


def test_layout_can_shrink_the_box(combo):
    combo.addItems([LONG_PATH])
    assert combo.minimumSizeHint().width() < combo.sizeHint().width()


def test_popup_lists_the_full_items(combo, monkeypatch):
    combo.addItems([LONG_PATH])
    monkeypatch.setattr(QtWidgets.QComboBox, "showPopup", lambda self: None)
    combo.showPopup()
    path_width = combo.view().fontMetrics().horizontalAdvance(LONG_PATH)
    assert combo.view().minimumWidth() >= path_width
    assert combo.view().model().index(0, 0).data() == LONG_PATH


@pytest.mark.parametrize("accept", [True, False])
def test_get_path_item(qt_offscreen, monkeypatch, accept):
    shown = []

    def exec_(dialog):
        shown.append(dialog)
        assert dialog.windowTitle() == "Choose"
        dialog.combo.setCurrentIndex(1)
        return (
            QtWidgets.QDialog.DialogCode.Accepted
            if accept
            else QtWidgets.QDialog.DialogCode.Rejected
        )

    monkeypatch.setattr(lib_qt.PathItemDialog, "exec", exec_)
    item, ok = lib_qt.get_path_item(
        None, "Choose", "Channel:", ["first", LONG_PATH]
    )
    assert len(shown) == 1
    assert ok is accept
    assert item == (LONG_PATH if accept else "")


def test_check_box_elides_long_names_in_the_middle(qt_offscreen):
    box = lib_qt.ElidedCheckBox("start_" + "x" * 200 + "_end")
    box.setToolTip("Click to show.")
    box.resize(box.minimumSizeHint())
    shown = box.displayed_text()
    assert box.is_elided()
    assert shown.startswith("start") and shown.endswith("_end")
    assert "\u2026" in shown
    assert box.text() == "start_" + "x" * 200 + "_end"
    assert box.minimumSizeHint().width() < box.sizeHint().width()
    policy = box.sizePolicy().horizontalPolicy()
    assert policy == QtWidgets.QSizePolicy.Policy.Preferred
    box.grab()  # paints the elided label
    box.resize(box.sizeHint())
    assert not box.is_elided()
    box.deleteLater()


def test_check_box_with_a_short_name_is_not_shrunk(qt_offscreen):
    box = lib_qt.ElidedCheckBox("short")
    assert box.minimumSizeHint() == box.sizeHint()
    box.resize(box.sizeHint())
    assert box.displayed_text() == "short"
    box.deleteLater()
