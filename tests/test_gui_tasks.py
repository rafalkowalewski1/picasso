"""Cancelable GUI tasks: ``picasso.lib_qt.run_task``.

A task runs a function on a worker thread behind a progress dialog with a
Cancel button. The function's progress updates are the cancellation
points, results are applied on the GUI thread and user input is blocked
while the task runs.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import threading
import time

import pytest
from PyQt6 import QtCore, QtGui, QtWidgets, sip

from picasso import lib, lib_qt


def _wait_until(qapp, condition, timeout: float = 10.0) -> None:
    start = time.monotonic()
    while not condition():
        qapp.processEvents()
        time.sleep(0.005)
        if time.monotonic() - start > timeout:
            raise TimeoutError("task did not end")


@pytest.fixture
def parent(qt_offscreen):
    widget = QtWidgets.QWidget()
    widget.show()
    yield widget
    widget.close()


def test_result_is_delivered_on_gui_thread(qt_offscreen, parent):
    results = []
    threads = []

    def fn(progress):
        threads.append(threading.current_thread())
        for i in range(10):
            progress.set_value(i + 1)
        return 42

    def on_finished(result):
        results.append((result, threading.current_thread()))

    task = lib.run_task(fn, "Working", parent, on_finished, maximum=10)
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert task.outcome == "finished"
    assert results == [(42, threading.main_thread())]
    assert threads[0] is not threading.main_thread()
    assert lib_qt._running_tasks == []


def test_cancel_stops_at_next_progress_update(qt_offscreen, parent):
    started = threading.Event()
    steps = []
    calls = []

    def fn(progress):
        started.set()
        for i in range(1000):
            steps.append(i)
            progress.set_value(i)
            time.sleep(0.002)
        return "done"

    task = lib.run_task(
        fn,
        "Working",
        parent,
        on_finished=lambda r: calls.append("finished"),
        on_canceled=lambda: calls.append("canceled"),
        maximum=1000,
    )
    started.wait(5)
    # as if the Cancel button was clicked
    task.dialog.canceled.emit()
    assert task.dialog.cancel_was_requested
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert task.outcome == "canceled"
    assert calls == ["canceled"]
    assert len(steps) < 1000


def test_late_result_after_cancel_is_discarded(qt_offscreen, parent):
    """A function without further progress updates still completes; its
    result must not be applied once the user canceled."""
    release = threading.Event()
    calls = []

    def fn(progress):
        release.wait(5)
        return "late"

    task = lib.run_task(
        fn,
        "Working",
        parent,
        on_finished=lambda r: calls.append(r),
        on_canceled=lambda: calls.append("canceled"),
    )
    task.cancel()
    release.set()
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert calls == ["canceled"]


def test_error_goes_to_on_failed(qt_offscreen, parent):
    errors = []

    def fn(progress):
        raise ValueError("bad input")

    task = lib.run_task(fn, "Working", parent, on_failed=errors.append)
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert task.outcome == "failed"
    assert isinstance(errors[0], ValueError)


def test_escape_and_close_cancel_instead_of_hiding(qt_offscreen, parent):
    release = threading.Event()

    def fn(progress):
        while not release.is_set():
            progress.set_value(0)
            time.sleep(0.005)

    task = lib.run_task(fn, "Working", parent, maximum=10)
    task.dialog.show()
    task.dialog.reject()  # Escape
    assert task.progress.canceled
    task.dialog.close()
    assert task.dialog.isVisible()  # only the task closes the dialog
    release.set()
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert task.outcome == "canceled"
    assert sip.isdeleted(task.dialog) or not task.dialog.isVisible()


def _click(button: QtWidgets.QPushButton) -> None:
    """Click through the event path user input takes (unlike
    ``QPushButton.click``, which the input filter cannot see)."""
    center = button.rect().center()
    for kind in (
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QEvent.Type.MouseButtonRelease,
    ):
        event = QtGui.QMouseEvent(
            kind,
            QtCore.QPointF(center),
            QtCore.QPointF(button.mapToGlobal(center)),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(button, event)


def test_input_is_blocked_while_running(qt_offscreen, parent):
    button = QtWidgets.QPushButton("Other action", parent)
    button.show()
    clicks = []
    button.clicked.connect(lambda: clicks.append(1))
    release = threading.Event()

    task = lib.run_task(lambda progress: release.wait(5), "Working", parent)
    _click(button)
    assert clicks == []
    # nor can the window be closed under the task
    parent.close()
    assert parent.isVisible()

    release.set()
    _wait_until(qt_offscreen, lambda: not task.is_running())
    _click(button)
    assert clicks == [1]


def test_phases_drive_one_dialog(qt_offscreen, parent):
    seen = []

    def fn(progress):
        lib.start_phase(progress, "First phase", 4)
        for i in range(4):
            progress.set_value(i + 1)
        lib.start_phase(progress, "Second phase", 2)
        for i in range(2):
            progress.set_value(i + 1)
        lib.start_phase(progress, "Third phase", 0)

    task = lib.run_task(fn, "Start", parent, maximum=0)
    task.progress.phase_changed.connect(seen.append)
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert seen == ["First phase", "Second phase", "Third phase"]


def test_task_progress_works_with_progress_consumers(qt_offscreen, parent):
    """Analysis functions normalize their progress argument; a
    TaskProgress must pass through unchanged."""
    results = []

    def fn(progress):
        assert isinstance(progress, lib.Progress)
        with lib.tracked_progress(progress) as tracker:
            assert tracker is progress
            lib.start_phase(tracker, "Working", 5)
            for i in range(5):
                tracker.set_value(i + 1)
        return progress.value()

    task = lib.run_task(fn, "Working", parent, results.append)
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert results == [5]


def test_callback_can_start_the_next_task(qt_offscreen, parent):
    order = []

    def second(result):
        order.append(result)

    def first(result):
        order.append(result)
        lib.run_task(lambda p: "second", "Second", parent, second)

    lib.run_task(lambda p: "first", "First", parent, first)
    _wait_until(qt_offscreen, lambda: len(order) == 2)
    _wait_until(qt_offscreen, lambda: not lib_qt._running_tasks)
    assert order == ["first", "second"]


def test_progress_dialog_sounds_once_after_all_phases(
    qt_offscreen, parent, monkeypatch
):
    """A function reporting several phases to a plain ProgressDialog
    must not trigger the finish sound when its first phase is done, only
    when the caller closes the dialog."""
    sounds = []
    monkeypatch.setattr(lib_qt.notifications, "notify_finished", sounds.append)
    dialog = lib_qt.ProgressDialog("Working", 0, 0, parent)
    lib.start_phase(dialog, "First phase", 2)
    dialog.set_value(2)
    lib.start_phase(dialog, "Second phase", 3)
    dialog.set_value(3)
    assert sounds == []
    dialog.close()
    assert len(sounds) == 1


def test_progress_dialog_stays_up_between_phases(qt_offscreen, parent):
    """Qt's auto-reset hid a plain ProgressDialog whenever a phase reached
    its maximum, so the next phase of the same function ran without a
    (modal) dialog."""
    dialog = lib_qt.ProgressDialog("Working", 0, 0, parent)
    dialog.show()
    lib.start_phase(dialog, "First phase", 2)
    dialog.set_value(2)
    assert dialog.value() == 2  # not reset to -1
    assert dialog.isVisible()
    lib.start_phase(dialog, "Second phase", 0)
    dialog.set_value(5)  # a busy phase reporting a count
    assert "Estimated time" not in dialog.labelText()
    assert dialog.isVisible()
    dialog.close()
    assert not dialog.isVisible()


# ---------------------------------------------------------------------------
# Play a minigame (picasso.gui.minigame)
# ---------------------------------------------------------------------------


@pytest.fixture
def game_offered():
    """The user switched the minigame on (it is off by default)."""
    from picasso.gui import minigame

    minigame._save_setting(minigame.OFFER_KEY, True)


def _blocked_task(parent, release, steps=None, **kwargs):
    """A task that reports progress until ``release`` is set."""

    def fn(progress):
        lib.start_phase(progress, "Crunching", 1000)
        i = 0
        while not release.is_set():
            i = min(i + 1, 999)
            progress.set_value(i)
            if steps is not None:
                steps.append(i)
            time.sleep(0.002)
        return "done"

    return lib.run_task(fn, "Working", parent, maximum=1000, **kwargs)


def test_play_button_sits_left_of_cancel(qt_offscreen, parent, game_offered):
    release = threading.Event()
    task = _blocked_task(parent, release)
    dialog = task.dialog
    dialog.show()
    qt_offscreen.processEvents()
    play, cancel = dialog._play_button, dialog._cancel_button
    assert play.text() == "Play a minigame"
    assert play.isVisible() and cancel.isVisible()
    assert play.geometry().right() < cancel.geometry().left()
    assert (
        abs(play.geometry().center().y() - cancel.geometry().center().y()) <= 1
    )
    release.set()
    _wait_until(qt_offscreen, lambda: not task.is_running())


def test_play_hands_the_task_to_the_game(qt_offscreen, parent, game_offered):
    """Play replaces the progress dialog by the game, which shows the
    progress, takes input while the task runs, cancels it, and stays
    open, no longer modal, once the task ended."""
    from picasso.gui import minigame

    release = threading.Event()
    calls = []
    task = _blocked_task(
        parent, release, on_canceled=lambda: calls.append("canceled")
    )
    task.dialog.show()
    _wait_until(qt_offscreen, lambda: task.dialog.value() > 0)
    task.dialog.play_requested.emit()  # as Play was clicked
    window = minigame.game_window()
    assert window is not None and window.isVisible()
    assert not task.dialog.isVisible()
    assert window.windowModality() == QtCore.Qt.WindowModality.ApplicationModal
    link = task.game_link
    assert window.link is link
    _wait_until(qt_offscreen, lambda: link.value > 10)
    assert link.label == "Crunching" and link.maximum == 1000
    assert window.strip.bar.value() > 0
    # input reaches the game while the task blocks the other windows
    _click(window.strip.cancel_button)
    assert task.progress.canceled
    release.set()
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert calls == ["canceled"]
    assert link.outcome == "canceled"
    assert window.isVisible()
    assert window.windowModality() == QtCore.Qt.WindowModality.NonModal
    assert window.strip.label.text() == "Canceled: Working"
    window.close()


def test_a_finished_task_leaves_the_game_open(
    qt_offscreen, parent, game_offered
):
    from picasso.gui import minigame

    release = threading.Event()
    results = []
    task = _blocked_task(parent, release, on_finished=results.append)
    task.play()
    window = minigame.game_window()
    window.go_on()
    release.set()
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert results == ["done"]
    assert window.isVisible()
    assert window.strip.label.text() == "Done: Working"
    assert window.game.state == "playing"
    assert not lib_qt.gui_busy()
    window.close()


def test_closing_the_game_brings_the_dialog_back(
    qt_offscreen, parent, game_offered
):
    from picasso.gui import minigame

    release = threading.Event()
    task = _blocked_task(parent, release)
    task.dialog.show()
    task.play()
    assert not task.dialog.isVisible()
    minigame.game_window().close()
    assert task.dialog.isVisible()
    assert task.is_running()  # the task goes on
    # and the dialog offers to play again
    assert task.dialog._play_button.text() == "Play a minigame"
    task.play()
    assert minigame.game_window().link is task.game_link
    release.set()
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert task.outcome == "finished"
    minigame.game_window().close()


def test_canceling_in_the_dialog_hides_play(
    qt_offscreen, parent, game_offered
):
    release = threading.Event()
    task = _blocked_task(parent, release)
    task.dialog.show()
    task.dialog.canceled.emit()
    assert task.dialog._play_button.isHidden()
    assert not task.dialog._cancel_button.isEnabled()
    release.set()
    _wait_until(qt_offscreen, lambda: not task.is_running())


def test_format_time_remaining():
    """Extrapolated from the rate so far (truncated to whole seconds);
    more than 10 hours show as 10."""
    assert lib_qt.format_time_remaining(11.0, 1, 3) == "00m:21s"
    assert lib_qt.format_time_remaining(3601.0, 1, 3) == "02h:00m:01s"
    assert lib_qt.format_time_remaining(3600.0, 1, 100) == "10h:59m:59s"


def test_no_play_button_unless_the_game_is_switched_on(qt_offscreen, parent):
    """The minigame is off by default."""
    release = threading.Event()
    task = _blocked_task(parent, release)
    task.dialog.show()
    qt_offscreen.processEvents()
    assert task.dialog._play_button is None
    assert task.dialog._cancel_button.isVisible()
    task.dialog.canceled.emit()  # cancel works without the button
    release.set()
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert task.outcome == "canceled"
