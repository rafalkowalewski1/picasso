"""Serpentine, the minigame of ``picasso.gui.minigame``.

The game (``Serpentine``) runs without Qt and is tested with seeded
random numbers; the window, the task bar above it and how a task hands
its progress to the game are tested offscreen.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from collections import deque

import numpy as np

from picasso import io
from picasso.gui import minigame


# ---------------------------------------------------------------------------
# Serpentine
# ---------------------------------------------------------------------------


def _snake(**kwargs) -> minigame.Serpentine:
    game = minigame.Serpentine(rng=np.random.default_rng(0), **kwargs)
    game.start()
    return game


def test_serpentine_moves_one_cell_per_step():
    game = _snake()
    head = game.body[0]
    game.food = (0, 0)  # out of the way
    game.step()
    assert game.body[0] == (head[0] + 1, head[1])
    assert len(game.body) == minigame.SERPENTINE_START_LENGTH
    assert game.state == "playing"


def test_eating_a_dot_grows_scores_and_speeds_up():
    game = _snake()
    x, y = game.body[0]
    interval = game.interval
    game.food = (x + 1, y)
    game.step()
    assert len(game.body) == minigame.SERPENTINE_START_LENGTH + 1
    assert (game.eaten, game.score) == (1, 10)
    assert game.interval < interval
    assert game.food is not None and game.food not in game.body
    # every 5 dots, a dot is worth 10 more
    game.eaten = 5
    x, y = game.body[0]
    game.food = (x + 1, y)
    game.step()
    assert game.score == 10 + 20


def test_turns_are_queued_and_cannot_reverse():
    game = _snake()
    game.turn(minigame.LEFT)  # back onto the line
    game.turn(minigame.RIGHT)  # already heading there
    assert game._turns == []
    game.turn(minigame.UP)
    game.turn(minigame.DOWN)  # back onto the queued turn
    game.turn(minigame.LEFT)
    game.turn(minigame.DOWN)  # at most two are kept
    assert game._turns == [minigame.UP, minigame.LEFT]
    game.food = (0, 0)
    x, y = game.body[0]
    game.step()
    game.step()
    assert game.body[0] == (x - 1, y - 1)


def test_the_wall_ends_the_game():
    game = _snake(best=5)
    game.food = (0, 0)
    columns = game.shape[1]
    while game.state == "playing":
        game.step()
    assert game.body[0][0] == columns - 1  # stopped at the edge
    assert game.state == "game_over"
    assert not game.new_best and game.best == 5


def test_running_into_the_line_ends_the_game_but_not_its_tail():
    game = _snake()
    game.food = (0, 0)
    # head at (5, 5) heading left, the line bending down and back
    game.body = deque([(5, 5), (6, 5), (6, 6), (5, 6)])
    game.direction = minigame.LEFT
    game.turn(minigame.DOWN)  # onto the tail, which moves on
    game.step()
    assert game.state == "playing"
    game.body = deque([(5, 5), (6, 5), (6, 6), (5, 6), (4, 6)])
    game.direction = minigame.LEFT
    game.turn(minigame.DOWN)
    game.step()
    assert game.state == "game_over"


def test_filling_the_field_ends_the_game():
    game = _snake(shape=(1, 6))
    assert list(game.body) == [(3, 0), (2, 0), (1, 0), (0, 0)]
    game.food = (4, 0)
    game.step()
    assert game.food == (5, 0)  # the only free cell
    game.step()
    assert game.filled and game.state == "game_over"
    assert game.new_best and game.best == game.score == 20


def test_a_busy_gui_slows_serpentine_down():
    game = _snake()
    game.food = (0, 0)
    head = game.body[0]
    assert not game.advance(game.interval / 2)
    assert game.advance(5.0)  # one step, however long the pause
    assert game.body[0] == (head[0] + 1, head[1])
    assert not game.advance(0.0)
    game.pause()
    assert not game.advance(5.0)


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def _press(qapp, window, key):
    """A key as the user types it, through the shortcuts."""
    from PyQt6 import QtTest

    QtTest.QTest.keyClick(qapp.focusWidget() or window.board, key)


def _activate(qapp, window):
    """Wait for ``window`` to become the active window: activation is
    delivered by the event loop, and shortcuts only work once it is."""
    for _ in range(200):
        if qapp.activeWindow() is window:
            return
        qapp.processEvents()
    raise AssertionError("the game window did not become active")


def test_serpentine_is_played_with_w_a_s_d(qt_offscreen):
    from PyQt6 import QtCore

    Key = QtCore.Qt.Key
    window = minigame.play()
    _activate(qt_offscreen, window)
    game = window.game
    assert window.focusWidget() is window.board
    assert game.state == "intro"
    _press(qt_offscreen, window, Key.Key_W)  # a steering key starts
    assert game.state == "playing"
    assert game._turns == [minigame.UP]
    _press(qt_offscreen, window, Key.Key_A)
    assert game._turns == [minigame.UP, minigame.LEFT]
    _press(qt_offscreen, window, Key.Key_Space)
    assert game.state == "paused"
    assert "Paused" in window.board.message
    _press(qt_offscreen, window, Key.Key_Space)
    assert game.state == "playing"
    assert window.board.message == []
    _press(qt_offscreen, window, Key.Key_R)  # no new game: R is not the game's
    assert game.state == "playing"
    window.close()


def test_the_arrow_keys_are_not_the_games(qt_offscreen):
    """Localize steps through frames and channels with them."""
    from PyQt6 import QtCore

    window = minigame.play()
    for key in ("Up", "Down", "Left", "Right"):
        _press(qt_offscreen, window, getattr(QtCore.Qt.Key, f"Key_{key}"))
    assert window.game.state == "intro"
    window.go_on()
    _press(qt_offscreen, window, QtCore.Qt.Key.Key_Up)
    assert window.game._turns == []
    window.close()


def test_the_game_keys_win_over_shortcuts(qt_offscreen):
    """A shortcut of Picasso on a key of the game (e.g., a customized
    one) does not take it from the game, while other keys, e.g., R,
    still reach the shortcuts."""
    from PyQt6 import QtCore, QtGui, QtWidgets

    other = QtWidgets.QWidget()
    fired = []
    for key in ("W", "R"):
        action = QtGui.QAction(other)
        action.setShortcut(key)
        action.setShortcutContext(
            QtCore.Qt.ShortcutContext.ApplicationShortcut
        )
        action.triggered.connect(lambda checked, key=key: fired.append(key))
        other.addAction(action)
    other.show()  # shortcuts of hidden widgets never fire
    window = minigame.play()
    _activate(qt_offscreen, window)
    _press(qt_offscreen, window, QtCore.Qt.Key.Key_W)
    assert window.game.state == "playing"
    _press(qt_offscreen, window, QtCore.Qt.Key.Key_R)
    assert fired == ["R"]
    window.close()
    other.close()


def test_a_click_starts_and_resumes(qt_offscreen):
    window = minigame.GameWindow(rng=np.random.default_rng(1))
    window.board.clicked.emit()
    assert window.game.state == "playing"
    window.pause()
    window.board.clicked.emit()
    assert window.game.state == "playing"
    window.close()


def test_the_window_plays_on_with_its_timer(qt_offscreen):
    window = minigame.GameWindow(rng=np.random.default_rng(7))
    game = window.game
    game.start()
    assert game.body[0] == (6, 8)
    game.food = (0, 0)
    window._last_tick -= 1.0
    window.tick()
    assert game.body[0] == (7, 8)  # one cell on, however long the wait
    assert "Length 4" in window.hud.text()
    window.close()


def test_a_new_best_score_is_saved(qt_offscreen):
    assert minigame._load_best() == 0
    window = minigame.GameWindow(rng=np.random.default_rng(8))
    game = window.game
    game.start()
    game.score = 70
    x, y = game.body[0]
    game.body = deque([(game.shape[1] - 1, y)] + list(game.body)[1:])
    game.food = (0, 0)
    window._last_tick -= 1.0
    window.tick()  # into the wall
    assert game.state == "game_over"
    assert "New best score!" in window.board.message
    assert minigame._load_best() == 70
    settings = io.load_user_settings()[minigame.SETTINGS_KEY]
    assert settings[minigame.BEST_KEY] == 70
    assert "Best 70" in window.hud.text()
    window.close()
    assert minigame.GameWindow().game.best == 70


def test_a_broken_best_score_setting_reads_as_none(qt_offscreen):
    settings = io.load_user_settings()
    settings[minigame.SETTINGS_KEY] = "garbage"
    io.save_user_settings(settings)
    assert minigame._load_best() == 0
    minigame._save_setting(minigame.BEST_KEY, "a lot")
    assert minigame._load_best() == 0
    minigame._save_best(10)
    assert minigame._load_best() == 10


# ---------------------------------------------------------------------------
# The task the player waits for
# ---------------------------------------------------------------------------


def test_task_link_reports_progress_and_ends_once(qapp):
    canceled = []
    link = minigame.TaskLink("Clustering", cancel=lambda: canceled.append(1))
    ended = []
    link.ended.connect(ended.append)
    link.start_phase("Finding neighbors", 100)
    link.set_progress(10)
    assert link.time_remaining() == ""  # no rate yet
    link.set_progress(50)
    assert link.time_remaining().endswith("s")
    link.set_maximum(200)  # a new length restarts the estimate
    assert link.time_remaining() == ""
    link.set_progress(60)
    assert link.time_remaining().endswith("s")
    link.cancel()
    link.cancel()
    assert canceled == [1] and link.canceling
    link.end("canceled")
    link.end("finished")
    assert ended == ["canceled"]
    assert link.message == "Clustering"
    assert not link.running


def test_a_task_without_cancel_cannot_be_canceled(qapp):
    link = minigame.TaskLink("Fitting")
    link.set_cancelable(True)
    assert not link.cancelable
    link.cancel()
    assert not link.canceling


def test_the_strip_shows_progress_then_how_the_task_ended(qt_offscreen):
    window = minigame.GameWindow(rng=np.random.default_rng(0))
    canceled = []
    link = minigame.TaskLink(
        "Clustering", window=None, cancel=lambda: canceled.append(1)
    )
    link.start_phase("Finding neighbors", 10)
    link.set_progress(4)
    window.attach(link)
    strip = window.strip
    assert not strip.isHidden()
    assert strip.label.text() == "Finding neighbors"
    assert (strip.bar.value(), strip.bar.maximum()) == (4, 10)
    assert strip.cancel_button.isVisibleTo(window)
    strip.cancel_button.click()
    assert canceled == [1]
    assert strip.label.text() == "Canceling..."
    assert not strip.cancel_button.isEnabled()
    link.end("canceled")
    assert strip.label.text() == "Canceled: Clustering"
    assert not strip.cancel_button.isVisibleTo(window)
    assert strip.back_button.isVisibleTo(window)
    strip.dismiss_button.click()
    assert strip.isHidden()


def test_the_game_is_modal_until_the_task_ended(qt_offscreen):
    """While a ``run_task`` task runs, the game takes the place of its
    application-modal progress dialog; it stays open once the task is
    done, but no longer blocks the other windows."""
    from PyQt6 import QtCore

    link = minigame.TaskLink("Clustering")
    window = minigame.play(link, modal=True)
    try:
        assert minigame.game_window() is window
        assert window.isVisible()
        assert (
            window.windowModality()
            == QtCore.Qt.WindowModality.ApplicationModal
        )
        window.go_on()
        link.end("finished")
        assert window.isVisible()
        assert window.windowModality() == QtCore.Qt.WindowModality.NonModal
        # the swap did not pause it
        assert window.game.state == "playing"
    finally:
        window.close()
    assert minigame.game_window() is None


def test_space_plays_and_never_cancels_the_task(qt_offscreen):
    """The keyboard belongs to the game, also with a task above it: the
    task bar's Cancel had the focus, so Space canceled the task instead
    of starting the game."""
    from PyQt6 import QtCore, QtTest

    canceled = []
    link = minigame.TaskLink("Clustering", cancel=lambda: canceled.append(1))
    window = minigame.play(link)
    assert window.focusWidget() is window.board

    def press_space():
        target = qt_offscreen.focusWidget() or window.board
        QtTest.QTest.keyClick(target, QtCore.Qt.Key.Key_Space)

    press_space()
    assert window.game.state == "playing"
    press_space()
    assert window.game.state == "paused"
    assert canceled == []
    # a click on a button does not take the keyboard from the game
    QtTest.QTest.mouseClick(
        window.strip.cancel_button, QtCore.Qt.MouseButton.LeftButton
    )
    assert canceled == [1]
    assert window.focusWidget() is window.board
    window.close()


def test_closing_the_game_detaches_a_running_task(qt_offscreen):
    link = minigame.TaskLink("Clustering")
    detached = []
    link.detached.connect(lambda: detached.append(1))
    window = minigame.play(link)
    assert minigame.play_button_text() == "Continue playing"
    window.close()
    assert detached == [1]
    assert minigame.play_button_text() == "Play a minigame"


def test_a_new_task_replaces_the_one_shown(qt_offscreen):
    first = minigame.TaskLink("Identifying")
    second = minigame.TaskLink("Fitting")
    detached = []
    first.detached.connect(lambda: detached.append("first"))
    window = minigame.play(first)
    minigame.follow(second)
    assert window.link is second
    assert window.strip.label.text() == "Fitting"
    assert detached == ["first"]
    window.close()
    minigame.follow(first)  # no game open: nothing happens
    assert minigame.game_window() is None


# ---------------------------------------------------------------------------
# Opting in (in the user settings)
# ---------------------------------------------------------------------------


def _saved_offer():
    return io.load_user_settings()[minigame.SETTINGS_KEY][minigame.OFFER_KEY]


def test_the_game_is_offered_only_once_switched_on(qapp):
    assert not minigame.offered()
    assert _saved_offer() is False  # the default is written, as all are
    minigame._save_setting(minigame.OFFER_KEY, True)  # as the user does
    assert minigame.offered()


def test_an_invalid_offer_setting_leaves_the_game_off(qapp):
    minigame._save_setting(minigame.OFFER_KEY, "yes please")
    assert not minigame.offered()
    assert _saved_offer() is False
