"""
picasso.gui.minigame
~~~~~~~~~~~~~~~~~~~~

Serpentine, a minigame to pass the time while Picasso works.

Fitting a movie in Localize or clustering in Render can take
minutes. Users who set ``OFFER_KEY`` to True in the ``SETTINGS_KEY``
section of the user settings (it is False by default) are offered
"Play a minigame" while such a task runs: in the progress dialog of
every task run with ``picasso.lib_qt.run_task`` and in the task bar
of Localize's status bar.

How to play: steer a line with W, A, S and D to the dots; each dot
makes it longer and faster. The game ends when the line runs into a
wall or into itself; Space starts the next one. Space also pauses (as
does switching to another window); the best score is kept in the user
settings. The arrow keys are left alone, as Localize steps through
frames and channels with them. When Picasso is busy, the line slows
down rather than jumping ahead.

The task the player waits for is shown above the game, with its
progress and a Cancel button. The game never closes itself: when the
task ends, the bar above the game says so and the player stops
whenever they like. While a ``run_task`` task runs, the game window
is application-modal like the progress dialog it replaces, so
Picasso's windows stay blocked until the task is done.

For developers: whatever runs the task describes it with a
``TaskLink`` and passes it to ``play`` (or ``follow``, to show it in
a game that is already open). The game itself (``Serpentine``) does
not depend on Qt.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets, sip

from .. import io, lib_qt

#: Longest time one step of the game may take in s, so that a busy GUI
#: slows the game down instead of moving on without the player.
MAX_STEP = 0.2
#: Section of the game in the user settings.
SETTINGS_KEY = "Minigame"
#: Key of the opt-in in ``SETTINGS_KEY``: whether long tasks offer the
#: game (False by default).
OFFER_KEY = "Offer during long tasks"
#: Key of the best score in ``SETTINGS_KEY``.
BEST_KEY = "Best score"


# ---------------------------------------------------------------------------
# Serpentine, without Qt
# ---------------------------------------------------------------------------

#: Serpentine's field in cells, (rows, columns).
SERPENTINE_SHAPE = (16, 24)
#: Length of the line at the start, in cells.
SERPENTINE_START_LENGTH = 4
#: Time per step at the start, in s.
SERPENTINE_START_INTERVAL = 0.15
#: Time per step at the highest speed, in s.
SERPENTINE_MIN_INTERVAL = 0.06
#: Each dot eaten shortens the time per step by this much, in s.
SERPENTINE_SPEEDUP = 0.003
#: The directions of the line, (dx, dy) in cells; y points down.
UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)


class Serpentine:
    """Serpentine, a mini-game, without Qt.

    A line moves across a field of cells, one cell per step, and the
    player steers it (``turn``) to the dots. Each dot eaten makes the
    line one cell longer and the steps faster, and scores 10 points
    times the level (one more every 5 dots). The game ends when the
    head leaves the field or runs into the line (or the line fills the
    field). ``state`` is "intro", "playing", "paused" or "game_over";
    ``advance`` moves the game on while playing.

    Parameters
    ----------
    best : int, optional
        Best score so far. Default 0.
    rng : np.random.Generator, optional
        Random numbers for the dots; a new generator if None (default).
    shape : tuple of int, optional
        The field (rows, columns). Default ``SERPENTINE_SHAPE``.
    """

    def __init__(
        self,
        best: int = 0,
        rng: np.random.Generator | None = None,
        shape: tuple[int, int] = SERPENTINE_SHAPE,
    ) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.shape = shape
        self.best = int(best)
        self.state = "intro"
        self._reset()

    def _reset(self) -> None:
        rows, columns = self.shape
        x, y = max(columns // 4, SERPENTINE_START_LENGTH - 1), rows // 2
        # head first, heading right
        self.body = deque((x - i, y) for i in range(SERPENTINE_START_LENGTH))
        self.direction = RIGHT
        self._turns: list[tuple[int, int]] = []
        self.score = 0
        self.eaten = 0
        self.new_best = False
        self.filled = False
        self._clock = 0.0
        self.food = self._place_food()

    @property
    def level(self) -> int:
        """Level of the game, one more every 5 dots eaten."""
        return 1 + self.eaten // 5

    @property
    def interval(self) -> float:
        """Time per step in s, shorter with each dot eaten."""
        return max(
            SERPENTINE_MIN_INTERVAL,
            SERPENTINE_START_INTERVAL - SERPENTINE_SPEEDUP * self.eaten,
        )

    def start(self) -> None:
        """Start a new game."""
        self._reset()
        self.state = "playing"

    def pause(self) -> None:
        """Pause a game that is being played."""
        if self.state == "playing":
            self.state = "paused"

    def resume(self) -> None:
        """Resume a paused game."""
        if self.state == "paused":
            self.state = "playing"

    def turn(self, direction: tuple[int, int]) -> None:
        """Steer the line in ``direction`` from the next step on.

        Up to two turns are kept for the next steps, so that a quick
        turn around a corner is not lost; a turn back onto the line
        itself is ignored.

        Parameters
        ----------
        direction : tuple of int
            ``UP``, ``DOWN``, ``LEFT`` or ``RIGHT``.
        """
        if self.state != "playing":
            return
        last = self._turns[-1] if self._turns else self.direction
        if direction in (last, (-last[0], -last[1])):
            return
        if len(self._turns) < 2:
            self._turns.append(direction)

    def advance(self, dt: float) -> bool:
        """Play on for ``dt`` s (at most ``MAX_STEP``), at most one step
        per call: a busy GUI slows the game down instead of moving the
        line several cells without the player.

        Parameters
        ----------
        dt : float
            Time since the last call in s.

        Returns
        -------
        stepped : bool
            Whether the line moved.
        """
        if self.state != "playing":
            return False
        self._clock += min(max(dt, 0.0), MAX_STEP)
        interval = self.interval
        if self._clock < interval:
            return False
        self._clock = min(self._clock - interval, interval)
        self.step()
        return True

    def step(self) -> None:
        """Move the line by one cell, eating a dot in its way."""
        if self._turns:
            self.direction = self._turns.pop(0)
        x, y = self.body[0]
        head = (x + self.direction[0], y + self.direction[1])
        rows, columns = self.shape
        eats = head == self.food
        # the tail moves on, unless the line grows
        body = list(self.body) if eats else list(self.body)[:-1]
        inside = 0 <= head[0] < columns and 0 <= head[1] < rows
        if not inside or head in set(body):
            self._end()
            return
        self.body.appendleft(head)
        if not eats:
            self.body.pop()
            return
        self.score += 10 * self.level
        self.eaten += 1
        self.food = self._place_food()
        if self.food is None:
            self.filled = True
            self._end()

    def _place_food(self) -> tuple[int, int] | None:
        rows, columns = self.shape
        occupied = set(self.body)
        free = [
            (x, y)
            for y in range(rows)
            for x in range(columns)
            if (x, y) not in occupied
        ]
        if not free:
            return None
        return free[int(self.rng.integers(len(free)))]

    def _end(self) -> None:
        self.state = "game_over"
        self.new_best = self.score > self.best
        self.best = max(self.best, self.score)


# ---------------------------------------------------------------------------
# The task the player waits for
# ---------------------------------------------------------------------------


class TaskLink(QtCore.QObject):
    """A task as the game shows it, driven by whatever runs the task.

    The owner of the task reports its progress (``set_progress``,
    ``start_phase``, ``set_label``) and its end (``end``); the game
    shows them and may ask to ``cancel``. ``detached`` tells the owner
    that the game no longer shows the running task (its window was
    closed, or it shows another task), e.g., to show its own progress
    display again.

    Parameters
    ----------
    description : str
        What the task does, e.g., "Clustering".
    window : QtWidgets.QWidget, optional
        The window the task belongs to, raised by "Back to Picasso".
        Default None.
    cancel : callable, optional
        Asks the task to stop. None (default) if it cannot be canceled.
    """

    #: The progress, label or cancelability changed.
    changed = QtCore.pyqtSignal()
    #: The task ended: "finished", "canceled" or "failed".
    ended = QtCore.pyqtSignal(str)
    #: The game stopped showing the task while it was running.
    detached = QtCore.pyqtSignal()

    def __init__(
        self,
        description: str,
        window: QtWidgets.QWidget | None = None,
        cancel: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.description = description
        self.label = description
        self.value = 0
        self.maximum = 0
        self.outcome: str | None = None
        self.message = ""
        self.window = window
        self._cancel = cancel
        self.cancelable = cancel is not None
        self.canceling = False
        # when the time estimate started, and the progress at that time
        self._t0: float | None = None
        self._v0 = 0

    @property
    def running(self) -> bool:
        """Whether the task has not ended yet."""
        return self.outcome is None

    def set_label(self, text: str) -> None:
        """Show ``text`` as what the task is doing now."""
        self.label = text
        self.changed.emit()

    def start_phase(self, label: str, maximum: int | None = None) -> None:
        """Start a new phase of the task at progress 0.

        Parameters
        ----------
        label : str
            What the phase does.
        maximum : int, optional
            Number of items of the phase; 0 if unknown. None (default)
            keeps the current one.
        """
        self.label = label
        self.value = 0
        if maximum is not None:
            self.maximum = int(maximum)
        self._t0 = None
        self.changed.emit()

    def set_maximum(self, maximum: int) -> None:
        """Change the number of items of the current phase (0 if
        unknown), which restarts the time estimate."""
        self.set_progress(self.value, maximum)

    def set_progress(self, value: int, maximum: int | None = None) -> None:
        """Report the progress of the current phase.

        Parameters
        ----------
        value : int
            Items finished.
        maximum : int, optional
            Number of items; 0 if unknown. None (default) keeps the
            current one. A new maximum restarts the time estimate.
        """
        if maximum is not None and int(maximum) != self.maximum:
            self.maximum = int(maximum)
            self._t0 = None
        self.value = int(value)
        if self.value > 0 and self._t0 is None:
            self._t0 = time.monotonic()
            self._v0 = self.value
        self.changed.emit()

    def set_cancelable(self, cancelable: bool) -> None:
        """Whether ``cancel`` can stop the task now."""
        cancelable = bool(cancelable) and self._cancel is not None
        if cancelable != self.cancelable:
            self.cancelable = cancelable
            self.changed.emit()

    def cancel(self) -> None:
        """Ask the task to stop, if it can be."""
        if not (self.running and self.cancelable) or self.canceling:
            return
        self.canceling = True
        self.changed.emit()
        self._cancel()

    def time_remaining(self) -> str:
        """Estimated time until the current phase is done, e.g.,
        "03m:12s", or "" while it cannot be estimated. The rate is
        measured from the first progress of the phase on."""
        done = self.value - self._v0
        if self._t0 is None or self.maximum <= 0 or done <= 0:
            return ""
        return lib_qt.format_time_remaining(
            time.monotonic() - self._t0, done, self.maximum - self._v0
        )

    def end(self, outcome: str, message: str | None = None) -> None:
        """Report that the task ended; only the first call counts.

        Parameters
        ----------
        outcome : {"finished", "canceled", "failed"}
            How the task ended.
        message : str, optional
            What to tell the player, e.g., the result. Default: the
            description.
        """
        if not self.running:
            return
        self.outcome = outcome
        self.message = message or self.description
        self.ended.emit(outcome)

    def raise_window(self) -> None:
        """Bring the window of the task to the front."""
        window = self.window
        if window is None or sip.isdeleted(window):
            return
        if window.isMinimized():
            window.showNormal()
        window.raise_()
        window.activateWindow()


class TaskStrip(QtWidgets.QFrame):
    """The task the player waits for, above the game: what it does, its
    progress, the time left and Cancel; once it ended, how, with "Back
    to Picasso" and "Dismiss"."""

    _OUTCOMES = {
        "finished": ("state-ok", "Done"),
        "canceled": ("state-warning", "Canceled"),
        "failed": ("state-danger", "Failed"),
    }

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.icon = QtWidgets.QLabel()
        self.label = QtWidgets.QLabel()
        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.label.setMinimumWidth(80)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedWidth(160)
        self.remaining = QtWidgets.QLabel()
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel)
        self.back_button = QtWidgets.QPushButton("Back to Picasso")
        self.back_button.clicked.connect(self._back)
        self.dismiss_button = QtWidgets.QPushButton("Dismiss")
        self.dismiss_button.clicked.connect(self.hide)
        # reached with Tab, but a click leaves the keyboard to the game:
        # Space on a focused Cancel would cancel the task
        for button in (
            self.cancel_button,
            self.back_button,
            self.dismiss_button,
        ):
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.TabFocus)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.addWidget(self.icon)
        layout.addWidget(self.label, 1)
        for widget in (
            self.bar,
            self.remaining,
            self.cancel_button,
            self.back_button,
            self.dismiss_button,
        ):
            layout.addWidget(widget)
        self.link: TaskLink | None = None
        self.hide()

    def attach(self, link: TaskLink) -> None:
        """Show ``link`` instead of the task shown so far."""
        self.detach()
        self.link = link
        link.changed.connect(self.refresh)
        link.ended.connect(self.refresh)
        self.refresh()
        self.show()

    def detach(self) -> None:
        """Stop showing the current task."""
        if self.link is None:
            return
        for signal in (self.link.changed, self.link.ended):
            try:
                signal.disconnect(self.refresh)
            except TypeError:
                pass
        self.link = None
        self.hide()

    def refresh(self, *args) -> None:
        """Show the current state of the task."""
        link = self.link
        if link is None:
            return
        running = link.running
        if running:
            self.icon.hide()
            text = (link.label or link.description).splitlines()[0]
            if link.canceling:
                text = "Canceling..."
            maximum = max(link.maximum, 0)
            self.bar.setRange(0, maximum)
            if maximum:
                self.bar.setValue(min(link.value, maximum))
            remaining = link.time_remaining()
            self.remaining.setText(f"{remaining} left" if remaining else "")
            self.cancel_button.setEnabled(
                link.cancelable and not link.canceling
            )
        else:
            from . import theme

            icon, title = self._OUTCOMES.get(
                link.outcome, self._OUTCOMES["finished"]
            )
            size = self.label.fontMetrics().height()
            self.icon.setPixmap(theme.icon(icon).pixmap(size, size))
            self.icon.show()
            text = f"{title}: {link.message.splitlines()[0]}"
        self.label.setText(text)
        self.label.setToolTip(text)
        for widget in (self.bar, self.remaining, self.cancel_button):
            widget.setVisible(running)
        for widget in (self.back_button, self.dismiss_button):
            widget.setVisible(not running)
        self.back_button.setEnabled(
            link.window is not None and not sip.isdeleted(link.window)
        )

    def _cancel(self) -> None:
        if self.link is not None:
            self.link.cancel()

    def _back(self) -> None:
        if self.link is not None:
            self.link.raise_window()


# ---------------------------------------------------------------------------
# The game window
# ---------------------------------------------------------------------------


def _paint_message(
    painter: QtGui.QPainter, widget: QtWidgets.QWidget, lines: list[str]
) -> None:
    """Paint ``lines`` on a dark panel in the center of ``widget``, the
    first line bold."""
    regular = QtGui.QFont(widget.font())
    bold = QtGui.QFont(regular)
    bold.setBold(True)
    bold.setPointSizeF(regular.pointSizeF() * 1.3)
    fonts = [bold] + [regular] * (len(lines) - 1)
    metrics = [QtGui.QFontMetricsF(font) for font in fonts]
    width = max(m.horizontalAdvance(line) for m, line in zip(metrics, lines))
    height = sum(m.lineSpacing() for m in metrics)
    pad = 14
    width = min(width + 2 * pad, widget.width() - 16)
    height += 2 * pad
    x = (widget.width() - width) / 2
    y = (widget.height() - height) / 2
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(0, 0, 0, 190))
    painter.drawRoundedRect(QtCore.QRectF(x, y, width, height), 8, 8)
    painter.setPen(QtGui.QColor(255, 255, 255))
    top = y + pad
    for font, m, line in zip(fonts, metrics, lines):
        painter.setFont(font)
        painter.drawText(
            QtCore.QRectF(x + pad, top, width - 2 * pad, m.lineSpacing()),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            line,
        )
        top += m.lineSpacing()


class SerpentineBoard(QtWidgets.QWidget):
    """Serpentine's field: the line, the dot and the game's messages.

    It takes the keyboard; the keys of the game (``GameWindow.KEYS``)
    go to the game even if a menu has a shortcut on them.
    """

    #: A click; the window goes on (start or resume a game).
    clicked = QtCore.pyqtSignal()

    def __init__(self, game: Serpentine, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.game = game
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.message: list[str] = []

    def sizeHint(self) -> QtCore.QSize:
        rows, columns = self.game.shape
        return QtCore.QSize(columns * 28, rows * 28)

    def event(self, event: QtCore.QEvent) -> bool:
        # claim the game's keys before any shortcut does (on macOS, also
        # the menus of other windows), so that they reach keyPressEvent
        if (
            event.type() == QtCore.QEvent.Type.ShortcutOverride
            and event.key() in GameWindow.KEYS
            and not event.modifiers()
        ):
            event.accept()
            return True
        return super().event(event)

    def field_rect(self) -> QtCore.QRectF:
        """Where the field is drawn: as large as fits, centered."""
        rows, columns = self.game.shape
        scale = min((self.width() - 8) / columns, (self.height() - 8) / rows)
        width, height = columns * scale, rows * scale
        return QtCore.QRectF(
            (self.width() - width) / 2,
            (self.height() - height) / 2,
            width,
            height,
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        game = self.game
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(14, 17, 22))
        rect = self.field_rect()
        cell = rect.width() / game.shape[1]
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(26, 31, 40))
        painter.drawRoundedRect(rect, cell / 2, cell / 2)
        # a faint dot in every cell
        painter.setBrush(QtGui.QColor(255, 255, 255, 18))
        rows, columns = game.shape
        for y in range(rows):
            for x in range(columns):
                painter.drawEllipse(
                    self._center(rect, cell, (x, y)), cell * 0.06, cell * 0.06
                )

        def center(xy):
            return self._center(rect, cell, xy)

        if game.food is not None:
            food = center(game.food)
            glow = QtGui.QRadialGradient(food, cell * 0.9)
            glow.setColorAt(0.0, QtGui.QColor(255, 176, 64, 150))
            glow.setColorAt(1.0, QtGui.QColor(255, 176, 64, 0))
            painter.setBrush(glow)
            painter.drawEllipse(food, cell * 0.9, cell * 0.9)
            painter.setBrush(QtGui.QColor(255, 196, 96))
            painter.drawEllipse(food, cell * 0.26, cell * 0.26)

        accent = QtGui.QColor(
            self.palette().color(QtGui.QPalette.ColorRole.Highlight)
        )
        body = list(game.body)
        path = QtGui.QPainterPath(center(body[0]))
        for xy in body[1:]:
            path.lineTo(center(xy))
        pen = QtGui.QPen(accent, cell * 0.62)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        # the head, a little wider and lighter, looking ahead
        head = center(body[0])
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(accent.lighter(130))
        painter.drawEllipse(head, cell * 0.4, cell * 0.4)
        dx, dy = game.direction
        painter.setBrush(QtGui.QColor(255, 255, 255))
        for side in (1, -1):
            eye = QtCore.QPointF(
                head.x() + cell * (0.15 * dx - 0.17 * dy * side),
                head.y() + cell * (0.15 * dy + 0.17 * dx * side),
            )
            painter.drawEllipse(eye, cell * 0.08, cell * 0.08)
        if self.message:
            _paint_message(painter, self, self.message)

    @staticmethod
    def _center(rect, cell, xy) -> QtCore.QPointF:
        return QtCore.QPointF(
            rect.x() + (xy[0] + 0.5) * cell, rect.y() + (xy[1] + 0.5) * cell
        )


class GameWindow(QtWidgets.QWidget):
    """The window of Serpentine: the task strip on top, the field and
    the score. Open it with ``play``.

    Parameters
    ----------
    best : int, optional
        Best score so far. None (default) reads it from the user
        settings.
    rng : np.random.Generator, optional
        Random numbers of the game. Default None, a new generator.
    """

    #: Interval of the game's timer in ms.
    TICK = 30
    #: The keys that steer the line; not the arrow keys, with which
    #: Localize steps through frames and channels.
    STEER = {
        QtCore.Qt.Key.Key_W: UP,
        QtCore.Qt.Key.Key_A: LEFT,
        QtCore.Qt.Key.Key_S: DOWN,
        QtCore.Qt.Key.Key_D: RIGHT,
    }
    #: All keys of the game.
    KEYS = frozenset(
        [
            *STEER,
            QtCore.Qt.Key.Key_Space,
            QtCore.Qt.Key.Key_Escape,
        ]
    )

    def __init__(
        self,
        best: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(None, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("Serpentine")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        # closing the last Picasso window quits even while playing
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.game = Serpentine(
            best=_load_best() if best is None else best, rng=rng
        )
        self.link: TaskLink | None = None
        self._swapping = False

        self.strip = TaskStrip(self)
        self.board = SerpentineBoard(self.game, self)
        self.board.clicked.connect(self._on_click)
        self.hud = QtWidgets.QLabel()
        hint = QtWidgets.QLabel("W, A, S, D: steer. Space: pause.")
        hint.setEnabled(False)  # dimmed
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.hud, 1)
        bottom.addWidget(hint)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.strip)
        layout.addWidget(self.board, 1)
        layout.addLayout(bottom)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.TICK)
        self.timer.timeout.connect(self.tick)
        self._last_tick = time.monotonic()
        self.timer.start()
        self.refresh()

    # -- the game --------------------------------------------------------

    def tick(self) -> None:
        """Play on for the time since the last tick."""
        now = time.monotonic()
        dt, self._last_tick = now - self._last_tick, now
        game = self.game
        if not game.advance(dt):
            return
        if game.state == "game_over":
            if game.new_best:
                _save_best(game.best)
            self.refresh()
        else:
            self.board.update()
            self._update_hud()

    def handle_key(self, event: QtGui.QKeyEvent) -> bool:
        """W, A, S, D: steer (and start); Space: start, pause or go on;
        Escape: pause. Returns whether the key was used."""
        key = event.key()
        game = self.game
        if key in self.STEER:
            if game.state != "playing":
                self.go_on()
            game.turn(self.STEER[key])
        elif key == QtCore.Qt.Key.Key_Space:
            if game.state == "playing":
                self.pause()
            else:
                self.go_on()
        elif key == QtCore.Qt.Key.Key_Escape:
            self.pause()
        else:
            return False
        return True

    def _on_click(self) -> None:
        if self.game.state != "playing":
            self.go_on()

    def go_on(self) -> None:
        """Start a game, or resume a paused one."""
        game = self.game
        if game.state == "paused":
            game.resume()
        elif game.state in ("intro", "game_over"):
            game.start()
        self.refresh()

    def pause(self) -> None:
        """Pause the game being played."""
        if self.game.state == "playing":
            self.game.pause()
            self.refresh()

    def refresh(self) -> None:
        """Show the state of the game: field, messages, score."""
        game = self.game
        state = game.state
        if state == "intro":
            message = [
                "Serpentine",
                "Steer the line with W, A, S and D.",
                "Each dot makes it longer and faster.",
                "Don't run into the walls or into yourself.",
                "Press Space or a steering key to start.",
            ]
        elif state == "paused":
            message = ["Paused", "Press Space to go on."]
        elif state == "game_over":
            title = "The field is full!" if game.filled else "Game over"
            best = (
                "New best score!" if game.new_best else f"Best: {game.best:,}"
            )
            message = [
                f"{title} {game.score:,} points",
                best,
                "Press Space to play again.",
            ]
        else:
            message = []
        self.board.message = message
        self.board.update()
        self._update_hud()

    def _update_hud(self) -> None:
        game = self.game
        parts = [
            f"Length {len(game.body)}",
            f"Level {game.level}",
            f"Score {game.score:,}",
            f"Best {game.best:,}",
        ]
        self.hud.setText("    ".join(parts))

    # -- the task --------------------------------------------------------

    def attach(self, link: TaskLink, modal: bool = False) -> None:
        """Show ``link`` above the game.

        Parameters
        ----------
        link : TaskLink
            The task.
        modal : bool, optional
            Whether the window blocks the rest of the application until
            the task ended, as a ``run_task`` progress dialog does.
            Default False.
        """
        if link is not self.link:
            self._release_link()
            self.link = link
            link.ended.connect(self._on_link_ended)
            self.strip.attach(link)
        self._set_modal(modal and link.running)

    def _release_link(self) -> None:
        link, self.link = self.link, None
        if link is None:
            return
        try:
            link.ended.disconnect(self._on_link_ended)
        except TypeError:
            pass
        self.strip.detach()
        if link.running:
            link.detached.emit()

    def _on_link_ended(self, outcome: str) -> None:
        self._set_modal(False)

    def _set_modal(self, modal: bool) -> None:
        """Make the window (not) application-modal. A visible window has
        to be hidden and shown again for that, which must not pause the
        game."""
        modality = (
            QtCore.Qt.WindowModality.ApplicationModal
            if modal
            else QtCore.Qt.WindowModality.NonModal
        )
        if self.windowModality() == modality:
            return
        if not self.isVisible():
            self.setWindowModality(modality)
            return
        geometry = self.geometry()
        self._swapping = True
        self.hide()
        self.setWindowModality(modality)
        self.setGeometry(geometry)
        self.show_game()
        QtCore.QTimer.singleShot(500, self._swapped)

    def _swapped(self) -> None:
        self._swapping = False

    def show_game(self) -> None:
        """Show the window in front, with the keyboard on the game."""
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()
        self.board.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)

    # -- Qt events -------------------------------------------------------

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if not self.handle_key(event):
            super().keyPressEvent(event)

    def changeEvent(self, event: QtCore.QEvent) -> None:
        kind = event.type()
        if (
            kind == QtCore.QEvent.Type.ActivationChange
            and not self.isActiveWindow()
            and not self._swapping
        ) or (
            kind == QtCore.QEvent.Type.WindowStateChange and self.isMinimized()
        ):
            self.pause()
        super().changeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        global _window
        self.timer.stop()
        self._release_link()
        if _window is self:
            _window = None
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Opening the game
# ---------------------------------------------------------------------------

_window: GameWindow | None = None


def game_window() -> GameWindow | None:
    """The open game window, or None."""
    return _window


def play(link: TaskLink | None = None, *, modal: bool = False) -> GameWindow:
    """Open the game (or bring it to the front) and show a task above
    it.

    Parameters
    ----------
    link : TaskLink, optional
        The task the player waits for. None (default) keeps the one
        shown, if any.
    modal : bool, optional
        Whether the game blocks the rest of the application until the
        task ended, see ``GameWindow.attach``. Default False.

    Returns
    -------
    window : GameWindow
        The game window.
    """
    global _window
    if _window is None:
        _window = GameWindow()
    if link is not None:
        _window.attach(link, modal=modal)
    _window.show_game()
    return _window


def follow(link: TaskLink) -> None:
    """Show ``link`` in the game if it is open, e.g., the next task
    after the one the player waited for."""
    if _window is not None:
        _window.attach(link)


def play_button_text() -> str:
    """Text of the buttons that open the game."""
    return "Continue playing" if _window is not None else "Play a minigame"


def play_icon() -> QtGui.QIcon:
    """Icon of the buttons that open the game."""
    from . import theme

    return theme.icon("minigame")


def _load_setting(key: str):
    """The value of ``key`` in the games' section of the user settings;
    None if there is none."""
    section = io.load_user_settings().get(SETTINGS_KEY)
    return section.get(key) if isinstance(section, dict) else None


def _save_setting(key: str, value) -> None:
    """Save ``value`` under ``key`` in the games' section of the user
    settings (replacing a section that is no mapping)."""
    settings = io.load_user_settings()
    if not isinstance(settings.get(SETTINGS_KEY), dict):
        settings[SETTINGS_KEY] = {}
    settings[SETTINGS_KEY][key] = value
    io.save_user_settings(settings)


def _load_best() -> int:
    """The best score from the user settings; 0 if there is none."""
    try:
        return max(int(_load_setting(BEST_KEY)), 0)
    except (TypeError, ValueError):
        return 0


def _save_best(best: int) -> None:
    """Save the best score in the user settings."""
    _save_setting(BEST_KEY, int(best))


# ---------------------------------------------------------------------------
# Opting in
# ---------------------------------------------------------------------------


def offered() -> bool:
    """Whether long tasks offer the game ("Play a minigame"): the setting
    ``OFFER_KEY``, which the user sets to True to opt in. A missing or
    invalid setting is saved as False."""
    value = _load_setting(OFFER_KEY)
    if not isinstance(value, bool):
        _save_setting(OFFER_KEY, False)
        return False
    return value
