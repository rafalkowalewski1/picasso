"""
picasso.render.overlays_qt
~~~~~~~~~~~~~~~~~~~~~~~~~~

Qt drawing on rendered images (QImage): picks, points, scale bar,
legend, minimap and rotation widgets, plus PDF/SVG export.

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from .. import lib
from .geometry import (
    to_rotation,
    viewport_height,
    viewport_width,
    adjust_viewport_decorator,
    map_to_view,
)

if TYPE_CHECKING:
    from PyQt6 import QtGui, QtCore, QtSvg
else:
    # PyQt6 is imported on first attribute access so that importing
    # picasso.render does not require PyQt6.
    QtGui = lib._LazyQtModule("PyQt6.QtGui")
    QtCore = lib._LazyQtModule("PyQt6.QtCore")
    QtSvg = lib._LazyQtModule("PyQt6.QtSvg")


POLYGON_POINTER_SIZE = 16  # must be even
# opacity of the fill of a brush pick, so that the localizations under
# the painted region stay visible
BRUSH_FILL_ALPHA = 70


def export_qimage_to_pdf(
    image: QtGui.QImage, path: str, dpi: int = 96
) -> None:
    """Write a rendered image to a PDF at its original physical size.

    The page is sized so that one image pixel is 1/96 inch regardless of
    ``dpi``, which only sets the resolution the image is rasterized at.

    Parameters
    ----------
    image : QtGui.QImage
        The rendered image.
    path : str
        Where to write the PDF.
    dpi : int, optional
        Resolution of the PDF writer. Default 96.
    """
    writer = QtGui.QPdfWriter(path)

    # Fixed physical page size (1 image pixel = 1/96 inch, regardless of dpi)
    width_mm = image.width() * 25.4 / 96
    height_mm = image.height() * 25.4 / 96

    page_size = QtGui.QPageSize(
        QtCore.QSizeF(width_mm, height_mm),
        QtGui.QPageSize.Unit.Millimeter,
    )
    writer.setPageSize(page_size)
    writer.setResolution(dpi)

    # Painter coordinates: 1 unit = 1/dpi inch, so full page =
    # (width_mm / 25.4) * dpi = image.width() * dpi / 96
    draw_width = image.width() * dpi / 96
    draw_height = image.height() * dpi / 96

    painter = QtGui.QPainter(writer)
    painter.drawImage(QtCore.QRectF(0, 0, draw_width, draw_height), image)
    painter.end()


def export_qimage_to_svg(image: QtGui.QImage, path: str):
    """Write a rendered image to an SVG, embedded at its pixel size.

    Parameters
    ----------
    image : QtGui.QImage
        The rendered image.
    path : str
        Where to write the SVG.
    """
    generator = QtSvg.QSvgGenerator()
    generator.setFileName(path)
    generator.setSize(image.size())
    generator.setViewBox(QtCore.QRect(0, 0, image.width(), image.height()))

    painter = QtGui.QPainter(generator)
    painter.drawImage(0, 0, image)
    painter.end()


def get_rectangle_pick_polygon(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    width: float,
    return_most_right: bool = False,
) -> QtGui.QPolygonF | tuple[float, float]:
    """Find QtGui.QPolygonF object used for drawing a rectangular
    pick.

    Parameters
    ----------
    start_x, start_y : float
        One end of the rectangle's center line.
    end_x, end_y : float
        The other end of the center line.
    width : float
        Width of the rectangle, perpendicular to the center line.
    return_most_right : bool, optional
        Also return the rightmost corner, where the GUI anchors the pick
        label. Default False.

    Returns
    -------
    p : QtGui.QPolygonF
        The polygon.
    most_right : tuple
        Only if ``return_most_right``: the ``(x, y)`` of the rightmost corner.
    """
    X, Y = lib.get_pick_rectangle_corners(
        start_x, start_y, end_x, end_y, width
    )
    p = QtGui.QPolygonF()
    for x, y in zip(X, Y):
        p.append(QtCore.QPointF(x, y))
    if return_most_right:
        ix_most_right = np.argmax(X)
        x_most_right = X[ix_most_right]
        y_most_right = Y[ix_most_right]
        return p, (x_most_right, y_most_right)
    return p


def _draw_picks_circle(
    image: QtGui.QImage,
    viewport: list[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # pick coords in camera pixels
    pick_size: float,  # diameter in camera pixels
    point_picks: bool = False,
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
) -> QtGui.QImage:
    """Draw circular picks onto the image of rendered localizations.
    See ``draw_picks`` for more details."""
    if color is None:
        color = QtGui.QColor("yellow")
    if point_picks:  # draw circular picks as points
        painter = QtGui.QPainter(image)
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(color)
        for i, pick in enumerate(picks):
            # convert from camera units to display units
            cx, cy = map_to_view(*pick, image.size(), viewport)
            painter.drawEllipse(QtCore.QPoint(cx, cy), 3, 3)
            if annotate_picks:
                painter.drawText(cx + 20, cy + 20, str(i))

    else:  # draw circles
        d = int(pick_size * image.width() / viewport_width(viewport))
        painter = QtGui.QPainter(image)
        painter.setPen(color)
        for i, pick in enumerate(picks):
            # check that the pick is within the view
            if (
                pick[0] < viewport[0][1]
                or pick[0] > viewport[1][1]
                or pick[1] < viewport[0][0]
                or pick[1] > viewport[1][0]
            ):
                continue

            # convert from camera units to display units
            cx, cy = map_to_view(*pick, image.size(), viewport)
            painter.drawEllipse(int(cx - d / 2), int(cy - d / 2), d, d)
            if annotate_picks:
                painter.drawText(int(cx + d / 2), int(cy + d / 2), str(i))
    painter.end()
    return image


def _draw_picks_rectangle(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    pick_size: float,  # width in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
) -> QtGui.QImage:
    """Draw rectangular picks onto the image of rendered
    localizations. See ``draw_picks`` for more details."""
    if color is None:
        color = QtGui.QColor("yellow")
    w = pick_size * image.width() / viewport_width(viewport)
    painter = QtGui.QPainter(image)
    painter.setPen(color)
    for i, pick in enumerate(picks):
        # convert from camera units to display units
        start_x, start_y = map_to_view(*pick[0], image.size(), viewport)
        end_x, end_y = map_to_view(*pick[1], image.size(), viewport)
        # draw a straight line across the pick
        painter.drawLine(start_x, start_y, end_x, end_y)
        # draw a rectangle
        polygon, most_right = get_rectangle_pick_polygon(
            start_x, start_y, end_x, end_y, w, return_most_right=True
        )
        painter.drawPolygon(polygon)
        if annotate_picks:
            painter.drawText(*most_right, str(i))
    painter.end()
    return image


def _draw_picks_polygon(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
) -> QtGui.QImage:
    """Draw polygon picks onto the image of rendered localizations. See
    ``draw_picks`` for more details."""
    if color is None:
        color = QtGui.QColor("yellow")
    painter = QtGui.QPainter(image)
    painter.setPen(color)
    for i, pick in enumerate(picks):
        oldpoint = []
        for point in pick:
            cx, cy = map_to_view(*point, image.size(), viewport)
            painter.drawEllipse(
                QtCore.QPoint(cx, cy),
                int(POLYGON_POINTER_SIZE / 2),
                int(POLYGON_POINTER_SIZE / 2),
            )
            if oldpoint != []:  # draw the line
                ox, oy = map_to_view(*oldpoint, image.size(), viewport)
                painter.drawLine(cx, cy, ox, oy)
            oldpoint = point

        # annotate picks
        if len(pick) and annotate_picks:
            painter.drawText(
                cx + int(POLYGON_POINTER_SIZE / 2) + 10,
                cy + int(POLYGON_POINTER_SIZE / 2) + 10,
                str(i),
            )
    painter.end()
    return image


def _draw_picks_square(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    pick_size: float,  # side length in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
) -> QtGui.QImage:
    """Draw square picks onto the image of rendered localizations."""
    if color is None:
        color = QtGui.QColor("yellow")
    w = int(pick_size * image.width() / viewport_width(viewport))
    painter = QtGui.QPainter(image)
    painter.setPen(color)
    for i, pick in enumerate(picks):
        # check that the pick is within the view
        if (
            pick[0] < viewport[0][1]
            or pick[0] > viewport[1][1]
            or pick[1] < viewport[0][0]
            or pick[1] > viewport[1][0]
        ):
            continue

        # convert from camera units to display units
        cx, cy = map_to_view(*pick, image.size(), viewport)
        painter.drawRect(int(cx - w / 2), int(cy - w / 2), w, w)

        # annotate picks
        if annotate_picks:
            painter.drawText(
                int(cx + w / 2) + 10, int(cy + w / 2) + 10, str(i)
            )
    painter.end()
    return image


def _draw_picks_box(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
) -> QtGui.QImage:
    """Draw box picks onto the image of rendered localizations. See
    ``draw_picks`` for more details."""
    if color is None:
        color = QtGui.QColor("yellow")
    painter = QtGui.QPainter(image)
    painter.setPen(color)
    for i, pick in enumerate(picks):
        X, Y = lib.get_pick_box_corners(pick)
        # unlike the click-placed shapes, a box can be larger than the
        # view, so cull on intersection rather than on its center
        if (
            max(X) < viewport[0][1]
            or min(X) > viewport[1][1]
            or max(Y) < viewport[0][0]
            or min(Y) > viewport[1][0]
        ):
            continue

        # convert from camera units to display units
        x0, y0 = map_to_view(min(X), min(Y), image.size(), viewport)
        x1, y1 = map_to_view(max(X), max(Y), image.size(), viewport)
        painter.drawRect(x0, y0, x1 - x0, y1 - y0)

        # annotate picks
        if annotate_picks:
            painter.drawText(x1 + 10, y1 + 10, str(i))
    painter.end()
    return image


def brush_pick_path(
    pick: list[tuple[float, list[tuple[float, float]]]],
    image_size: QtCore.QSize,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
) -> QtGui.QPainterPath:
    """Return the painted outline of a brush pick in display units.

    Each stroke is swept with a round-capped, round-joined pen of its
    own width - the same region ``lib.check_if_in_brush_stroke`` tests
    against - and the strokes of a pick are united into a single path,
    so that the region can be filled once. Filling stroke by stroke
    would darken every overlap, and the strokes of a merged pick always
    overlap.

    Parameters
    ----------
    pick : list of tuples
        One brush pick, i.e., a list of ``(width, path)`` strokes in
        camera pixels.
    image_size : QSize
        Size of the image the pick is drawn onto.
    viewport : tuple
        Current field of view in camera pixels, ``((y_min, x_min),
        (y_max, x_max))``.

    Returns
    -------
    path : QPainterPath
        The painted region in display coordinates.
    """
    scale = image_size.width() / viewport_width(viewport)
    region = QtGui.QPainterPath()
    for stroke in pick:
        width, X, Y = lib.brush_stroke_arrays(stroke)
        path = QtGui.QPainterPath()
        for j, (x, y) in enumerate(zip(X, Y)):
            cx, cy = map_to_view(x, y, image_size, viewport)
            point = QtCore.QPointF(cx, cy)
            if j == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        if len(X) == 1:  # a dot: sweeping a zero-length path
            path.lineTo(path.currentPosition())
        pen = QtGui.QPen(
            QtGui.QColor("black"),
            max(width * scale, 1.0),
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenCapStyle.RoundCap,
            QtCore.Qt.PenJoinStyle.RoundJoin,
        )
        stroker = QtGui.QPainterPathStroker(pen)
        region = region.united(stroker.createStroke(path))
    return region.simplified()


def _draw_picks_brush(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
) -> QtGui.QImage:
    """Draw brush picks onto the image of rendered localizations, as a
    translucent highlight with a solid outline. See ``draw_picks`` for
    more details."""
    if color is None:
        color = QtGui.QColor("yellow")
    fill = QtGui.QColor(color)
    fill.setAlpha(BRUSH_FILL_ALPHA)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(color)
    for i, pick in enumerate(picks):
        if not len(pick):
            continue
        x_min, x_max, y_min, y_max = lib.pick_bounds(pick, "Brush", None)
        # a painted region can be larger than the view, so cull on
        # intersection rather than on a center
        if (
            x_max < viewport[0][1]
            or x_min > viewport[1][1]
            or y_max < viewport[0][0]
            or y_min > viewport[1][0]
        ):
            continue

        region = brush_pick_path(pick, image.size(), viewport)
        painter.fillPath(region, QtGui.QBrush(fill))
        painter.drawPath(region)

        # annotate picks at the start of the first stroke
        if annotate_picks:
            cx, cy = map_to_view(
                pick[0][1][0][0], pick[0][1][0][1], image.size(), viewport
            )
            painter.drawText(cx + 10, cy + 10, str(i))
    painter.end()
    return image


@adjust_viewport_decorator
def draw_picks(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ],
    picks: list[tuple],  # pick coords in camera pixels
    pick_size: float | None,  # diameter in camera pixels
    point_picks: bool = False,
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
) -> QtGui.QImage:
    """Draw all selected picks onto the image (QImage) of rendered
    localizations.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    viewport : tuple
        Current field of view in camera pixels, ((y_min, y_max), (x_min,
        x_max)).
    pick_shape : {"Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"}
        Shape of the picks to be drawn.
    picks : list of tuples
        List of picks, where each pick is a tuple specifying the pick
        coordinates. Note: this must match the format of the given pick
        shape.
    pick_size : float or None
        Size of the picks in camera pixels. For "Circle", this is the
        diameter; for "Rectangle", this is the width; for "Square", this
        is the side length. This parameter is ignored for "Polygon",
        "Box" and "Brush" picks, which carry their own extent.
    point_picks : bool, optional
        If True and pick_shape is "Circle", draw picks as points instead
        of circles. Default is False.
    annotate_picks : bool, optional
        If True, annotate each pick with its index in the picks list.
        Default is False.
    color : QtGui.QColor, optional
        Color of the picks. Default is yellow.

    Returns
    -------
    image : QImage
        Image with the drawn picks.

    Raises
    ------
    ValueError
        If ``pick_shape`` is not recognized.
    """
    image = image.copy()
    if pick_shape == "Circle":
        return _draw_picks_circle(
            image,
            viewport=viewport,
            picks=picks,
            pick_size=pick_size,
            point_picks=point_picks,
            annotate_picks=annotate_picks,
            color=color,
        )
    elif pick_shape == "Rectangle":
        return _draw_picks_rectangle(
            image,
            viewport=viewport,
            picks=picks,
            pick_size=pick_size,
            annotate_picks=annotate_picks,
            color=color,
        )
    elif pick_shape == "Polygon":
        return _draw_picks_polygon(
            image,
            viewport=viewport,
            picks=picks,
            annotate_picks=annotate_picks,
            color=color,
        )
    elif pick_shape == "Square":
        return _draw_picks_square(
            image,
            viewport=viewport,
            picks=picks,
            pick_size=pick_size,
            annotate_picks=annotate_picks,
            color=color,
        )
    elif pick_shape == "Box":
        return _draw_picks_box(
            image,
            viewport=viewport,
            picks=picks,
            annotate_picks=annotate_picks,
            color=color,
        )
    elif pick_shape == "Brush":
        return _draw_picks_brush(
            image,
            viewport=viewport,
            picks=picks,
            annotate_picks=annotate_picks,
            color=color,
        )
    else:
        raise ValueError(f"Unknown pick shape: {pick_shape}")


@adjust_viewport_decorator
def draw_points(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    points: list[tuple],  # points in camera pixels,
    pixelsize: int | float,  # camera pixel size in nm
    color: QtGui.QColor | None = None,  # default: yellow
    mark_width: int = 20,  # width of the drawn crosses in display pixels
    cursor: tuple | None = None,  # live cursor position in camera pixels
) -> QtGui.QImage:
    """Draw points, lines and distances between them onto image.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    viewport : tuple
        Current field of view in camera pixels, ((y_min, y_max), (x_min,
        x_max)).
    points : list of tuples
        List of points, where each point is a tuple specifying the point
        coordinates in camera pixels.
    pixelsize : int or float
        Camera pixel size in nm.
    color : QtGui.QColor, optional
        Color of the points, lines and text. Default is yellow.
    mark_width : int, optional
        Width of the drawn crosses in display pixels. Default is 20.
    cursor : tuple or None, optional
        Current cursor position in camera pixels. If given, it is drawn
        as a cross and, when at least one point exists, a line with the
        live distance to the last point is shown. Default is None.

    Returns
    -------
    image : QImage
        Image with the drawn points.
    """
    if color is None:
        color = QtGui.QColor("yellow")
    painter = QtGui.QPainter(image)
    painter.setPen(color)

    def draw_cross(x, y):
        """Draw a cross marker centered at display coordinates."""
        painter.drawPoint(x, y)
        painter.drawLine(x, y, int(x + mark_width / 2), y)
        painter.drawLine(x, y, x, int(y + mark_width / 2))
        painter.drawLine(x, y, int(x - mark_width / 2), y)
        painter.drawLine(x, y, x, int(y - mark_width / 2))

    def draw_distance(x1, y1, x2, y2, p1, p2):
        """Draw a line and the distance label between two points."""
        painter.drawLine(x1, y1, x2, y2)
        font = painter.font()
        font.setPixelSize(20)
        painter.setFont(font)
        # get distance with 2 decimal places
        distance = (
            float(
                int(
                    np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                    * pixelsize
                    * 100
                )
            )
            / 100
        )
        painter.drawText(
            int((x1 + x2) / 2 + mark_width),
            int((y1 + y2) / 2 + mark_width),
            str(distance) + " nm",
        )

    cx = []
    cy = []
    ox = []  # together with oldpoint used for drawing
    oy = []  # lines between points
    oldpoint = []
    for point in points:
        # convert to display units
        if oldpoint != []:
            ox, oy = map_to_view(*oldpoint, image.size(), viewport=viewport)
        cx, cy = map_to_view(*point, image.size(), viewport=viewport)

        # draw a cross
        draw_cross(cx, cy)

        # draw a line between points and show distance
        if oldpoint != []:
            draw_distance(cx, cy, ox, oy, oldpoint, point)
        oldpoint = point

    # draw the live cursor as a cross and the running distance to the
    # last placed point
    if cursor is not None:
        ccx, ccy = map_to_view(*cursor, image.size(), viewport=viewport)
        draw_cross(ccx, ccy)
        if points:
            lx, ly = map_to_view(*points[-1], image.size(), viewport=viewport)
            draw_distance(ccx, ccy, lx, ly, points[-1], cursor)

    painter.end()
    return image


@adjust_viewport_decorator
def draw_scalebar(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],
    scalebar_length_nm: int | float,
    pixelsize: int | float,
    display_length: bool = True,
    color: QtGui.QColor | None = None,  # default: white
    display_height: int = 10,
    margin: tuple[int, int] = (35, 20),
    text_spacer: int = 40,
    text_fontsize: int = 20,
) -> QtGui.QImage:
    """Draw a scalebar into rendered localizations (QImage).

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    viewport : tuple
        Current field of view in camera pixels, ((y_min, y_max), (x_min,
        x_max)).
    scalebar_length_nm : int or float
        Scale bar length in nm.
    pixelsize : int or float
        Camera pixel size in nm.
    color : QColor, optional
        Color of the scalebar and text. Default is white.
    display_length : bool, optional
        Whether to display scalebar length in nm. Default is True.
    display_height : int, optional
        Thickness of the scalebar in display pixels. Default is 10.
    margin : tuple of int, optional
        Margins from the right and bottom edges in display pixels.
        Default is (35, 20).
    text_spacer : int, optional
        Spacing between the scalebar and the displayed length text in
        display pixels. Only used if display_length is True. Default is
        40.
    text_fontsize : int, optional
        Font size of the displayed length text in display pixels. Only
        used if display_length is True. Default is 20.

    Returns
    -------
    image : QImage
        Image with the drawn scalebar.
    """
    if color is None:
        color = QtGui.QColor("white")
    painter = QtGui.QPainter(image)
    painter.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
    painter.setBrush(QtGui.QBrush(color))

    length_camerapxl = scalebar_length_nm / pixelsize
    length_displaypxl = int(
        round(image.width() * length_camerapxl / viewport_width(viewport))
    )

    # draw a rectangle
    x = image.width() - length_displaypxl - margin[0]
    y = image.height() - display_height - margin[1]
    painter.drawRect(x, y, length_displaypxl, display_height)

    # display scalebar's length
    if display_length:
        font = painter.font()
        font.setPixelSize(text_fontsize)
        painter.setFont(font)
        painter.setPen(color)
        text_width = length_displaypxl + 2 * text_spacer
        text_height = text_spacer
        painter.drawText(
            x - text_spacer,
            y - 25,
            text_width,
            text_height,
            QtCore.Qt.AlignmentFlag.AlignHCenter,
            f"{str(scalebar_length_nm)} nm",
        )
    return image


def draw_legend(
    image: QtGui.QImage,
    channel_names: list[str],
    channel_colors: list[tuple[int, int, int]],
    init_pos: tuple[int, int] = (12, 26),
    dy: int = 24,
    padding: int = 4,
    text_fontsize: int = 16,
) -> QtGui.QImage:
    """Draw a legend for multichannel data in the top left corner over
    rendered localizations (QImage).

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    channel_names : list of str
        List of channel names to be displayed in the legend.
    channel_colors : list of tuples
        List of RGB tuples corresponding to the colors of the channels.
        Must range between 0 and 255.
    init_pos : tuple of int, optional
        Initial position (x, y) of the first channel name in display
        pixels. Default is (12, 26).
    dy : int, optional
        Space between channel names in display pixels. Default is 24.
    padding : int, optional
        Padding around the text in display pixels. Default is 4.
    text_fontsize : int, optional
        Font size of the channel names in display pixels. Default is 16.

    Returns
    -------
    image : QImage
        Image with the drawn legend.
    """
    assert len(channel_names) == len(channel_colors), (
        "Length of channel_names must match number of channels in " "dataset."
    )
    n_channels = len(channel_names)
    painter = QtGui.QPainter(image)
    # initial positions
    x, y = init_pos
    font = painter.font()
    font.setPixelSize(text_fontsize)
    painter.setFont(font)
    fm = QtGui.QFontMetrics(font)
    for i in range(n_channels):
        text = channel_names[i]
        # draw black background
        text_rect = fm.boundingRect(text)
        bg_rect = QtCore.QRect(
            x - padding,
            y - fm.ascent() - padding,
            text_rect.width() + 2 * padding,
            fm.height() + 2 * padding,
        )
        painter.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.black))
        painter.drawRect(bg_rect)
        # draw colored text
        color_rgb = channel_colors[i]
        color = QtGui.QColor(color_rgb[0], color_rgb[1], color_rgb[2])
        painter.setPen(QtGui.QPen(color))
        painter.drawText(QtCore.QPoint(x, y), text)
        y += dy
    return image


@adjust_viewport_decorator
def draw_minimap(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    max_viewport_size: tuple[float, float],  # in camera pixels,
    color_main: QtGui.QColor | None = None,  # default: yellow
    color_frame: QtGui.QColor | None = None,  # default: white
    length_minimap: int = 100,
    margin: tuple[int, int] = (20, 20),
) -> QtGui.QImage:
    """Draw a minimap showing the position of current viewport.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    viewport : tuple
        Current field of view in camera pixels, ((y_min, y_max), (x_min,
        x_max)).
    max_viewport_size : tuple
        Maximum viewport size in camera pixels, i.e., the acquired
        movie size (height, width).
    color_main, color_frame : QColor, optional
        Colors of the viewport and the minimap frame. Default is yellow
        and white, respectively.
    length_minimap : int, optional
        Length of the minimap in pixels. Default is 100.
    margin : tuple of int, optional
        Margins from the right and top edges in display pixels.
        Default is (20, 20).

    Returns
    -------
    image : QImage
        Image with the drawn minimap.
    """
    if color_main is None:
        color_main = QtGui.QColor("yellow")
    if color_frame is None:
        color_frame = QtGui.QColor("white")
    movie_height, movie_width = max_viewport_size
    height_minimap = int(movie_height / movie_width * length_minimap)
    # draw in the upper right corner, overview rectangle
    x = image.width() - length_minimap - margin[0]
    y = margin[1]
    painter = QtGui.QPainter(image)
    painter.setPen(color_frame)
    painter.drawRect(x, y, length_minimap, height_minimap)
    painter.setPen(color_main)
    length = int(viewport_width(viewport) / movie_width * length_minimap)
    length = max(5, length)
    height = int(viewport_height(viewport) / movie_height * height_minimap)
    height = max(5, height)
    x_vp = int(viewport[0][1] / movie_width * length_minimap)
    y_vp = int(viewport[0][0] / movie_height * height_minimap)
    painter.drawRect(x + x_vp, y + y_vp, length, height)
    return image


def draw_rotation(
    image: QtGui.QImage,
    ang: tuple[float, float, float] | Rotation,
    axis_length: int = 30,
    axis_center: tuple[int, int] = (50, -50),  # bottom left
) -> QtGui.QImage:
    """Draw rotation axes icon on the image.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    ang : tuple of float or scipy.spatial.transform.Rotation
        Rotation of the localizations; either a scipy Rotation or a
        tuple of 3 rotation angles around the x, y and z axes in
        radians (legacy Euler convention, see ``rotation_matrix``).
    axis_length : int, optional
        Length of the rotation axes in display pixels. Default is 30.
    axis_center : tuple of int, optional
        Position of the rotation axes icon in display pixels, with
        origin in the top left corner. Negative values indicated
        counting from the bottom right corner. Default is (50, -50).

    Returns
    -------
    image : QImage
        Image with the drawn rotation axes icon.
    """
    painter = QtGui.QPainter(image)
    x = (
        axis_center[0]
        if axis_center[0] >= 0
        else image.width() + axis_center[0]
    )
    y = (
        axis_center[1]
        if axis_center[1] >= 0
        else image.height() + axis_center[1]
    )
    center = QtCore.QPoint(x, y)

    # set the ends of the x line
    xx = axis_length
    xy = 0
    xz = 0

    # set the ends of the y line
    yx = 0
    yy = axis_length
    yz = 0

    # set the ends of the z line
    zx = 0
    zy = 0
    zz = axis_length

    # rotate these points
    coordinates = [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]
    R = to_rotation(ang)
    coordinates = R.apply(coordinates).astype(int)
    (xx, xy, xz) = coordinates[0]
    (yx, yy, yz) = coordinates[1]
    (zx, zy, zz) = coordinates[2]

    # translate the x and y coordinates of the end points towards
    # bottom right edge of the window
    xx += x
    xy += y
    yx += x
    yy += y
    zx += x
    zy += y

    # set the points at the ends of the lines
    point_x = QtCore.QPoint(xx, xy)
    point_y = QtCore.QPoint(yx, yy)
    point_z = QtCore.QPoint(zx, zy)
    line_x = QtCore.QLine(center, point_x)
    line_y = QtCore.QLine(center, point_y)
    line_z = QtCore.QLine(center, point_z)
    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(1, 0, 0, 1)))
    painter.drawLine(line_x)
    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(0, 1, 1, 1)))
    painter.drawLine(line_y)
    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(0, 1, 0, 1)))
    painter.drawLine(line_z)
    return image


def draw_rotation_angles(
    image: QtGui.QImage,
    ang: tuple[float, float, float],
    color: QtGui.QColor | None = None,  # default: white
) -> QtGui.QImage:
    """Draw rotation angles (numbers in degrees) on the image.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    ang : tuple of float
        Rotation angles (or rotation-vector components) around x, y,
        and z axes in radians, used only for the displayed text.
    color : QColor, optional
        Color of the text. Default is white.

    Returns
    -------
    image : QImage
        Image with the drawn rotation angles.
    """
    if color is None:
        color = QtGui.QColor("white")
    angx, angy, angz = [int(np.round(_ * 180 / np.pi, 0)) for _ in ang]
    text = f"{angx} {angy} {angz}"
    x = image.width() - len(text) * 8 - 10
    y = image.height() - 20
    painter = QtGui.QPainter(image)
    font = painter.font()
    font.setPixelSize(12)
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(QtCore.QPoint(x, y), text)
    return image


def rgb_to_qimage(
    image: lib.IntArray3D, return_bgra: bool = False
) -> QtGui.QImage | tuple[QtGui.QImage, lib.IntArray3D]:
    """Convert a numpy array of shape (height, width, 3) with integer
    values between 0 and 255 to a QImage.

    Parameters
    ----------
    image : IntArray3D
        RGB image as a numpy array of shape (height, width, 3) with
        integer values between 0 and 255.
    return_bgra : bool, optional
        If True, return the BGRA numpy array instead of a QImage.
        Default is False.

    Returns
    -------
    qimage : QImage
        The converted QImage.
    bgra : IntArray3D
        The BGRA numpy array. Only returned if return_bgra is True.
    """
    bgra = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    bgra[:, :, 0] = image[:, :, 2]  # R -> B
    bgra[:, :, 1] = image[:, :, 1]  # G -> G
    bgra[:, :, 2] = image[:, :, 0]  # B -> R
    bgra[:, :, 3] = 255  # A -> 255 (opaque)
    Y, X = image.shape[:2]
    qimage = QtGui.QImage(bgra.data, X, Y, QtGui.QImage.Format.Format_RGB32)
    qimage = qimage.copy()  # make a deep copy to own the data DO NOT DELETE
    if return_bgra:
        return qimage, bgra
    return qimage


def optimal_scalebar_length(pixelsize: int | float, width: int | float) -> int:
    """Calculate optimal scale bar length in nm based on the image
    width.

    Parameters
    ----------
    pixelsize : int or float
        Camera pixel size in nm.
    width : int or float
        Image width in camera pixels.

    Returns
    -------
    scalebar : int
        Suggested scale bar length in nm.
    """
    width_nm = width * pixelsize
    optimal_scalebar = width_nm / 8
    # approximate to the nearest thousands, hundreds, tens or ones
    if optimal_scalebar > 10_000:
        scalebar = 10_000
    elif optimal_scalebar > 1_000:
        scalebar = int(1_000 * round(optimal_scalebar / 1_000))
    elif optimal_scalebar > 100:
        scalebar = int(100 * round(optimal_scalebar / 100))
    elif optimal_scalebar > 10:
        scalebar = int(10 * round(optimal_scalebar / 10))
    else:
        scalebar = int(round(optimal_scalebar))
    return scalebar
