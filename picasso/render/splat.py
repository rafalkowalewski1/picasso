"""
picasso.render.splat
~~~~~~~~~~~~~~~~~~~~

Raw splat stage: turn localization coordinates into (blurred) count
images. ``_RenderColumns`` carries the per-localization arrays and the
``_render_*`` functions dispatch per blur method (CPU backend).

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from concurrent import futures
from typing import Literal

import numba
import numpy as np
import pandas as pd
from scipy import signal, ndimage
from scipy.spatial.transform import Rotation

from .. import lib
from .backend import SplatBackend
from .kernels import (
    _render_setup,
    _render_setup3d,
    _render_setup3d_anisotropic,
    _fill,
    _fill3d,
    _fill_gaussian,
    _fill_gaussian_theta,
    _fill_gaussian_rot,
    _fill_gaussian_rot_theta,
)
from .geometry import to_rotation


def render(
    locs: pd.DataFrame,
    info: dict,
    *,
    disp_px_size: float,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    max_blur_width: float | None = None,
    ang: tuple | Rotation | None = None,
    indices: lib.IntArray1D | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations given FOV and blur method.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered.
    info : dict
        Contains localizations metadata.
    disp_px_size : float
        Display pixel size in nm.
    viewport : tuple, optional
        Field of view to be rendered (in camera pixels). The input is
        ``((y_min, x_min), (y_max, x_max))``. If None, all localizations
        are rendered.
    blur_method : {"gaussian", "gaussian_iso", "smooth", "convolve"} or None, \
            optional
        Defines localizations' blur. The string has to be one of
        'gaussian', 'gaussian_iso', 'smooth', 'convolve'. If None, no
        blurring is applied. 'gaussian' uses localization precisions
        of each localization to blur it (different in each dimension).
        'gaussian_iso' is similar but averages x and y localization
        precisions, so that blur is isotropic. 'smooth' applies a one
        pixel blur. 'convolve' applies the same blur to all
        localizations which is the median localization precision.
    min_blur_width : float, optional
        Minimum size of blur (camera pixels).
    max_blur_width : float, optional
        Localizations whose ``lpx`` or ``lpy`` exceeds this (camera
        pixels) are not rendered by 'gaussian' and 'gaussian_iso'
        (see ``_extract_render_columns``). If None (default), all
        localizations are rendered.
    ang : tuple or scipy.spatial.transform.Rotation, optional
        Rotation of locs; either a scipy Rotation (e.g. built from a
        quaternion) or a tuple of 3 rotation angles around the x, y
        and z axes in radians (legacy Euler convention, see
        ``rotation_matrix``). If None, locs are not rotated.
    indices : lib.IntArray1D, optional
        Positions of the rows of ``locs`` to render (e.g. a viewport
        pre-selection from ``spatial_index``); the other rows are
        ignored. If None (default), all rows are rendered.

    Raises
    ------
    Exception
        If blur_method not one of 'gaussian', 'gaussian_iso', 'smooth',
        'convolve' or None.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    return _render_arrays(
        _extract_render_columns(
            locs, blur_method, ang, max_blur_width, indices
        ),
        info,
        disp_px_size=disp_px_size,
        viewport=viewport,
        blur_method=blur_method,
        min_blur_width=min_blur_width,
        ang=ang,
    )


class _RenderColumns:
    """Numpy views of the localization columns one render needs.

    Chunked parallel rendering slices these arrays instead of
    DataFrames: the per-chunk pandas overhead (``iloc``, ``to_numpy``)
    holds the GIL and measurably caps the thread pool's efficiency.
    Extraction happens once per channel; ``angle`` is stored in radians
    and ``lpz`` with its fallback already applied, so chunk slices are
    plain array views.
    """

    __slots__ = ("x", "y", "lpx", "lpy", "lpz", "angle", "z", "indices")

    def __init__(
        self,
        x,
        y,
        lpx=None,
        lpy=None,
        lpz=None,
        angle=None,
        z=None,
        indices=None,
    ):
        self.x = x
        self.y = y
        self.lpx = lpx
        self.lpy = lpy
        self.lpz = lpz
        self.angle = angle
        self.z = z
        #: rows to render (positions into the column arrays), or None
        #: for all of them: the render-index pyramid's viewport
        #: selection travels this way, so a GPU backend keeps the whole
        #: channel resident and reads only the selected rows
        self.indices = indices

    def __len__(self) -> int:
        """Number of rows to render."""
        if self.indices is not None:
            return len(self.indices)
        return len(self.x)

    def slice(self, start: int, stop: int) -> "_RenderColumns":
        """Row range (of the rows to render) as array views, no copies."""
        if self.indices is not None:
            return _RenderColumns(
                self.x,
                self.y,
                self.lpx,
                self.lpy,
                self.lpz,
                self.angle,
                self.z,
                self.indices[start:stop],
            )

        def cut(array):
            return None if array is None else array[start:stop]

        return _RenderColumns(
            self.x[start:stop],
            self.y[start:stop],
            cut(self.lpx),
            cut(self.lpy),
            cut(self.lpz),
            cut(self.angle),
            cut(self.z),
        )

    def materialize(self) -> "_RenderColumns":
        """The selected rows gathered into contiguous arrays (a copy,
        like ``DataFrame.iloc`` with the same indices); a no-op without
        ``indices``."""
        if self.indices is None:
            return self

        def take(array):
            return None if array is None else array[self.indices]

        return _RenderColumns(
            self.x[self.indices],
            self.y[self.indices],
            take(self.lpx),
            take(self.lpy),
            take(self.lpz),
            take(self.angle),
            take(self.z),
        )


def _extract_render_columns(
    locs: pd.DataFrame,
    blur_method: str | None,
    ang: tuple | Rotation | None,
    max_blur_width: float | None = None,
    indices: lib.IntArray1D | None = None,
) -> _RenderColumns:
    """Pull the columns ``blur_method`` (and rotation) needs out of the
    DataFrame, converting angle to radians and applying the lpz
    fallback once per channel.

    With ``max_blur_width`` (camera pixels), the per-localization blur
    methods (``gaussian``, ``gaussian_iso``) drop localizations whose
    ``lpx`` or ``lpy`` exceeds it: such precisions are useless
    artifacts of unfiltered data, their blur would cover a large FOV
    with a negligible intensity, and rendering them costs too much.
    Filtering here keeps every backend in agreement, including the count
    of rendered localizations.

    ``indices`` (positions into ``locs``) restrict the render to those
    rows without copying the columns (see ``_RenderColumns``); the
    filter above is applied to them as well."""
    need_lp = blur_method in ("gaussian", "gaussian_iso", "convolve")
    lpx = locs["lpx"].to_numpy() if need_lp else None
    lpy = locs["lpy"].to_numpy() if need_lp else None
    angle = None
    if blur_method == "gaussian" and "angle" in locs:
        # the stored column is in degrees, the kernels expect radians
        angle = np.deg2rad(locs["angle"].to_numpy())
    z = None
    lpz = None
    if ang is not None:
        z = locs["z"].to_numpy()
        if blur_method in ("gaussian", "gaussian_iso"):
            if "lpz" in locs:
                lpz = locs["lpz"].to_numpy()
            else:
                # if lpz not found, make it twice the mean of lpx and lpy
                lpz = 2 * locs[["lpx", "lpy"]].to_numpy().mean(axis=1)
    columns = [locs["x"].to_numpy(), locs["y"].to_numpy(), lpx, lpy, lpz]
    columns += [angle, z]
    if max_blur_width is not None and blur_method in (
        "gaussian",
        "gaussian_iso",
    ):
        keep = (lpx <= max_blur_width) & (lpy <= max_blur_width)
        if indices is not None:
            indices = indices[keep[indices]]
        elif not keep.all():
            columns = [None if c is None else c[keep] for c in columns]
    if indices is not None:
        indices = np.ascontiguousarray(indices, dtype=np.uint32)
    return _RenderColumns(*columns, indices=indices)


def _render_arrays(
    columns: _RenderColumns,
    info: dict,
    *,
    disp_px_size: float,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    ang: tuple | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """``render`` on pre-extracted column arrays (see ``render`` for
    the parameters). The chunked parallel scheduler calls this per row
    slice so no pandas work happens inside worker tasks."""
    # a row selection is gathered first: the kernels take dense arrays
    columns = columns.materialize()
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    oversampling = pixelsize / disp_px_size

    if viewport is None:
        height = lib.get_from_metadata(info, "Height", raise_error=True)
        width = lib.get_from_metadata(info, "Width", raise_error=True)
        viewport = [(0, 0), (height, width)]

    (y_min, x_min), (y_max, x_max) = viewport
    if blur_method is None:
        # no blur
        return _render_hist_arrays(
            columns,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            ang=ang,
        )
    elif blur_method == "gaussian":
        # individual localization precision
        return _render_gaussian(
            columns,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
            ang=ang,
        )
    elif blur_method == "gaussian_iso":
        # individual localization precision (same for x and y)
        return _render_gaussian_iso(
            columns,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
            ang=ang,
        )
    elif blur_method == "smooth":
        # one pixel blur
        return _render_smooth(
            columns,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            ang=ang,
        )
    elif blur_method == "convolve":
        # global localization precision
        return _render_convolve(
            columns,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
            ang=ang,
        )
    else:
        raise Exception("blur_method not understood.")


def _render_hist(
    locs: pd.DataFrame,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    ang: tuple[float, float, float] | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations with no blur by assigning them to pixels.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels)
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels)
    ang : tuple or scipy.spatial.transform.Rotation, optional
        Rotation of locs; either a scipy Rotation (e.g. built from a
        quaternion) or a tuple of 3 rotation angles around the x, y
        and z axes in radians (legacy Euler convention, see
        ``rotation_matrix``). If None, locs are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    return _render_hist_arrays(
        _extract_render_columns(locs, None, ang),
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
        ang=ang,
    )


def _render_hist_arrays(
    columns: _RenderColumns,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    ang: tuple[float, float, float] | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """``_render_hist`` on pre-extracted column arrays."""
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        columns.x,
        columns.y,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )
    if ang is not None:
        x, y, _, _ = _locs_rotation_arrays(
            columns,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )
    _fill(image, x, y)
    n = len(x)
    return n, image


@numba.jit(nopython=True, nogil=True)
def render_hist3d(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    z_min: float,
    z_max: float,
    pixelsize: float,
) -> tuple[int, lib.FloatArray3D]:
    """Render localizations in 3D with no blur by assigning them to
    pixels.

    Parameters
    ----------
    x, y : lib.FloatArray1D
        Lateral coordinates of the localizations (camera pixels).
    z : lib.FloatArray1D
        Axial coordinates of the localizations (nm).
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    z_min : float
        Minimum z coordinate to be rendered (nm).
    z_max : float
        Maximum z coordinate to be rendered (nm).
    pixelsize : float
        Camera pixel size in nm, used for converting z coordinates.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray3D
        Rendered 3D image.
    """
    z_min = z_min / pixelsize
    z_max = z_max / pixelsize

    image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view = _render_setup3d(
        x,
        y,
        z,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
        z_min,
        z_max,
        pixelsize,
    )
    _fill3d(image, x, y, z)
    n = len(x)
    return n, image


@numba.jit(nopython=True, nogil=True)
def render_hist3d_anisotropic(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D,
    oversampling_x: float,
    oversampling_y: float,
    oversampling_z: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    z_min: float,
    z_max: float,
    pixelsize: float,
) -> tuple[int, lib.FloatArray3D]:
    """Render localizations in 3D with no blur by assigning them to
    pixels. Allows for different pixel sizes in x, y and z
    (oversampling).

    Parameters
    ----------
    x, y : lib.FloatArray1D
        Lateral coordinates of the localizations (camera pixels).
    z : lib.FloatArray1D
        Axial coordinates of the localizations (nm).
    oversampling_x, oversampling_y, oversampling_z : float (default=1)
        Number of super-resolution pixels per camera pixel in x, y, and
        z directions.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    z_min : float
        Minimum z coordinate to be rendered (nm).
    z_max : float
        Maximum z coordinate to be rendered (nm).
    pixelsize : float
        Camera pixel size in nm, used for converting z coordinates.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray3D
        Rendered 3D image.
    """
    z_min = z_min / pixelsize
    z_max = z_max / pixelsize

    image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view = (
        _render_setup3d_anisotropic(
            x,
            y,
            z,
            oversampling_x,
            oversampling_y,
            oversampling_z,
            y_min,
            x_min,
            y_max,
            x_max,
            z_min,
            z_max,
            pixelsize,
        )
    )
    _fill3d(image, x, y, z)
    n = len(x)
    return n, image


def _render_gaussian(
    columns: _RenderColumns,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations with with individual localization precision
    which differs in x and y.

    Parameters
    ----------
    columns : _RenderColumns
        Column arrays of the localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, y_max : float
        Minimum and maximum y coordinates to be rendered (camera pixels).
    x_min, x_max : float
        Minimum and maximum x coordinates to be rendered (camera pixels).
    min_blur_width : float
        Minimum localization precision (camera pixels).
    ang : tuple or scipy.spatial.transform.Rotation, optional
        Rotation of localizations; either a scipy Rotation (e.g. built
        from a quaternion) or a tuple of 3 rotation angles around the
        x, y and z axes in radians (legacy Euler convention, see
        ``rotation_matrix``). If None, localizations are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        columns.x,
        columns.y,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    if ang is None:  # not rotated
        blur_width = oversampling * np.maximum(columns.lpx, min_blur_width)
        blur_height = oversampling * np.maximum(columns.lpy, min_blur_width)
        sy = blur_height[in_view]
        sx = blur_width[in_view]

        if columns.angle is not None:
            # per-localization in-plane rotation of the precision
            # ellipse (already converted to radians at extraction)
            angle = columns.angle[in_view]
            _fill_gaussian_theta(
                image, x, y, sx, sy, angle, n_pixel_x, n_pixel_y
            )
        else:
            _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    else:  # rotated
        x, y, in_view, z = _locs_rotation_arrays(
            columns,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )
        blur_width = oversampling * np.maximum(columns.lpx, min_blur_width)
        blur_height = oversampling * np.maximum(columns.lpy, min_blur_width)
        # lpz carries its fallback from extraction already
        blur_depth = oversampling * np.maximum(columns.lpz, min_blur_width)

        sy = blur_height[in_view]
        sx = blur_width[in_view]
        sz = blur_depth[in_view]

        rot_matrix = np.ascontiguousarray(
            to_rotation(ang).as_matrix(), dtype=np.float32
        )
        if columns.angle is not None:
            # per-localization in-plane rotation (radians), composed
            # with the global rotation
            angle = columns.angle[in_view]
            _fill_gaussian_rot_theta(
                image,
                x,
                y,
                sx,
                sy,
                sz,
                angle,
                n_pixel_x,
                n_pixel_y,
                rot_matrix,
            )
        else:
            _fill_gaussian_rot(
                image, x, y, sx, sy, sz, n_pixel_x, n_pixel_y, rot_matrix
            )

    n = len(x)
    return n, image


def _render_gaussian_iso(
    columns: _RenderColumns,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Same as ``_render_gaussian``, but uses the same localization
    precision in x and y."""
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        columns.x,
        columns.y,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    if ang is None:  # not rotated
        blur_width = oversampling * np.maximum(columns.lpx, min_blur_width)
        blur_height = oversampling * np.maximum(columns.lpy, min_blur_width)
        sy = (blur_height[in_view] + blur_width[in_view]) / 2
        sx = sy

        _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    else:  # rotated
        x, y, in_view, z = _locs_rotation_arrays(
            columns,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )
        blur_width = oversampling * np.maximum(columns.lpx, min_blur_width)
        blur_height = oversampling * np.maximum(columns.lpy, min_blur_width)
        # lpz carries its fallback from extraction already
        blur_depth = oversampling * np.maximum(columns.lpz, min_blur_width)

        sy = (blur_height[in_view] + blur_width[in_view]) / 2
        sx = sy
        sz = blur_depth[in_view]

        # isotropic in-plane blur: per-loc rotation about z has no effect
        rot_matrix = np.ascontiguousarray(
            to_rotation(ang).as_matrix(), dtype=np.float32
        )
        _fill_gaussian_rot(
            image, x, y, sx, sy, sz, n_pixel_x, n_pixel_y, rot_matrix
        )

    return len(x), image


def _render_convolve(
    columns: _RenderColumns,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations with with global localization precision,
    i.e. each localization is blurred by the median localization
    precision in x and y.

    Parameters
    ----------
    columns : _RenderColumns
        Column arrays of the localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    min_blur_width : float
        Minimum localization precision (camera pixels).
    ang : tuple or scipy.spatial.transform.Rotation, optional
        Rotation of localizations; either a scipy Rotation (e.g. built
        from a quaternion) or a tuple of 3 rotation angles around the
        x, y and z axes in radians (legacy Euler convention, see
        ``rotation_matrix``). If None, localizations are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        columns.x,
        columns.y,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )
    if ang is not None:  # rotate
        x, y, in_view, _ = _locs_rotation_arrays(
            columns,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )

    n = len(x)
    if n == 0:
        return 0, image
    else:
        _fill(image, x, y)
        blur_width = oversampling * max(
            np.median(columns.lpx[in_view]), min_blur_width
        )
        blur_height = oversampling * max(
            np.median(columns.lpy[in_view]), min_blur_width
        )
        return n, _fftconvolve(image, blur_width, blur_height)


def _render_smooth(
    columns: _RenderColumns,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    ang: tuple[float, float, float] | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations with with blur of one display pixel (set by
    oversampling).

    Parameters
    ----------
    columns : _RenderColumns
        Column arrays of the localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    ang : tuple or scipy.spatial.transform.Rotation, optional
        Rotation of localizations; either a scipy Rotation (e.g. built
        from a quaternion) or a tuple of 3 rotation angles around the
        x, y and z axes in radians (legacy Euler convention, see
        ``rotation_matrix``). If None, localizations are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        columns.x,
        columns.y,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    if ang is not None:  # rotate
        x, y, _, _ = _locs_rotation_arrays(
            columns,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )

    n = len(x)
    if n == 0:
        return 0, image
    else:
        _fill(image, x, y)
        return n, _fftconvolve(image, 1, 1)


def _fftconvolve(
    image: lib.FloatArray2D,
    blur_width: float,
    blur_height: float,
) -> lib.FloatArray2D:
    """Blur (convolves) 2D image using fast fourier transform or with
    Gaussian filter applied (faster for small kernels).

    Parameters
    ----------
    image : lib.FloatArray2D
        Image with rendered but not blurred localizations.
    blur_width, blur_height : float
        Blur width and height in pixels.

    Returns
    -------
    image : lib.FloatArray2D
        Blurred image.
    """
    kernel_width = 10 * int(np.round(blur_width)) + 1
    kernel_height = 10 * int(np.round(blur_height)) + 1
    # Spatial separable convolution is faster than FFT for the small
    # kernels typical of SMLM precisions (~1-3 px). Switch to FFT only
    # when the kernel is large relative to the image.
    n_y, n_x = image.shape
    spatial = (
        kernel_height < 0.05 * n_y
        and kernel_width < 0.05 * n_x
        and max(kernel_height, kernel_width) <= 101
    )
    if spatial:
        out = np.empty_like(image, dtype=np.float32)
        ndimage.gaussian_filter(
            image,
            sigma=(blur_height, blur_width),
            output=out,
            mode="constant",
            cval=0.0,
            truncate=5.0,
        )
        return out
    kernel_y = signal.windows.gaussian(kernel_height, blur_height)
    kernel_x = signal.windows.gaussian(kernel_width, blur_width)
    kernel = np.outer(kernel_y, kernel_x)
    kernel /= kernel.sum()
    image = signal.fftconvolve(image, kernel, mode="same")
    return image.astype(np.float32)


def locs_rotation(
    locs: pd.DataFrame,
    oversampling: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    ang: tuple[float, float, float] | Rotation,
) -> tuple[
    lib.FloatArray1D, lib.FloatArray1D, lib.BoolArray1D, lib.FloatArray1D
]:
    """Rotate localizations within a FOV.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rotated.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinate to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinate to be rendered (camera pixels).
    ang : tuple or scipy.spatial.transform.Rotation
        Rotation of localizations; either a scipy Rotation or a tuple
        of 3 rotation angles around the x, y and z axes in radians
        (legacy Euler convention, see ``rotation_matrix``).

    Returns
    -------
    x : lib.FloatArray1D
        New (rotated) x coordinates
    y : lib.FloatArray1D
        New y coordinates
    in_view : lib.BoolArray1D
        Indeces of locs that are rendered
    z : lib.FloatArray1D
        New z coordinates
    """
    return _locs_rotation_arrays(
        _extract_render_columns(locs, None, ang),
        oversampling,
        x_min,
        x_max,
        y_min,
        y_max,
        ang,
    )


def _locs_rotation_arrays(
    columns: _RenderColumns,
    oversampling: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    ang: tuple[float, float, float] | Rotation,
) -> tuple[
    lib.FloatArray1D, lib.FloatArray1D, lib.BoolArray1D, lib.FloatArray1D
]:
    """``locs_rotation`` on pre-extracted column arrays."""
    # z is translated to pixels
    locs_coord = np.column_stack((columns.x, columns.y, columns.z))

    # x and y are in range (x_min/y_min, x_max/y_max) so they need to be
    # shifted (scipy rotation is around origin)
    locs_coord[:, 0] -= x_min + (x_max - x_min) / 2
    locs_coord[:, 1] -= y_min + (y_max - y_min) / 2

    # rotate locs
    R = to_rotation(ang)
    locs_coord = R.apply(locs_coord)

    # unshift locs
    locs_coord[:, 0] += x_min + (x_max - x_min) / 2
    locs_coord[:, 1] += y_min + (y_max - y_min) / 2

    # output
    x = locs_coord[:, 0]
    y = locs_coord[:, 1]
    z = locs_coord[:, 2]
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    z = z[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    z *= oversampling
    return x, y, in_view, z


#: Blur methods whose channels may be split into row chunks rendered
#: separately and summed ("pseudo-channels"). Per-localization methods
#: (hist, gaussian, gaussian_iso) are additive by construction. 'smooth'
#: is exact as well: its one-pixel blur kernel is identical for every
#: chunk and convolution is linear; the extra per-chunk image filters
#: are cheap next to the per-loc fill, and ``_MIN_CHUNK_LOCS`` keeps the
#: few-locs/big-image regime whole. 'convolve' must not be chunked: its
#: blur width is the median precision of the rendered locs, and chunk
#: medians differ from the global median.
_CHUNKABLE_BLUR_METHODS = (None, "gaussian", "gaussian_iso", "smooth")

#: Minimum rows per chunk: below this, per-task overhead (image
#: allocation and summing, ~1 ms) stops being negligible against the
#: fill work it parallelizes. Found empirically on real data.
_MIN_CHUNK_LOCS = 100_000


def _render_worker_budget() -> int:
    """Number of worker threads the render pool may use, from the
    ``Render`` section of the user settings file (see
    ``lib.n_workers``)."""
    return lib.n_workers(
        lib.RENDER_CPU_UTILIZATION_DEFAULT, settings_section="Render"
    )


def _chunk_tasks(
    n_locs_per_channel: list[int], budget: int
) -> list[tuple[int, int, int]]:
    """Split channels into ``(channel, start, stop)`` row-chunk tasks.

    Aims for about two tasks per worker across the whole render so the
    pool stays load-balanced regardless of channel-size skew, while
    keeping every chunk at least ``_MIN_CHUNK_LOCS`` rows.

    Parameters
    ----------
    n_locs_per_channel : list of int
        Number of localizations per channel.
    budget : int
        Worker threads available to the render pool.

    Returns
    -------
    tasks : list of (int, int, int)
        ``(channel index, start row, stop row)`` per task; the chunks of
        each channel tile it exactly, in ascending row order.
    """
    total = sum(n_locs_per_channel)
    target = max(_MIN_CHUNK_LOCS, -(-total // (2 * budget)))
    tasks = []
    for i, n in enumerate(n_locs_per_channel):
        k = min(
            max(1, n // _MIN_CHUNK_LOCS),
            max(1, -(-n // target)),
        )
        bounds = np.linspace(0, n, k + 1).round().astype(np.int64)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            tasks.append((i, int(start), int(stop)))
    return tasks


class CpuBackend(SplatBackend):
    """CPU reference splat backend: the render thread pool.

    For per-localization blur methods (``_CHUNKABLE_BLUR_METHODS``),
    channels are additionally split into row chunks rendered as
    independent tasks whose images are summed per channel — splatting is
    additive, so any row partition yields the same image up to float
    summation order. One flat task pool covers all channels and chunks,
    so the load stays balanced regardless of channel-size skew and a
    single large channel also renders in parallel. Chunk images are
    summed in fixed row order, making the result deterministic for a
    given worker budget; with a budget of 1 the exact legacy sequential
    path runs. Threads give real parallelism because the fill kernels
    release the GIL (``nogil=True``) and never mutate their inputs. The
    pool only lives for the duration of one render, keeping Picasso
    polite on shared workstations; a single channel too small to chunk
    renders on the calling thread without reading the settings file.

    Safe to call concurrently: every render builds its own task list
    and pool, and the kernels never mutate their inputs.
    """

    name = "cpu"

    def render_channels(
        self,
        columns: list[_RenderColumns],
        info: list[list[dict]],
        *,
        disp_px_size: float,
        viewport: tuple[tuple[float, float], tuple[float, float]] | None,
        blur_method: (
            Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
        ),
        min_blur_width: float,
        ang: tuple | Rotation | None,
    ) -> list[tuple[int, lib.FloatArray2D]]:
        """See ``backend.SplatBackend.render_channels``."""

        def render_rows(i: int, start: int, stop: int):
            chunk = columns[i]
            if stop - start < len(chunk):
                chunk = chunk.slice(start, stop)
            return _render_arrays(
                chunk,
                info[i],
                disp_px_size=disp_px_size,
                viewport=viewport,
                blur_method=blur_method,
                min_blur_width=min_blur_width,
                ang=ang,
            )

        n_channels = len(columns)
        sizes = [len(channel) for channel in columns]
        total = sum(sizes)
        chunkable = blur_method in _CHUNKABLE_BLUR_METHODS
        # a single channel too small to chunk renders on the calling
        # thread without touching the settings file
        if n_channels == 1 and (not chunkable or total < 2 * _MIN_CHUNK_LOCS):
            return [render_rows(0, 0, total)]

        budget = _render_worker_budget()
        if chunkable and budget > 1:
            tasks = _chunk_tasks(sizes, budget)
        else:
            tasks = [(i, 0, n) for i, n in enumerate(sizes)]
        n_workers = min(len(tasks), budget)

        if n_workers == 1:
            chunk_results = [render_rows(*task) for task in tasks]
        else:
            # dispatch biggest tasks first so none serializes the tail
            # of the pool; results are mapped back to task order
            order = sorted(
                range(len(tasks)),
                key=lambda j: tasks[j][2] - tasks[j][1],
                reverse=True,
            )
            chunk_results = [None] * len(tasks)
            with futures.ThreadPoolExecutor(n_workers) as executor:
                for j, result in zip(
                    order,
                    executor.map(lambda j: render_rows(*tasks[j]), order),
                ):
                    chunk_results[j] = result

        # sum each channel's chunk images in fixed row order, so a given
        # worker budget always produces the same float rounding
        per_channel = [[] for _ in range(n_channels)]
        for (i, _, _), result in zip(tasks, chunk_results):
            per_channel[i].append(result)
        renderings = []
        for results in per_channel:
            n = sum(result[0] for result in results)
            image = results[0][1]
            for _, other in results[1:]:
                image += other
            renderings.append((n, image))
        return renderings
