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

from typing import Literal

import numba
import numpy as np
import pandas as pd
from scipy import signal, ndimage
from scipy.spatial.transform import Rotation

from .. import lib
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
    ang: tuple | Rotation | None = None,
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
    ang : tuple or scipy.spatial.transform.Rotation, optional
        Rotation of locs; either a scipy Rotation (e.g. built from a
        quaternion) or a tuple of 3 rotation angles around the x, y
        and z axes in radians (legacy Euler convention, see
        ``rotation_matrix``). If None, locs are not rotated.

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
        _extract_render_columns(locs, blur_method, ang),
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

    __slots__ = ("x", "y", "lpx", "lpy", "lpz", "angle", "z")

    def __init__(self, x, y, lpx=None, lpy=None, lpz=None, angle=None, z=None):
        self.x = x
        self.y = y
        self.lpx = lpx
        self.lpy = lpy
        self.lpz = lpz
        self.angle = angle
        self.z = z

    def __len__(self) -> int:
        return len(self.x)

    def slice(self, start: int, stop: int) -> "_RenderColumns":
        """Row range as array views (no copies)."""

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


def _extract_render_columns(
    locs: pd.DataFrame,
    blur_method: str | None,
    ang: tuple | Rotation | None,
) -> _RenderColumns:
    """Pull the columns ``blur_method`` (and rotation) needs out of the
    DataFrame, converting angle to radians and applying the lpz
    fallback once per channel."""
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
    return _RenderColumns(
        locs["x"].to_numpy(), locs["y"].to_numpy(), lpx, lpy, lpz, angle, z
    )


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
