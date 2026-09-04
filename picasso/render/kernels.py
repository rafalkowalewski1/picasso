"""
picasso.render.kernels
~~~~~~~~~~~~~~~~~~~~~~

Numba-compiled CPU kernels for the render package: splat setup and
drawing, histogram filling and the fused image-composition kernels.

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numba
import numpy as np

from .. import lib


_DRAW_MAX_SIGMA = 3  # max. sigma from mean to render (mu +/- 3 sigma)


@numba.njit(nogil=True)
def _render_setup(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
) -> tuple[
    lib.FloatArray2D,
    int,
    int,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.BoolArray1D,
]:
    """Find coordinates to be rendered and sets up an empty image
    array.

    Parameters
    ----------
    x, y : lib.FloatArray1D
        x and y coordinates of the localizations to be rendered (1D
        arrays).
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).

    Returns
    -------
    image : lib.FloatArray2D
        Empty image array.
    n_pixel_y : int
        Number of pixels in y.
    n_pixel_x : int
        Number of pixels in x.
    x : lib.FloatArray1D
        x coordinates to be rendered.
    y : lib.FloatArray1D
        y coordinates to be rendered.
    in_view : lib.BoolArray1D
        Indeces of the localizations to be rendered.
    """
    n_pixel_y = int(np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    image = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    return image, n_pixel_y, n_pixel_x, x, y, in_view


@numba.njit(nogil=True)
def _render_setup_anisotropic(  # used in Average
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    oversampling_x: float,
    oversampling_y: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
) -> tuple[
    lib.FloatArray2D,
    int,
    int,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.BoolArray1D,
]:
    """Find coordinates to be rendered and sets up an empty image
    array. Allows for different pixel sizes in x and y (oversampling).

    Parameters
    ----------
    x, y : lib.FloatArray1D
        x and y coordinates of the localizations to be rendered (1D
        arrays).
    oversampling_x, oversampling_y : float
        Number of super-resolution pixels per camera pixel in x and y.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).

    Returns
    -------
    image : lib.FloatArray2D
        Empty image array.
    n_pixel_y : int
        Number of pixels in y.
    n_pixel_x : int
        Number of pixels in x.
    x : lib.FloatArray1D
        x coordinates to be rendered.
    y : lib.FloatArray1D
        y coordinates to be rendered.
    in_view : lib.BoolArray1D
        Indeces of the localizations to be rendered.
    """
    n_pixel_y = int(np.ceil(oversampling_y * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling_x * (x_max - x_min)))
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling_x * (x - x_min)
    y = oversampling_y * (y - y_min)
    image = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    return image, n_pixel_y, n_pixel_x, x, y, in_view


@numba.njit(nogil=True)
def _render_setup3d(
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
) -> tuple[
    lib.FloatArray3D,
    int,
    int,
    int,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.BoolArray1D,
]:
    """Find coordinates to be rendered in 3D and sets up an empty image
    array.

    Parameters
    ----------
    x, y, z : lib.FloatArray1D
        x, y and z coordinates of the localizations to be rendered (1D
        arrays).
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinate to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinate to be rendered (camera pixels).
    z_min : float
        Minimum z coordinate to be rendered (nm).
    z_max : float
        Maximum z coordinate to be rendered (nm).
    pixelsize : float
        Camera pixel size, used for converting z coordinates.

    Returns
    -------
    image : lib.FloatArray3D
        Empty image array.
    n_pixel_y, n_pixel_x, n_pixel_z : int
        Number of pixels in y, x, and z.
    x, y, z : lib.FloatArray1D
        x, y, z coordinates to be rendered.
    in_view : lib.BoolArray1D
        Indeces of the localizations to be rendered.
    """
    n_pixel_y = int(np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
    n_pixel_z = int(np.ceil(oversampling * (z_max - z_min)))
    # divide on a copy -- rendering must never mutate the caller's arrays
    # (see TestRenderPurity in test_render.py)
    z = z.copy()
    z /= pixelsize
    in_view = (
        (x > x_min)
        & (y > y_min)
        & (z > z_min)
        & (x < x_max)
        & (y < y_max)
        & (z < z_max)
    )
    x = x[in_view]
    y = y[in_view]
    z = z[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    z = oversampling * (z - z_min)
    image = np.zeros((n_pixel_y, n_pixel_x, n_pixel_z), dtype=np.float32)
    return image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view


@numba.njit(nogil=True)
def _render_setup3d_anisotropic(
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
) -> tuple[
    lib.FloatArray3D,
    int,
    int,
    int,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.BoolArray1D,
]:
    """Find coordinates to be rendered in 3D and sets up an empty image
    array. Allows for different pixel sizes in x, y and z
    (oversampling).

    Parameters
    ----------
    x, y, z : lib.FloatArray1D
        x, y and z coordinates of the localizations to be rendered (1D
        arrays).
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinate to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinate to be rendered (camera pixels).
    z_min : float
        Minimum z coordinate to be rendered (nm).
    z_max : float
        Maximum z coordinate to be rendered (nm).
    pixelsize : float
        Camera pixel size, used for converting z coordinates.

    Returns
    -------
    image : lib.FloatArray3D
        Empty image array.
    n_pixel_y, n_pixel_x, n_pixel_z : int
        Number of pixels in y, x, and z.
    x, y, z : lib.FloatArray1D
        x, y, z coordinates to be rendered.
    in_view : lib.BoolArray1D
        Indeces of the localizations to be rendered.
    """
    n_pixel_y = int(np.ceil(oversampling_y * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling_x * (x_max - x_min)))
    n_pixel_z = int(np.ceil(oversampling_z * (z_max - z_min)))
    # divide on a copy -- rendering must never mutate the caller's arrays
    # (see TestRenderPurity in test_render.py)
    z = z.copy()
    z /= pixelsize
    in_view = (
        (x > x_min)
        & (y > y_min)
        & (z > z_min)
        & (x < x_max)
        & (y < y_max)
        & (z < z_max)
    )
    x = x[in_view]
    y = y[in_view]
    z = z[in_view]
    x = oversampling_x * (x - x_min)
    y = oversampling_y * (y - y_min)
    z = oversampling_z * (z - z_min)
    image = np.zeros((n_pixel_y, n_pixel_x, n_pixel_z), dtype=np.float32)
    return image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view


@numba.njit(nogil=True)
def _fill(
    image: lib.FloatArray2D, x: lib.FloatArray1D, y: lib.FloatArray1D
) -> None:
    """Fill image with x and y coordinates. Image is not blurred.

    Parameters
    ----------
    image : lib.FloatArray2D
        Empty image array.
    x, y : lib.FloatArray1D
        x and y coordinates to be rendered.
    """
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    for i, j in zip(x, y):
        image[j, i] += 1


@numba.njit(nogil=True)
def _fill3d(
    image: lib.FloatArray3D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D,
) -> None:
    """Fill image with x, y and z coordinates. Image is not blurred.

    Parameters
    ----------
    image : lib.FloatArray3D
        Empty image array.
    x, y, z : lib.FloatArray1D
        x, y and z coordinates to be rendered.
    """
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    z = z.astype(np.int32)
    for i, j, k in zip(x, y, z):
        image[j, i, k] += 1


@numba.njit(cache=True, nogil=True)
def _draw_gaussian_loc(
    image: lib.FloatArray2D,
    x_: float,
    y_: float,
    sx_: float,
    sy_: float,
    n_pixel_x: int,
    n_pixel_y: int,
) -> None:
    """Render a single separable 2D Gaussian into ``image``."""
    if not (sx_ > 0.0 and sy_ > 0.0):
        # Degenerate localization (e.g. lpx/lpy of exactly 0 from a
        # singular CRLB fit); also catches NaN. Skip instead of
        # dividing by zero.
        return
    max_y_off = _DRAW_MAX_SIGMA * sy_
    i_min = np.int32(y_ - max_y_off)
    if i_min < 0:
        i_min = 0
    i_max = np.int32(y_ + max_y_off + 1)
    if i_max > n_pixel_y:
        i_max = n_pixel_y
    max_x_off = _DRAW_MAX_SIGMA * sx_
    j_min = np.int32(x_ - max_x_off)
    if j_min < 0:
        j_min = 0
    j_max = np.int32(x_ + max_x_off) + 1
    if j_max > n_pixel_x:
        j_max = n_pixel_x
    nx = j_max - j_min
    ny = i_max - i_min
    if nx <= 0 or ny <= 0:
        return
    inv_2sx2 = 1.0 / (2.0 * sx_ * sx_)
    inv_2sy2 = 1.0 / (2.0 * sy_ * sy_)
    norm = 1.0 / (2.0 * np.pi * sx_ * sy_)
    # Separable kernel: factor exp(-(dx^2/(2sx^2) + dy^2/(2sy^2)))
    # into 1D gx * 1D gy. O(K) exp calls per loc instead of O(K^2).
    gx = np.empty(nx, dtype=np.float32)
    gy = np.empty(ny, dtype=np.float32)
    for jj in range(nx):
        dx = (j_min + jj) + 0.5 - x_
        gx[jj] = np.exp(-dx * dx * inv_2sx2)
    for ii in range(ny):
        dy = (i_min + ii) + 0.5 - y_
        gy[ii] = norm * np.exp(-dy * dy * inv_2sy2)
    for ii in range(ny):
        gy_i = gy[ii]
        row = image[i_min + ii]
        for jj in range(nx):
            row[j_min + jj] += gy_i * gx[jj]


@numba.njit(cache=True, nogil=True)
def _fill_gaussian(
    image: lib.FloatArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    sx: lib.FloatArray1D,
    sy: lib.FloatArray1D,
    n_pixel_x: int,
    n_pixel_y: int,
) -> None:
    """Fill image with blurred x and y coordinates. Each localization
    is rendered as a 2D Gaussian centered at (x, y) with standard
    deviations (sx, sy).

    Parameters
    ----------
    image : lib.FloatArray2D
        Empty image array.
    x, y : lib.FloatArray1D
        x and y coordinates to be rendered.
    sx, sy : lib.FloatArray1D
        Localization precision in x and y for each localization.
    n_pixel_x, n_pixel_y : int
        Number of pixels in x and y.
    """
    n_locs = len(x)
    if n_locs == 0:
        return

    for i in range(n_locs):
        _draw_gaussian_loc(
            image, x[i], y[i], sx[i], sy[i], n_pixel_x, n_pixel_y
        )


@numba.njit(cache=True, nogil=True)
def _draw_gaussian_theta_loc(
    image: lib.FloatArray2D,
    x_: float,
    y_: float,
    sx_: float,
    sy_: float,
    angle_: float,
    n_pixel_x: int,
    n_pixel_y: int,
) -> None:
    """Render a single in-plane rotated 2D Gaussian into ``image``.

    The elliptical Gaussian with standard deviations (``sx_``, ``sy_``)
    is rotated in the image plane by ``angle_`` (radians) via its 2x2
    covariance matrix. Unlike ``_draw_gaussian_loc`` the rotated kernel
    is not separable (it has a cross term), so pixels are evaluated with
    the full bivariate quadratic form, as in ``_draw_gaussian_rot_loc``.
    """
    c = np.cos(angle_)
    s = np.sin(angle_)
    vx = sx_ * sx_
    vy = sy_ * sy_
    cxx = vx * c * c + vy * s * s
    cyy = vx * s * s + vy * c * c
    cxy = (vx - vy) * s * c
    det2d = cxx * cyy - cxy * cxy
    if det2d < 1e-10:
        return
    inv_xx = cyy / det2d
    inv_yy = cxx / det2d
    inv_xy = -cxy / det2d
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det2d))
    max_x_off = _DRAW_MAX_SIGMA * np.sqrt(cxx)
    max_y_off = _DRAW_MAX_SIGMA * np.sqrt(cyy)
    j_min = int(x_ - max_x_off)
    if j_min < 0:
        j_min = 0
    j_max = int(x_ + max_x_off + 1)
    if j_max > n_pixel_x:
        j_max = n_pixel_x
    i_min = int(y_ - max_y_off)
    if i_min < 0:
        i_min = 0
    i_max = int(y_ + max_y_off + 1)
    if i_max > n_pixel_y:
        i_max = n_pixel_y
    for i in range(i_min, i_max):
        b = np.float32(i + 0.5 - y_)
        for j in range(j_min, j_max):
            a = np.float32(j + 0.5 - x_)
            exponent = a * a * inv_xx + 2.0 * a * b * inv_xy + b * b * inv_yy
            image[i, j] += norm * np.exp(-0.5 * exponent)


@numba.njit(cache=True, nogil=True)
def _fill_gaussian_theta(
    image: lib.FloatArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    sx: lib.FloatArray1D,
    sy: lib.FloatArray1D,
    angle: lib.FloatArray1D,
    n_pixel_x: int,
    n_pixel_y: int,
) -> None:
    """Fill image with in-plane rotated gaussian-blurred localizations.

    Each localization is rendered as a 2D Gaussian centered at (x, y)
    with standard deviations (sx, sy) rotated in the image plane by its
    own ``angle`` (radians).

    Parameters
    ----------
    image : lib.FloatArray2D
        Empty image array.
    x, y : lib.FloatArray1D
        x and y coordinates to be rendered.
    sx, sy : lib.FloatArray1D
        Localization precision in x and y for each localization.
    angle : lib.FloatArray1D
        In-plane rotation angle (radians) for each localization.
    n_pixel_x, n_pixel_y : int
        Number of pixels in x and y.
    """
    n_locs = len(x)
    if n_locs == 0:
        return

    for i in range(n_locs):
        _draw_gaussian_theta_loc(
            image, x[i], y[i], sx[i], sy[i], angle[i], n_pixel_x, n_pixel_y
        )


@numba.njit(cache=True, nogil=True)
def _draw_gaussian_cov3d_loc(
    image: lib.FloatArray2D,
    x_: float,
    y_: float,
    cov: lib.Array3x3,
    n_pixel_x: int,
    n_pixel_y: int,
    rot_matrix: lib.Array3x3,
    rot_matrixT: lib.Array3x3,
) -> None:
    """Render a single 3D Gaussian with local covariance ``cov`` into
    ``image``: ``cov`` is rotated by the global ``rot_matrix`` and the
    top-left 2x2 block is projected, inverted and drawn as a bivariate
    Gaussian. Shared by ``_draw_gaussian_rot_loc`` (diagonal ``cov``) and
    ``_draw_gaussian_rot_theta_loc`` (in-plane rotated ``cov``)."""
    cov_rot = rot_matrix @ cov @ rot_matrixT
    s00 = cov_rot[0, 0]
    s01 = cov_rot[0, 1]
    s10 = cov_rot[1, 0]
    s11 = cov_rot[1, 1]
    det2d = s00 * s11 - s01 * s10
    if det2d < 1e-10:
        return
    inv00 = s11 / det2d
    inv01 = -s01 / det2d
    inv10 = -s10 / det2d
    inv11 = s00 / det2d
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det2d))
    max_x_off = _DRAW_MAX_SIGMA * np.sqrt(s00)
    max_y_off = _DRAW_MAX_SIGMA * np.sqrt(s11)
    j_min = int(x_ - max_x_off)
    if j_min < 0:
        j_min = 0
    j_max = int(x_ + max_x_off + 1)
    if j_max > n_pixel_x:
        j_max = n_pixel_x
    i_min = int(y_ - max_y_off)
    if i_min < 0:
        i_min = 0
    i_max = int(y_ + max_y_off + 1)
    if i_max > n_pixel_y:
        i_max = n_pixel_y
    for i in range(i_min, i_max):
        b = np.float32(i + 0.5 - y_)
        for j in range(j_min, j_max):
            a = np.float32(j + 0.5 - x_)
            exponent = a * a * inv00 + a * b * (inv01 + inv10) + b * b * inv11
            image[i, j] += norm * np.exp(-0.5 * exponent)


@numba.njit(cache=True, nogil=True)
def _draw_gaussian_rot_loc(
    image: lib.FloatArray2D,
    x_: float,
    y_: float,
    sx_: float,
    sy_: float,
    sz_: float,
    n_pixel_x: int,
    n_pixel_y: int,
    rot_matrix: lib.Array3x3,
    rot_matrixT: lib.Array3x3,
) -> None:
    """Render a single rotated 2D Gaussian (projected from 3D) into
    ``image``."""
    cov = np.zeros((3, 3), dtype=np.float32)
    cov[0, 0] = sx_ * sx_
    cov[1, 1] = sy_ * sy_
    cov[2, 2] = sz_ * sz_
    _draw_gaussian_cov3d_loc(
        image, x_, y_, cov, n_pixel_x, n_pixel_y, rot_matrix, rot_matrixT
    )


@numba.njit(cache=True, nogil=True)
def _draw_gaussian_rot_theta_loc(
    image: lib.FloatArray2D,
    x_: float,
    y_: float,
    sx_: float,
    sy_: float,
    sz_: float,
    angle_: float,
    n_pixel_x: int,
    n_pixel_y: int,
    rot_matrix: lib.Array3x3,
    rot_matrixT: lib.Array3x3,
) -> None:
    """Render a single rotated 2D Gaussian (projected from 3D) into
    ``image``. The in-plane precision ellipse (``sx_``, ``sy_``) is
    first rotated by ``angle_`` (radians) about the z-axis, then the
    full 3D covariance is rotated by the global ``rot_matrix``."""
    c = np.cos(angle_)
    s = np.sin(angle_)
    vx = sx_ * sx_
    vy = sy_ * sy_
    cov = np.zeros((3, 3), dtype=np.float32)
    cov[0, 0] = vx * c * c + vy * s * s
    cov[1, 1] = vx * s * s + vy * c * c
    cov[0, 1] = (vx - vy) * c * s
    cov[1, 0] = cov[0, 1]
    cov[2, 2] = sz_ * sz_
    _draw_gaussian_cov3d_loc(
        image, x_, y_, cov, n_pixel_x, n_pixel_y, rot_matrix, rot_matrixT
    )


@numba.njit(cache=True, nogil=True)
def _fill_gaussian_rot(
    image: lib.FloatArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    sx: lib.FloatArray1D,
    sy: lib.FloatArray1D,
    sz: lib.FloatArray1D,
    n_pixel_x: int,
    n_pixel_y: int,
    rot_matrix: lib.Array3x3,
) -> None:
    """Fill image with rotated gaussian-blurred localizations.

    Localization precisions (sx, sy and sz) are treated as standard
    deviations of the gaussians to be rendered.

    Parameters
    ----------
    image : lib.FloatArray2D
        Empty image array.
    x, y, z : lib.FloatArray1D
        3D coordinates to be rendered.
    sx, sy, sz : lib.FloatArray1D
        Localization precision in x, y and z for each localization.
    n_pixel_x, n_pixel_y : int
        Number of pixels in x and y.
    rot_matrix : lib.Array3x3
        Rotation matrix (float32) applied to the localizations.
    """
    n_locs = len(x)
    if n_locs == 0:
        return
    rot_matrixT = np.ascontiguousarray(rot_matrix.T)

    for i in range(n_locs):
        _draw_gaussian_rot_loc(
            image,
            x[i],
            y[i],
            sx[i],
            sy[i],
            sz[i],
            n_pixel_x,
            n_pixel_y,
            rot_matrix,
            rot_matrixT,
        )


@numba.njit(cache=True, nogil=True)
def _fill_gaussian_rot_theta(
    image: lib.FloatArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    sx: lib.FloatArray1D,
    sy: lib.FloatArray1D,
    sz: lib.FloatArray1D,
    angle: lib.FloatArray1D,
    n_pixel_x: int,
    n_pixel_y: int,
    rot_matrix: lib.Array3x3,
) -> None:
    """Fill image with rotated gaussian-blurred localizations, each with
    its own in-plane rotation.

    Same as ``_fill_gaussian_rot`` but the precision ellipse (sx, sy) of
    every localization is first rotated in the image plane by its own
    ``angle`` (radians) about the z-axis, before the global rotation.

    Parameters
    ----------
    image : lib.FloatArray2D
        Empty image array.
    x, y, z : lib.FloatArray1D
        3D coordinates to be rendered.
    sx, sy, sz : lib.FloatArray1D
        Localization precision in x, y and z for each localization.
    angle : lib.FloatArray1D
        In-plane rotation angle (radians) for each localization,
        applied about the z-axis before the global ``rot_matrix``.
    n_pixel_x, n_pixel_y : int
        Number of pixels in x and y.
    rot_matrix : lib.Array3x3
        Rotation matrix (float32) applied to the localizations.
    """
    n_locs = len(x)
    if n_locs == 0:
        return
    rot_matrixT = np.ascontiguousarray(rot_matrix.T)

    for i in range(n_locs):
        _draw_gaussian_rot_theta_loc(
            image,
            x[i],
            y[i],
            sx[i],
            sy[i],
            sz[i],
            angle[i],
            n_pixel_x,
            n_pixel_y,
            rot_matrix,
            rot_matrixT,
        )


@numba.njit(nogil=True)
def inverse_3x3(a: lib.Array3x3) -> lib.Array3x3:
    """Calculate inverse of a 3x3 matrix. This function is faster than
    ``np.linalg.inv``.

    Parameters
    ----------
    a : lib.Array3x3
        3x3 matrix.

    Returns
    -------
    c : lib.Array3x3
        Inverse of ``a``.
    """
    c = np.zeros((3, 3), dtype=np.float32)
    det = determinant_3x3(a)

    c[0, 0] = (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]) / det
    c[0, 1] = (a[0, 2] * a[2, 1] - a[0, 1] * a[2, 2]) / det
    c[0, 2] = (a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1]) / det

    c[1, 0] = (a[1, 2] * a[2, 0] - a[1, 0] * a[2, 2]) / det
    c[1, 1] = (a[0, 0] * a[2, 2] - a[0, 2] * a[2, 0]) / det
    c[1, 2] = (a[0, 2] * a[1, 0] - a[0, 0] * a[1, 2]) / det

    c[2, 0] = (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]) / det
    c[2, 1] = (a[0, 1] * a[2, 0] - a[0, 0] * a[2, 1]) / det
    c[2, 2] = (a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]) / det

    return c


@numba.njit(nogil=True)
def determinant_3x3(a: lib.Array3x3) -> np.float32:
    """Calculate determinant of a 3x3 matrix. This function is faster
    than ``np.linalg.det``.

    Parameters
    ----------
    a : lib.Array3x3
        3x3 matrix.

    Returns
    -------
    det : np.float32
        Determinant of ``a``.
    """
    det = np.float32(
        a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
        - a[0, 1] * (a[1, 0] * a[2, 2] - a[2, 0] * a[1, 2])
        + a[0, 2] * (a[1, 0] * a[2, 1] - a[2, 0] * a[1, 1])
    )
    return det


@numba.jit(nopython=True, nogil=True)
def render_hist_numba(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    oversampling: float,
    t_min: float,
    t_max: float,
) -> tuple[int, lib.FloatArray2D]:
    """Calculate 2D histogram of xy coordinates. Similar to
    ``_render_hist`` but modified to work with numba.

    Parameters
    ----------
    x, y : lib.FloatArray1D
        1D arrays of xy coordinates.
    oversampling : float
        Number of histogram pixels per camera pixel.
    t_min, t_max : float
        Minimum and maximum bounds of the histogram.

    Returns
    -------
    n : int
        Number of localizations in the histogram.
    image : lib.FloatArray2D
        2D histogram of xy coordinates.
    """
    n_pixel = int(np.ceil(oversampling * (t_max - t_min)))
    in_view = (x > t_min) & (y > t_min) & (x < t_max) & (y < t_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - t_min)
    y = oversampling * (y - t_min)
    image = np.zeros((n_pixel, n_pixel), dtype=np.float32)
    _fill(image, x, y)
    return len(x), image


@numba.njit(cache=True, nogil=True)
def _compose_multi_lut(
    raw: lib.FloatArray3D,
    luts: lib.FloatArray3D,
    vmin: float,
    vmax: float,
    rel: lib.FloatArray1D,
    bg: lib.FloatArray1D,
    has_bg: bool,
) -> tuple[lib.FloatArray3D, np.float32]:
    """Fused replacement for the multi-channel numpy post-processing
    chain (contrast scale -> intensity scale -> LUT gather -> additive
    blend -> background compositing), reading the raw stack once.

    Reproduces the chain's float32 arithmetic step for step (subtract
    then divide, non-finite to zero, clip, truncating int cast for the
    LUT index) so results match the legacy path bit-for-bit up to
    quantization. Returns the float RGB image plus its global maximum,
    which ``_quantize_rgb`` needs for ``to_8bit``'s renormalization.
    """
    n_channels, n_y, n_x = raw.shape
    vmin32 = np.float32(vmin)
    rng32 = np.float32(vmax - vmin)
    one = np.float32(1.0)
    zero = np.float32(0.0)
    rgb = np.empty((n_y, n_x, 3), dtype=np.float32)
    max_value = zero
    for i in range(n_y):
        for j in range(n_x):
            r = zero
            g = zero
            b = zero
            coverage = zero
            for c in range(n_channels):
                v = (raw[c, i, j] - vmin32) / rng32
                if not np.isfinite(v):
                    v = zero
                if v < zero:
                    v = zero
                elif v > one:
                    v = one
                v = v * rel[c]
                idx = np.int32(v * np.float32(255.0))
                if idx < 0:
                    idx = 0
                elif idx > 255:
                    idx = 255
                r += luts[c, idx, 0]
                g += luts[c, idx, 1]
                b += luts[c, idx, 2]
                coverage += v
            if r > one:
                r = one
            if g > one:
                g = one
            if b > one:
                b = one
            if has_bg:
                if coverage < zero:
                    coverage = zero
                elif coverage > one:
                    coverage = one
                remainder = one - coverage
                r += bg[0] * remainder
                g += bg[1] * remainder
                b += bg[2] * remainder
                if r > one:
                    r = one
                if g > one:
                    g = one
                if b > one:
                    b = one
            rgb[i, j, 0] = r
            rgb[i, j, 1] = g
            rgb[i, j, 2] = b
            if r > max_value:
                max_value = r
            if g > max_value:
                max_value = g
            if b > max_value:
                max_value = b
    return rgb, max_value


@numba.njit(cache=True, nogil=True)
def _quantize_rgb(
    rgb: lib.FloatArray3D, max_value: np.float32
) -> lib.IntArray3D:
    """``to_8bit`` as a kernel: divide by the global maximum (when
    positive) and round to uint8, matching numpy's half-to-even."""
    n_y, n_x, _ = rgb.shape
    denom = max_value if max_value > np.float32(0.0) else np.float32(1.0)
    out = np.empty((n_y, n_x, 3), dtype=np.uint8)
    for i in range(n_y):
        for j in range(n_x):
            for k in range(3):
                t = (rgb[i, j, k] / denom) * np.float32(255.0)
                out[i, j, k] = np.uint8(np.rint(t))
    return out


@numba.njit(cache=True, nogil=True)
def _compose_single(
    raw: lib.FloatArray2D,
    cmap: lib.IntArray2D,
    vmin: float,
    vmax: float,
) -> lib.IntArray3D:
    """Fused replacement for the single-channel chain
    (``scale_contrast`` -> ``to_8bit`` -> ``apply_colormap``): contrast
    scale with clipping, renormalize by the global maximum, round to the
    256-entry colormap index and gather RGB."""
    n_y, n_x = raw.shape
    vmin32 = np.float32(vmin)
    rng32 = np.float32(vmax - vmin)
    one = np.float32(1.0)
    zero = np.float32(0.0)
    scaled = np.empty((n_y, n_x), dtype=np.float32)
    max_value = zero
    for i in range(n_y):
        for j in range(n_x):
            v = (raw[i, j] - vmin32) / rng32
            if not np.isfinite(v):
                v = zero
            if v < zero:
                v = zero
            elif v > one:
                v = one
            scaled[i, j] = v
            if v > max_value:
                max_value = v
    denom = max_value if max_value > zero else one
    out = np.empty((n_y, n_x, 3), dtype=np.uint8)
    for i in range(n_y):
        for j in range(n_x):
            t = (scaled[i, j] / denom) * np.float32(255.0)
            idx = np.int64(np.rint(t))
            out[i, j, 0] = cmap[idx, 0]
            out[i, j, 1] = cmap[idx, 1]
            out[i, j, 2] = cmap[idx, 2]
    return out
