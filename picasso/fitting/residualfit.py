"""
picasso.fitting.residualfit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-emitter fitting of whole frames: find the peaks in what the current
fit leaves unexplained, fit them, and repeat - the 3D-DAOSTORM analysis of
Babcock, Sigal & Zhuang (2012), ported from its maintained reference
implementation, storm-analysis (v2.4.1, model ``2dfixed``, maximum
likelihood).

Every frame is analyzed on its own, in cycles:

1. The background is a broad Gaussian low pass (``background_sigma``) of
   the frame minus the current fit.
2. Peaks are found in the residual - frame minus background minus fit -
   smoothed with the PSF and divided by its expected noise, i.e., in units
   of the signal-to-noise ratio: local maxima above ``threshold`` within
   ``peak_radius``. A pixel is returned at most once per frame.
3. The new peaks join the fit. Each peak is fitted on its own, with every
   other peak held fixed in the model, by Levenberg-Marquardt steps on the
   Poisson deviance of its region (a disc of ``roi_size`` pixels). Peaks
   near a new peak or near a removed one are fitted again, until all have
   converged.
4. Of two peaks closer than the PSF width, the dimmer one is removed, and
   so is every peak whose significance is not above ``threshold``.

The cycles stop at the first cycle that finds no new peak, or after
``n_cycles``; peaks that did not converge are dropped. (storm-analysis
cycles on until its threshold schedule has run down, at least five
cycles, but a cycle without new peaks changes nothing, so the results are
the same.) The model of a pixel is the sum of the photons of every peak
whose region covers it plus the mean of their backgrounds, so the
background is counted once, however many peaks overlap.

Picasso's port differs from storm-analysis in three ways. The PSF is the
Gaussian integrated over each pixel, with the photon count as its
amplitude, instead of a Gaussian sampled at the pixel centers with the
peak height as its amplitude; the filters are separable convolutions with
wrap-around edges on the frame padded by mirroring, where storm-analysis
convolves by FFT; and every emitter gets a Cramer-Rao bound from the
Fisher matrix of itself and its neighbors, which storm-analysis does not
compute.

This module is documented for reference only and is not part of the
public API; :func:`picasso.localize.localize_multi_emitter` runs it.

References
----------
Babcock, H., Sigal, Y. M. & Zhuang, X. "A high-density 3D localization
algorithm for stochastic optical reconstruction microscopy." Optical
Nanoscopy 1, 6 (2012). https://doi.org/10.1186/2192-2853-1-6

storm-analysis (Zhuang Lab, Harvard University), v2.4.1:
``sa_library/fitting.py``, ``sa_library/multi_fit.c``,
``sa_library/dao_fit.c``, ``sa_library/ia_utilities.c`` and
``daostorm_3d/find_peaks.py``. https://github.com/ZhuangLab/storm-analysis
License (MIT): ``LICENSES/storm-analysis-LICENSE.txt``.

Huang, F., Schwartz, S. L., Byars, J. M. & Lidke, K. A. "Simultaneous
multiple-emitter fitting for single molecule super-resolution imaging."
Biomed. Opt. Express 2, 1377-1393 (2011).
https://doi.org/10.1364/BOE.2.001377 (the Fisher matrix correction for
close emitters, Eq. 11)

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np

from picasso.fitting.gaussfit import (
    _estimator_terms,
    _lm_accept_or_reject,
    _lm_finalize,
    _lm_reject_step,
)
from picasso.fitting.splinefit import (
    _FASTMATH,
    _LAMBDA_INITIAL,
    _lm_solve_step,
    FIT_STATE_CONVERGED,
    FIT_STATE_NEG_CURVATURE_MLE,
    FIT_STATE_SINGULAR_HESSIAN,
)

__all__ = [
    "THRESHOLD",
    "N_CYCLES",
    "PEAK_RADIUS",
    "BACKGROUND_SIGMA",
    "roi_size",
    "padding",
    "FrameResult",
    "analyze_frame",
    "find_peaks",
    "fit_free_sigma",
]

#: Default peak threshold, in units of the signal-to-noise ratio, and the
#: minimum significance a peak keeps ("6 sigma").
THRESHOLD = 6.0
#: Default number of find-fit-subtract cycles per frame.
N_CYCLES = 20
#: Default radius (camera pixels) within which a peak must be the maximum.
PEAK_RADIUS = 5.0
#: Default width (camera pixels) of the Gaussian that estimates the
#: background.
BACKGROUND_SIGMA = 8.0

# Peak status (ia_utilities_c.py, multi_fit.h).
_RUNNING = 0
_CONVERGED = 1
_ERROR = 2

# Levenberg-Marquardt damping, per peak (multi_fit.h).
_LAMBDA_START = 1.0
_LAMBDA_DOWN = 0.75
_LAMBDA_MAX = 1.0e20
_LAMBDA_MIN = 1.0e-3
_LAMBDA_UP = 4.0
# Relative change of a peak's error below which it has converged
# (MultiFitter.default_tol).
_TOLERANCE = 1.0e-6
# Most fitting sweeps over the running peaks per call (MultiFitter.doFit).
_MAX_FIT_ITERATIONS = 2000
# Peaks closer than this many PSF widths to a new or a removed peak are
# fitted again (PeakFinderFitter.unconverged_dist).
_NEIGHBORHOOD = 5.0
# Floor of the variance the residual is divided by (PeakFinderGaussian).
_VARIANCE_FLOOR = 0.1
# Filter kernels reach this many standard deviations.
_KERNEL_TRUNCATE = 4.0
# The region of a peak is a disc of int(8 sigma) pixels (PeakFinderGaussian).
_ROI_SIZE_FACTOR = 8.0
# Emitters closer than this many PSF widths are fitted jointly in the
# Cramer-Rao bound and counted in ``n_emitters``.
_NEIGHBOR_DISTANCE = 4.0
# Most neighbors (nearest first) in an emitter's joint Fisher matrix.
_MAX_NEIGHBORS = 8

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def roi_size(sigma: float) -> int:
    """Side length (camera pixels) of the disc a peak is fitted in.

    Parameters
    ----------
    sigma : float
        PSF standard deviation in camera pixels.

    Returns
    -------
    roi_size : int
        ``int(8 * sigma)``, as storm-analysis uses for a fixed width.
    """
    return int(_ROI_SIZE_FACTOR * sigma)


def padding(sigma: float) -> int:
    """Margin (camera pixels) by which a frame is padded, by mirroring.

    Parameters
    ----------
    sigma : float
        PSF standard deviation in camera pixels.

    Returns
    -------
    padding : int
        ``int(roi_size / 2 + 2)``: the region of a peak on the edge of the
        frame stays within the padded frame.
    """
    return int(roi_size(sigma) / 2 + 2)


# ----------------------------------------------------------------------
# Filters (PeakFinderGaussian)
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True)
def _gaussian_kernel(sigma: float, size: int, square: bool) -> np.ndarray:
    """Normalized 1D Gaussian kernel, or its square (for the variance of a
    smoothed image), reaching ``_KERNEL_TRUNCATE`` standard deviations but
    no further than half the image."""
    radius = int(math.ceil(_KERNEL_TRUNCATE * sigma))
    radius = max(0, min(radius, (size - 1) // 2))
    kernel = np.empty(2 * radius + 1)
    total = 0.0
    for i in range(-radius, radius + 1):
        value = math.exp(-0.5 * i * i / (sigma * sigma))
        kernel[i + radius] = value
        total += value
    for i in range(2 * radius + 1):
        kernel[i] /= total
        if square:
            kernel[i] *= kernel[i]
    return kernel


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _convolve(image, kernel, out, tmp):
    """Separable convolution with wrap-around edges: the circular
    convolution storm-analysis runs by FFT on its padded frames."""
    height, width = image.shape
    radius = len(kernel) // 2
    # The kernel taps are the outer loops and the pixels of a row the
    # inner ones, which run over contiguous memory; every pixel still sums
    # its taps in kernel order. Along x, each row is first extended by the
    # columns it wraps around to.
    row = np.empty(width + 2 * radius)
    for y in range(height):
        for x in range(width + 2 * radius):
            xx = x - radius
            if xx < 0:
                xx += width
            elif xx >= width:
                xx -= width
            row[x] = image[y, xx]
        for x in range(width):
            tmp[y, x] = 0.0
        for k in range(2 * radius + 1):
            weight = kernel[k]
            for x in range(width):
                tmp[y, x] += weight * row[x + k]
    for y in range(height):
        for x in range(width):
            out[y, x] = 0.0
        for k in range(2 * radius + 1):
            yy = y + k - radius
            if yy < 0:
                yy += height
            elif yy >= height:
                yy -= height
            weight = kernel[k]
            for x in range(width):
                out[y, x] += weight * tmp[yy, x]


# ----------------------------------------------------------------------
# Local maxima (ia_utilities.c: findLocalMaxima, isLocalMaxima)
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True)
def _is_local_maximum(image, value, radius, cx, cy):
    """Whether ``value`` at ``(cx, cy)`` is the maximum within ``radius``.

    Of two equal pixels, the one with the larger x and y wins, so that
    exactly one of them is a maximum."""
    height, width = image.shape
    rr = int(radius * radius)
    sy = max(cy - int(radius), 0)
    ey = min(cy + int(radius), height - 1)
    reach = int(math.ceil(radius))
    sx = max(cx - reach, 0)
    ex = min(cx + reach, width - 1)
    for y in range(sy, ey + 1):
        dy = (y - cy) * (y - cy)
        for x in range(sx, ex + 1):
            dx = (x - cx) * (x - cx)
            if dx + dy <= rr:
                if y <= cy and x <= cx:
                    if image[y, x] > value:
                        return False
                else:
                    if image[y, x] >= value:
                        return False
    return True


@numba.njit(nogil=True, cache=True)
def _find_maxima(image, threshold, radius, margin, taken, out_x, out_y):
    """Local maxima above ``threshold``, at least ``margin`` from the
    edges, that were not returned before (``taken``). Returns the number
    found, written into ``out_x``/``out_y``."""
    height, width = image.shape
    n = 0
    for y in range(margin, height - margin):
        for x in range(margin, width - margin):
            value = image[y, x]
            if value > threshold and taken[y, x] < 1:
                if _is_local_maximum(image, value, radius, x, y):
                    taken[y, x] += 1
                    if n < len(out_x):
                        out_x[n] = x
                        out_y[n] = y
                        n += 1
    return n


# ----------------------------------------------------------------------
# Neighbor search on a grid of cells
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True)
def _build_grid(x, y, n, cell):
    """Bucket ``n`` points into square cells of side ``cell``; returns
    (cell of each point, start of each cell in ``order``, ``order``,
    cells along x, cells along y, x origin, y origin)."""
    x0 = 0.0
    y0 = 0.0
    nx = 1
    ny = 1
    if n > 0:
        x0 = x[0]
        y0 = y[0]
        x1 = x[0]
        y1 = y[0]
        for i in range(n):
            x0 = min(x0, x[i])
            y0 = min(y0, y[i])
            x1 = max(x1, x[i])
            y1 = max(y1, y[i])
        nx = int((x1 - x0) / cell) + 1
        ny = int((y1 - y0) / cell) + 1
    cells = np.empty(n, dtype=np.int64)
    counts = np.zeros(nx * ny + 1, dtype=np.int64)
    for i in range(n):
        cx = int((x[i] - x0) / cell)
        cy = int((y[i] - y0) / cell)
        cells[i] = cy * nx + cx
        counts[cells[i] + 1] += 1
    for c in range(nx * ny):
        counts[c + 1] += counts[c]
    order = np.empty(n, dtype=np.int64)
    fill = counts[:-1].copy()
    for i in range(n):
        order[fill[cells[i]]] = i
        fill[cells[i]] += 1
    return cells, counts, order, nx, ny, x0, y0


@numba.njit(nogil=True, cache=True)
def _neighbors(qx, qy, radius, x, y, counts, order, nx, ny, x0, y0, cell, out):
    """Indices of the points within ``radius`` (<= ``cell``) of
    ``(qx, qy)``, the query point included; returns their number."""
    cx = int(math.floor((qx - x0) / cell))
    cy = int(math.floor((qy - y0) / cell))
    rr = radius * radius
    n = 0
    for gy in range(max(cy - 1, 0), min(cy + 2, ny)):
        for gx in range(max(cx - 1, 0), min(cx + 2, nx)):
            c = gy * nx + gx
            for k in range(counts[c], counts[c + 1]):
                i = order[k]
                dx = x[i] - qx
                dy = y[i] - qy
                if dx * dx + dy * dy <= rr:
                    if n < len(out):
                        out[n] = i
                        n += 1
    return n


# ----------------------------------------------------------------------
# The pixel-integrated Gaussian of a peak over its region
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _profile(center, origin, size, sigma, e, de):
    """Gaussian integrated over the pixels ``origin .. origin + size - 1``
    (pixel ``c`` spans ``[c - 0.5, c + 0.5]``), and its derivative with
    respect to ``center``."""
    inv = _INV_SQRT2 / sigma
    norm = _INV_SQRT2PI / sigma
    inv2 = 0.5 / (sigma * sigma)
    for i in range(size):
        a_plus = origin + i - center + 0.5
        a_minus = origin + i - center - 0.5
        e[i] = 0.5 * (math.erf(a_plus * inv) - math.erf(a_minus * inv))
        de[i] = -norm * (
            math.exp(-a_plus * a_plus * inv2)
            - math.exp(-a_minus * a_minus * inv2)
        )


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _peak_shape(x, y, xi, yi, sigma, roi_dx, roi_dy, ex, dex, ey, dey, psf):
    """Fill ``psf`` with the unit-integral PSF of a peak at ``(x, y)`` over
    the disc of its region, which starts at ``(xi, yi)``."""
    size = len(ex)
    _profile(x, xi, size, sigma, ex, dex)
    _profile(y, yi, size, sigma, ey, dey)
    for j in range(len(roi_dx)):
        psf[j] = ex[roi_dx[j]] * ey[roi_dy[j]]


# ----------------------------------------------------------------------
# The fit image (multi_fit.c: mFitAddPeak, mFitSubtractPeak, mFitCalcErr)
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _add_peak(
    photons,
    bg,
    xi,
    yi,
    psf,
    roi_dx,
    roi_dy,
    variance,
    use_var,
    f_data,
    bg_data,
    bg_counts,
    sign,
):
    """Add (``sign`` 1) or subtract (``sign`` -1) a peak to or from the
    fit image."""
    for j in range(len(roi_dx)):
        row = yi + roi_dy[j]
        col = xi + roi_dx[j]
        var = variance[row, col] if use_var else 0.0
        f_data[row, col] += sign * photons * psf[j]
        bg_data[row, col] += sign * (bg + var)
        bg_counts[row, col] += sign


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _peak_error(xi, yi, roi_dx, roi_dy, x_data, f_data, bg_data, bg_counts):
    """Poisson deviance of the current fit over a peak's region; returns
    (ok, error), not ok where the model is not positive."""
    error = 0.0
    for j in range(len(roi_dx)):
        row = yi + roi_dy[j]
        col = xi + roi_dx[j]
        fi = f_data[row, col] + bg_data[row, col] / bg_counts[row, col]
        if fi <= 0.0:
            return False, np.inf
        xv = x_data[row, col]
        error += 2.0 * ((fi - xv) - xv * math.log(fi / xv))
    return True, error


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _jacobian_hessian(
    photons,
    xi,
    yi,
    roi_dx,
    roi_dy,
    ex,
    dex,
    ey,
    dey,
    x_data,
    f_data,
    bg_data,
    bg_counts,
    jac,
    hess,
):
    """Gradient ``jac`` and Hessian ``hess`` of the deviance of a peak's
    region with respect to ``[photons, x, y, bg]`` (dao_fit.c:
    daoCalcJH2DFixed). The background enters with derivative 1, as in
    storm-analysis, although a pixel averages the backgrounds of the
    peaks covering it."""
    for p in range(4):
        jac[p] = 0.0
        for q in range(4):
            hess[p, q] = 0.0
    jt = np.empty(4)
    for j in range(len(roi_dx)):
        row = yi + roi_dy[j]
        col = xi + roi_dx[j]
        fi = f_data[row, col] + bg_data[row, col] / bg_counts[row, col]
        xv = x_data[row, col]
        a = roi_dx[j]
        b = roi_dy[j]
        jt[0] = ex[a] * ey[b]
        jt[1] = photons * dex[a] * ey[b]
        jt[2] = photons * ex[a] * dey[b]
        jt[3] = 1.0
        t1 = 2.0 * (1.0 - xv / fi)
        t2 = 2.0 * xv / (fi * fi)
        for p in range(4):
            jac[p] += t1 * jt[p]
            for q in range(p, 4):
                hess[p, q] += t2 * jt[p] * jt[q]
    for p in range(4):
        for q in range(p):
            hess[p, q] = hess[q, p]


@numba.njit(nogil=True, cache=True)
def _solve_cholesky(a, b, n):
    """Solve ``a x = b`` for symmetric positive definite ``a`` in place
    (the solution is left in ``b``); False if ``a`` is not positive
    definite or the solution is not finite (LAPACK dposv)."""
    for j in range(n):
        s = a[j, j]
        for k in range(j):
            s -= a[j, k] * a[j, k]
        if not s > 0.0:
            return False
        a[j, j] = math.sqrt(s)
        for i in range(j + 1, n):
            s = a[i, j]
            for k in range(j):
                s -= a[i, k] * a[j, k]
            a[i, j] = s / a[j, j]
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= a[i, k] * b[k]
        b[i] = s / a[i, i]
    for i in range(n - 1, -1, -1):
        s = b[i]
        for k in range(i + 1, n):
            s -= a[k, i] * b[k]
        b[i] = s / a[i, i]
    for i in range(n):
        if not np.isfinite(b[i]):
            return False
    return True


# ----------------------------------------------------------------------
# One frame
# ----------------------------------------------------------------------


class FrameResult(NamedTuple):
    """The emitters found in one frame by :func:`analyze_frame`, in the
    pixel coordinates of the frame (pixel centers on integers)."""

    x: np.ndarray
    y: np.ndarray
    photons: np.ndarray
    bg: np.ndarray
    significance: np.ndarray
    #: Poisson deviance of the fit over the emitter's region.
    error: np.ndarray
    iterations: np.ndarray
    #: Number of pixels in the region the deviance is summed over.
    n_pixels: np.ndarray
    #: Cramer-Rao variances of photons, x, y and bg, see :func:`crlb`.
    crlb: np.ndarray
    #: Emitters within 4 PSF widths, the emitter itself included.
    n_emitters: np.ndarray


@numba.njit(nogil=True, cache=True)
def _grow(array, capacity):
    """1D ``array`` copied into one ``capacity`` long."""
    out = np.zeros(capacity, dtype=array.dtype)
    out[: len(array)] = array
    return out


@numba.njit(nogil=True, cache=True)
def _grow2(array, capacity):
    """2D ``array`` copied into one with ``capacity`` rows."""
    out = np.zeros((capacity, array.shape[1]), dtype=array.dtype)
    out[: len(array)] = array
    return out


@numba.njit(nogil=True, cache=True)
def _compact(
    keep,
    n,
    p_n,
    p_x,
    p_y,
    p_bg,
    p_xi,
    p_yi,
    p_status,
    p_lambda,
    p_error,
    p_iter,
    p_added,
    p_psf,
):
    """Remove the peaks that ``keep`` excludes, preserving the order;
    returns the new count."""
    m = 0
    for i in range(n):
        if keep[i]:
            if i != m:
                p_n[m] = p_n[i]
                p_x[m] = p_x[i]
                p_y[m] = p_y[i]
                p_bg[m] = p_bg[i]
                p_xi[m] = p_xi[i]
                p_yi[m] = p_yi[i]
                p_status[m] = p_status[i]
                p_lambda[m] = p_lambda[i]
                p_error[m] = p_error[i]
                p_iter[m] = p_iter[i]
                p_added[m] = p_added[i]
                for j in range(p_psf.shape[1]):
                    p_psf[m, j] = p_psf[i, j]
            m += 1
    return m


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _iterate(
    n,
    p_n,
    p_x,
    p_y,
    p_bg,
    p_xi,
    p_yi,
    p_status,
    p_lambda,
    p_error,
    p_iter,
    p_added,
    p_psf,
    sigma,
    roi,
    xoff,
    roi_dx,
    roi_dy,
    x_data,
    variance,
    use_var,
    f_data,
    bg_data,
    bg_counts,
    tolerance,
    ex,
    dex,
    ey,
    dey,
    psf,
):
    """One Levenberg-Marquardt step of every running peak, the others held
    fixed, then recenter the regions (multi_fit.c: mFitIterateLM,
    mFitRecenterPeaks)."""
    height, width = x_data.shape
    jac = np.empty(4)
    hess = np.empty((4, 4))
    w_jac = np.empty(4)
    w_hess = np.empty((4, 4))
    for i in range(n):
        if p_status[i] != _RUNNING:
            continue
        lam = p_lambda[i]
        ok, start_error = _peak_error(
            p_xi[i],
            p_yi[i],
            roi_dx,
            roi_dy,
            x_data,
            f_data,
            bg_data,
            bg_counts,
        )
        _profile(p_x[i], p_xi[i], roi, sigma, ex, dex)
        _profile(p_y[i], p_yi[i], roi, sigma, ey, dey)
        _jacobian_hessian(
            p_n[i],
            p_xi[i],
            p_yi[i],
            roi_dx,
            roi_dy,
            ex,
            dex,
            ey,
            dey,
            x_data,
            f_data,
            bg_data,
            bg_counts,
            jac,
            hess,
        )
        # the peak leaves the fit image while it is improved
        _add_peak(
            p_n[i],
            p_bg[i],
            p_xi[i],
            p_yi[i],
            p_psf[i],
            roi_dx,
            roi_dy,
            variance,
            use_var,
            f_data,
            bg_data,
            bg_counts,
            -1,
        )
        added = False
        status = _RUNNING
        w_n = p_n[i]
        w_x = p_x[i]
        w_y = p_y[i]
        w_bg = p_bg[i]
        w_xi = p_xi[i]
        w_yi = p_yi[i]
        error = start_error
        iterations = p_iter[i]
        while True:
            if lam > _LAMBDA_MAX:
                status = _ERROR
                break
            iterations += 1
            status = _RUNNING
            for p in range(4):
                w_jac[p] = jac[p]
                for q in range(4):
                    w_hess[p, q] = hess[p, q]
                w_hess[p, p] *= 1.0 + lam
            if not _solve_cholesky(w_hess, w_jac, 4):
                status = _ERROR
                lam *= _LAMBDA_UP
                continue
            w_n = p_n[i] - w_jac[0]
            w_x = p_x[i] - w_jac[1]
            w_y = p_y[i] - w_jac[2]
            w_bg = p_bg[i] - w_jac[3]
            w_xi = p_xi[i]
            w_yi = p_yi[i]
            # mFitUpdate: move the region with the peak
            dx = w_x - xoff - w_xi
            if dx < -0.5 or dx > 1.5:
                w_xi = int(math.floor(w_x - xoff))
            dy = w_y - xoff - w_yi
            if dy < -0.5 or dy > 1.5:
                w_yi = int(math.floor(w_y - xoff))
            # mFitCheck: within the image and a non-negative amplitude
            if (
                w_xi < 0
                or w_xi >= width - roi
                or w_yi < 0
                or w_yi >= height - roi
                or w_n < 0.0
                or not np.isfinite(w_x + w_y + w_n + w_bg)
            ):
                lam *= _LAMBDA_UP
                status = _ERROR
                continue
            _peak_shape(
                w_x,
                w_y,
                w_xi,
                w_yi,
                sigma,
                roi_dx,
                roi_dy,
                ex,
                dex,
                ey,
                dey,
                psf,
            )
            _add_peak(
                w_n,
                w_bg,
                w_xi,
                w_yi,
                psf,
                roi_dx,
                roi_dy,
                variance,
                use_var,
                f_data,
                bg_data,
                bg_counts,
                1,
            )
            ok, error = _peak_error(
                w_xi,
                w_yi,
                roi_dx,
                roi_dy,
                x_data,
                f_data,
                bg_data,
                bg_counts,
            )
            if not ok:
                _add_peak(
                    w_n,
                    w_bg,
                    w_xi,
                    w_yi,
                    psf,
                    roi_dx,
                    roi_dy,
                    variance,
                    use_var,
                    f_data,
                    bg_data,
                    bg_counts,
                    -1,
                )
                lam *= _LAMBDA_UP
                status = _ERROR
                continue
            if error > start_error:
                if (error - start_error) / start_error < tolerance:
                    status = _CONVERGED
                    added = True
                    break
                _add_peak(
                    w_n,
                    w_bg,
                    w_xi,
                    w_yi,
                    psf,
                    roi_dx,
                    roi_dy,
                    variance,
                    use_var,
                    f_data,
                    bg_data,
                    bg_counts,
                    -1,
                )
                lam *= _LAMBDA_UP
            else:
                if (start_error - error) / start_error < tolerance:
                    status = _CONVERGED
                    added = True
                    break
                if lam > _LAMBDA_MIN:
                    lam *= _LAMBDA_DOWN
                added = True
                break
        p_iter[i] = iterations
        p_lambda[i] = lam
        p_status[i] = status
        if added:
            p_n[i] = w_n
            p_x[i] = w_x
            p_y[i] = w_y
            p_bg[i] = w_bg
            p_xi[i] = w_xi
            p_yi[i] = w_yi
            p_error[i] = error
            for j in range(len(roi_dx)):
                p_psf[i, j] = psf[j]
            p_added[i] = True
        else:
            p_added[i] = False
            p_status[i] = _ERROR
    # mFitRecenterPeaks
    for i in range(n):
        if p_status[i] != _RUNNING:
            continue
        dx = p_x[i] - xoff - p_xi[i]
        dy = p_y[i] - xoff - p_yi[i]
        if dx < -0.25 or dx > 1.25 or dy < -0.25 or dy > 1.25:
            _add_peak(
                p_n[i],
                p_bg[i],
                p_xi[i],
                p_yi[i],
                p_psf[i],
                roi_dx,
                roi_dy,
                variance,
                use_var,
                f_data,
                bg_data,
                bg_counts,
                -1,
            )
            p_xi[i] = int(math.floor(p_x[i] - xoff))
            p_yi[i] = int(math.floor(p_y[i] - xoff))
            if (
                p_xi[i] < 0
                or p_xi[i] >= width - roi
                or p_yi[i] < 0
                or p_yi[i] >= height - roi
                or p_n[i] < 0.0
            ):
                p_status[i] = _ERROR
                p_added[i] = False
            else:
                _peak_shape(
                    p_x[i],
                    p_y[i],
                    p_xi[i],
                    p_yi[i],
                    sigma,
                    roi_dx,
                    roi_dy,
                    ex,
                    dex,
                    ey,
                    dey,
                    psf,
                )
                for j in range(len(roi_dx)):
                    p_psf[i, j] = psf[j]
                _add_peak(
                    p_n[i],
                    p_bg[i],
                    p_xi[i],
                    p_yi[i],
                    p_psf[i],
                    roi_dx,
                    roi_dy,
                    variance,
                    use_var,
                    f_data,
                    bg_data,
                    bg_counts,
                    1,
                )


@numba.njit(nogil=True, cache=True)
def _n_running(n, p_status):
    count = 0
    for i in range(n):
        if p_status[i] == _RUNNING:
            count += 1
    return count


@numba.njit(nogil=True, cache=True)
def _significance(
    n, p_n, p_xi, p_yi, p_psf, roi_dx, roi_dy, x_data, variance, use_var, out
):
    """Signal over the square root of the background, both weighted by
    the PSF (multi_fit.c: mFitPeakFgSum, mFitPeakBgSum)."""
    for i in range(n):
        psf_sum = 0.0
        fg_sum = 0.0
        bg_sum = 0.0
        for j in range(len(roi_dx)):
            row = p_yi[i] + roi_dy[j]
            col = p_xi[i] + roi_dx[j]
            psf = p_psf[i, j]
            psf_sum += psf
            fg_sum += psf * psf
            var = variance[row, col] if use_var else 0.0
            bg = (x_data[row, col] - var) - p_n[i] * psf
            bg_sum += bg * psf * psf
        fg_sum = p_n[i] * fg_sum
        if psf_sum > 0.0:
            fg_sum /= psf_sum
        else:
            fg_sum = 1.0
        if bg_sum > 0.0 and psf_sum > 0.0:
            bg_sum /= psf_sum * psf_sum
        else:
            bg_sum = 1.0
        out[i] = fg_sum / math.sqrt(bg_sum)


@numba.njit(nogil=True, cache=True)
def _do_fit(
    n,
    p_n,
    p_x,
    p_y,
    p_bg,
    p_xi,
    p_yi,
    p_status,
    p_lambda,
    p_error,
    p_iter,
    p_added,
    p_psf,
    sigma,
    roi,
    xoff,
    roi_dx,
    roi_dy,
    x_data,
    variance,
    use_var,
    f_data,
    bg_data,
    bg_counts,
    tolerance,
    ex,
    dex,
    ey,
    dey,
    psf,
):
    """Iterate until no peak is running (MultiFitter.doFit)."""
    _iterate(
        n,
        p_n,
        p_x,
        p_y,
        p_bg,
        p_xi,
        p_yi,
        p_status,
        p_lambda,
        p_error,
        p_iter,
        p_added,
        p_psf,
        sigma,
        roi,
        xoff,
        roi_dx,
        roi_dy,
        x_data,
        variance,
        use_var,
        f_data,
        bg_data,
        bg_counts,
        tolerance,
        ex,
        dex,
        ey,
        dey,
        psf,
    )
    i = 0
    while _n_running(n, p_status) > 0 and i < _MAX_FIT_ITERATIONS:
        _iterate(
            n,
            p_n,
            p_x,
            p_y,
            p_bg,
            p_xi,
            p_yi,
            p_status,
            p_lambda,
            p_error,
            p_iter,
            p_added,
            p_psf,
            sigma,
            roi,
            xoff,
            roi_dx,
            roi_dy,
            x_data,
            variance,
            use_var,
            f_data,
            bg_data,
            bg_counts,
            tolerance,
            ex,
            dex,
            ey,
            dey,
            psf,
        )
        i += 1


@numba.njit(nogil=True, cache=True)
def _mark_running_near(new_x, new_y, n_new, n, p_x, p_y, p_status, radius):
    """Converged peaks within ``radius`` of a new peak run again
    (ia_utilities.c: runningIfHasNeighbors)."""
    if n_new == 0 or n == 0:
        return
    cells, counts, order, nx, ny, x0, y0 = _build_grid(
        new_x, new_y, n_new, radius
    )
    out = np.empty(n_new, dtype=np.int64)
    for i in range(n):
        if p_status[i] != _CONVERGED:
            continue
        if _neighbors(
            p_x[i],
            p_y[i],
            radius,
            new_x,
            new_y,
            counts,
            order,
            nx,
            ny,
            x0,
            y0,
            radius,
            out,
        ):
            p_status[i] = _RUNNING


@numba.njit(nogil=True, cache=True)
def _prune(
    n, p_n, p_x, p_y, p_status, significance, sigma, threshold, neighborhood
):
    """Remove the dimmer of two peaks closer than ``sigma`` and the peaks
    not more significant than ``threshold``; their neighbors run again
    (ia_utilities.c: markDimmerPeaks, markLowSignificancePeaks). Returns
    the number of peaks marked for removal."""
    if n == 0:
        return 0
    cells, counts, order, nx, ny, x0, y0 = _build_grid(
        p_x, p_y, n, neighborhood
    )
    out = np.empty(n, dtype=np.int64)
    removed = 0
    for i in range(n):
        if p_status[i] == _ERROR:
            continue
        m = _neighbors(
            p_x[i],
            p_y[i],
            sigma,
            p_x,
            p_y,
            counts,
            order,
            nx,
            ny,
            x0,
            y0,
            neighborhood,
            out,
        )
        dimmer = False
        for k in range(m):
            if p_n[out[k]] > p_n[i]:
                dimmer = True
                break
        if dimmer:
            removed += 1
            p_status[i] = _ERROR
            m = _neighbors(
                p_x[i],
                p_y[i],
                neighborhood,
                p_x,
                p_y,
                counts,
                order,
                nx,
                ny,
                x0,
                y0,
                neighborhood,
                out,
            )
            for k in range(m):
                if p_status[out[k]] == _CONVERGED:
                    p_status[out[k]] = _RUNNING
    for i in range(n):
        if p_status[i] == _ERROR:
            continue
        if significance[i] > threshold:
            continue
        p_status[i] = _ERROR
        removed += 1
        m = _neighbors(
            p_x[i],
            p_y[i],
            neighborhood,
            p_x,
            p_y,
            counts,
            order,
            nx,
            ny,
            x0,
            y0,
            neighborhood,
            out,
        )
        for k in range(m):
            if p_status[out[k]] == _CONVERGED:
                p_status[out[k]] = _RUNNING
    return removed


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _local_crlb(
    i,
    n,
    p_n,
    p_x,
    p_y,
    p_xi,
    p_yi,
    sigma,
    roi,
    roi_dx,
    roi_dy,
    f_data,
    bg_data,
    bg_counts,
    cell_counts,
    cell_order,
    nx,
    ny,
    x0,
    y0,
    neighbor_distance,
    max_neighbors,
    out_var,
):
    """Cramer-Rao variances of ``[photons, x, y, bg]`` of peak ``i`` from
    the Fisher matrix of it and its nearest neighbors (at most
    ``max_neighbors`` within ``neighbor_distance``) over their regions,
    with Huang et al.'s correction of the x-x and y-y terms of close
    emitters (Eq. 11). Returns the number of emitters in the matrix."""
    candidates = np.empty(n, dtype=np.int64)
    m = _neighbors(
        p_x[i],
        p_y[i],
        neighbor_distance,
        p_x,
        p_y,
        cell_counts,
        cell_order,
        nx,
        ny,
        x0,
        y0,
        neighbor_distance,
        candidates,
    )
    # i first, then the nearest neighbors
    dist = np.empty(m)
    for k in range(m):
        j = candidates[k]
        dist[k] = (p_x[j] - p_x[i]) ** 2 + (p_y[j] - p_y[i]) ** 2
        if j == i:
            dist[k] = -1.0
    order = np.argsort(dist)
    n_local = min(m, max_neighbors + 1)
    members = np.empty(n_local, dtype=np.int64)
    for k in range(n_local):
        members[k] = candidates[order[k]]
    n_par = 4 * n_local
    fisher = np.zeros((n_par, n_par))
    # the union of the regions: a pixel enters once
    height, width = f_data.shape
    seen = np.zeros((roi * 3, roi * 3), dtype=np.bool_)
    ox = p_xi[i] - roi
    oy = p_yi[i] - roi
    ex = np.empty(roi)
    dex = np.empty(roi)
    ey = np.empty(roi)
    dey = np.empty(roi)
    grad = np.zeros(n_par)
    profiles = np.empty((n_local, 4, roi))
    disc = np.zeros((roi, roi), dtype=np.bool_)
    for r in range(len(roi_dx)):
        disc[roi_dy[r], roi_dx[r]] = True
    for k in range(n_local):
        j = members[k]
        _profile(p_x[j], p_xi[j], roi, sigma, ex, dex)
        _profile(p_y[j], p_yi[j], roi, sigma, ey, dey)
        for c in range(roi):
            profiles[k, 0, c] = ex[c]
            profiles[k, 1, c] = dex[c]
            profiles[k, 2, c] = ey[c]
            profiles[k, 3, c] = dey[c]
    for k in range(n_local):
        j = members[k]
        for r in range(len(roi_dx)):
            row = p_yi[j] + roi_dy[r]
            col = p_xi[j] + roi_dx[r]
            sx = col - ox
            sy = row - oy
            if sx < 0 or sy < 0 or sx >= 3 * roi or sy >= 3 * roi:
                continue
            if seen[sy, sx]:
                continue
            seen[sy, sx] = True
            count = bg_counts[row, col]
            if count <= 0:
                continue
            mu = f_data[row, col] + bg_data[row, col] / count
            if not mu > 0.0:
                continue
            for p in range(n_par):
                grad[p] = 0.0
            for kk in range(n_local):
                jj = members[kk]
                a = col - p_xi[jj]
                b = row - p_yi[jj]
                if a < 0 or b < 0 or a >= roi or b >= roi or not disc[b, a]:
                    continue
                exv = profiles[kk, 0, a]
                dexv = profiles[kk, 1, a]
                eyv = profiles[kk, 2, b]
                deyv = profiles[kk, 3, b]
                grad[4 * kk] = exv * eyv
                grad[4 * kk + 1] = p_n[jj] * dexv * eyv
                grad[4 * kk + 2] = p_n[jj] * exv * deyv
                grad[4 * kk + 3] = 1.0 / count
            for p in range(n_par):
                if grad[p] == 0.0:
                    continue
                gp = grad[p] / mu
                for q in range(p, n_par):
                    fisher[p, q] += gp * grad[q]
    for p in range(n_par):
        for q in range(p):
            fisher[p, q] = fisher[q, p]
    cov = np.linalg.pinv(fisher)
    regular = True
    for p in range(4):
        if not (np.isfinite(cov[p, p]) and cov[p, p] > 0.0):
            regular = False
    if not regular and n_local > 1:
        # A singular matrix: Huang et al. (2011), Eq. 11. Sigma from the
        # matrix without the x-x and y-y terms between emitters, then those
        # terms scaled by A / (A + 1), A = (theta_p - theta_q)^2 /
        # (sigma_p sigma_q). Applied only here: with free photons and
        # backgrounds it can make a regular matrix indefinite, and the
        # pruning keeps emitters a PSF width apart, where the plain bound
        # holds (their Fig. 4a).
        zeroed = fisher.copy()
        for a in range(n_local):
            for b in range(n_local):
                if a != b:
                    zeroed[4 * a + 1, 4 * b + 1] = 0.0
                    zeroed[4 * a + 2, 4 * b + 2] = 0.0
        cov0 = np.linalg.pinv(zeroed)
        for a in range(n_local):
            for b in range(n_local):
                if a == b:
                    continue
                for axis in (1, 2):
                    p = 4 * a + axis
                    q = 4 * b + axis
                    s = math.sqrt(abs(cov0[p, p] * cov0[q, q]))
                    if axis == 1:
                        diff = p_x[members[a]] - p_x[members[b]]
                    else:
                        diff = p_y[members[a]] - p_y[members[b]]
                    if s > 0.0:
                        big_a = diff * diff / s
                        fisher[p, q] *= big_a / (big_a + 1.0)
        cov = np.linalg.pinv(fisher)
    for p in range(4):
        out_var[p] = cov[p, p]
    return n_local


@numba.njit(nogil=True, cache=True)
def analyze_frame(
    frame,
    variance,
    use_variance,
    peak_mask,
    seed_x,
    seed_y,
    sigma,
    threshold,
    n_cycles,
    peak_radius,
    background_sigma,
):
    """Find and fit the emitters of one frame.

    Parameters
    ----------
    frame : np.ndarray
        ``(height, width)`` frame in photons (float64), padded by
        :func:`padding` pixels on every side by mirroring. Values below one
        photon are taken as one, as the Poisson likelihood needs positive
        data.
    variance : np.ndarray
        Per-pixel sCMOS readout variance in photons squared, padded like
        ``frame``; ignored unless ``use_variance``.
    use_variance : bool
        Whether ``variance`` is applied (Huang et al., Nat. Methods 10,
        653-658, 2013): it is added to the data and the model, and to the
        noise that the peak threshold is relative to.
    peak_mask : np.ndarray
        ``(height, width)`` bool, where new peaks may be found.
    seed_x, seed_y : np.ndarray
        Integer pixel positions (padded frame) that start the first cycle
        instead of the peak finder; empty for none.
    sigma : float
        PSF standard deviation in camera pixels, fixed.
    threshold : float
        Peak threshold and minimum significance, in units of the
        signal-to-noise ratio.
    n_cycles : int
        Most find-fit-subtract cycles.
    peak_radius : float
        Radius (camera pixels) within which a new peak must be the
        maximum.
    background_sigma : float
        Width (camera pixels) of the background low pass.

    Returns
    -------
    result : FrameResult
        The emitters, in the coordinates of the padded frame.
    """
    height, width = frame.shape
    roi = int(_ROI_SIZE_FACTOR * sigma)
    margin = int(roi / 2 + 2)
    xoff = 0.5 * roi - 1.0
    # disc of the region (multi_fit.c: mFitInitializeROIIndexing)
    roi_dx_list = []
    roi_dy_list = []
    center = 0.5 * roi
    for i in range(roi):
        dyy = i - center + 0.5
        for j in range(roi):
            dxx = j - center + 0.5
            if dxx * dxx + dyy * dyy <= center * center:
                roi_dx_list.append(j)
                roi_dy_list.append(i)
    n_roi = len(roi_dx_list)
    roi_dx = np.empty(n_roi, dtype=np.int64)
    roi_dy = np.empty(n_roi, dtype=np.int64)
    for j in range(n_roi):
        roi_dx[j] = roi_dx_list[j]
        roi_dy[j] = roi_dy_list[j]

    image = np.empty((height, width))
    x_data = np.empty((height, width))
    for y in range(height):
        for x in range(width):
            value = max(frame[y, x], 1.0)
            image[y, x] = value
            x_data[y, x] = value + (variance[y, x] if use_variance else 0.0)
    f_data = np.zeros((height, width))
    bg_data = np.zeros((height, width))
    bg_counts = np.zeros((height, width), dtype=np.int64)
    background = np.empty((height, width))
    work = np.empty((height, width))
    work2 = np.empty((height, width))
    fg = np.empty((height, width))
    snr = np.empty((height, width))
    tmp = np.empty((height, width))
    taken = np.zeros((height, width), dtype=np.int64)
    size = min(height, width)
    bg_kernel = _gaussian_kernel(background_sigma, size, False)
    # the foreground is smoothed with the PSF: its point-sampled
    # equivalent is a Gaussian of sqrt(sigma^2 + 1/12)
    fg_sigma = math.sqrt(sigma * sigma + 1.0 / 12.0)
    fg_kernel = _gaussian_kernel(fg_sigma, size, False)
    fg_kernel2 = _gaussian_kernel(fg_sigma, size, True)

    capacity = 256
    p_n = np.zeros(capacity)
    p_x = np.zeros(capacity)
    p_y = np.zeros(capacity)
    p_bg = np.zeros(capacity)
    p_xi = np.zeros(capacity, dtype=np.int64)
    p_yi = np.zeros(capacity, dtype=np.int64)
    p_status = np.zeros(capacity, dtype=np.int64)
    p_lambda = np.zeros(capacity)
    p_error = np.zeros(capacity)
    p_iter = np.zeros(capacity, dtype=np.int64)
    p_added = np.zeros(capacity, dtype=np.bool_)
    p_psf = np.zeros((capacity, n_roi))
    n = 0
    ex = np.empty(roi)
    dex = np.empty(roi)
    ey = np.empty(roi)
    dey = np.empty(roi)
    psf = np.empty(n_roi)
    significance = np.zeros(capacity)
    max_new = height * width // 4 + 16
    new_x = np.empty(max_new, dtype=np.int64)
    new_y = np.empty(max_new, dtype=np.int64)
    neighborhood = _NEIGHBORHOOD * sigma

    for cycle in range(n_cycles):
        # 1. background of what the fit does not explain
        for y in range(height):
            for x in range(width):
                work[y, x] = image[y, x] - f_data[y, x]
        _convolve(work, bg_kernel, background, tmp)
        # 2. new peaks
        if cycle == 0 and len(seed_x) > 0:
            n_new = 0
            for k in range(len(seed_x)):
                if n_new < max_new:
                    new_x[n_new] = seed_x[k]
                    new_y[n_new] = seed_y[k]
                    taken[seed_y[k], seed_x[k]] += 1
                    n_new += 1
        else:
            for y in range(height):
                for x in range(width):
                    var = background[y, x] + f_data[y, x]
                    if use_variance:
                        var += variance[y, x]
                    work[y, x] = var
                    work2[y, x] = image[y, x] - background[y, x] - f_data[y, x]
            _convolve(work, fg_kernel2, snr, tmp)
            _convolve(work2, fg_kernel, fg, tmp)
            for y in range(height):
                for x in range(width):
                    var = snr[y, x]
                    if var <= _VARIANCE_FLOOR:
                        var = _VARIANCE_FLOOR
                    value = fg[y, x] / math.sqrt(var)
                    snr[y, x] = value if peak_mask[y, x] else 0.0
            n_new = _find_maxima(
                snr, threshold, peak_radius, margin + 1, taken, new_x, new_y
            )
        # A cycle without new peaks leaves the fit as it is, so every
        # later cycle would find the same nothing. storm-analysis cycles
        # on until its threshold schedule has run down (at least five
        # cycles), to the same result.
        if n_new == 0:
            break
        # 3. fit (PeakFitter.fitPeaks)
        if n > 0:
            fnx = np.empty(n_new)
            fny = np.empty(n_new)
            for k in range(n_new):
                fnx[k] = new_x[k]
                fny[k] = new_y[k]
            _mark_running_near(
                fnx, fny, n_new, n, p_x, p_y, p_status, neighborhood
            )
        if n + n_new > capacity:
            capacity = 2 * (n + n_new)
            p_n = _grow(p_n, capacity)
            p_x = _grow(p_x, capacity)
            p_y = _grow(p_y, capacity)
            p_bg = _grow(p_bg, capacity)
            p_xi = _grow(p_xi, capacity)
            p_yi = _grow(p_yi, capacity)
            p_status = _grow(p_status, capacity)
            p_lambda = _grow(p_lambda, capacity)
            p_error = _grow(p_error, capacity)
            p_iter = _grow(p_iter, capacity)
            p_added = _grow(p_added, capacity)
            p_psf = _grow2(p_psf, capacity)
            significance = np.zeros(capacity)
        start = n
        # dao_fit.c: daoNewPeaks ("finder" peaks)
        for k in range(n_new):
            i = n
            p_x[i] = float(new_x[k])
            p_y[i] = float(new_y[k])
            p_xi[i] = int(math.floor(p_x[i] - xoff))
            p_yi[i] = int(math.floor(p_y[i] - xoff))
            p_status[i] = _RUNNING
            p_lambda[i] = _LAMBDA_START
            p_error[i] = 0.0
            p_iter[i] = 0
            p_added[i] = False
            p_n[i] = 1.0
            p_bg[i] = 1.0
            n += 1
            if (
                p_xi[i] < 0
                or p_xi[i] >= width - roi
                or p_yi[i] < 0
                or p_yi[i] >= height - roi
            ):
                p_status[i] = _ERROR
                continue
            _peak_shape(
                p_x[i],
                p_y[i],
                p_xi[i],
                p_yi[i],
                sigma,
                roi_dx,
                roi_dy,
                ex,
                dex,
                ey,
                dey,
                psf,
            )
            bg_sum = 0.0
            sp = 0.0
            sx = 0.0
            for j in range(n_roi):
                row = p_yi[i] + roi_dy[j]
                col = p_xi[i] + roi_dx[j]
                bg_sum += background[row, col]
                t1 = psf[j]
                sp += t1 * t1 * t1
                sx += (
                    t1
                    * t1
                    * (
                        x_data[row, col]
                        - f_data[row, col]
                        - background[row, col]
                    )
                )
                p_psf[i, j] = psf[j]
            p_bg[i] = bg_sum / n_roi
            p_n[i] = sx / sp
            if p_n[i] <= 0.0:
                p_status[i] = _ERROR
                continue
            _add_peak(
                p_n[i],
                p_bg[i],
                p_xi[i],
                p_yi[i],
                p_psf[i],
                roi_dx,
                roi_dy,
                variance,
                use_variance,
                f_data,
                bg_data,
                bg_counts,
                1,
            )
            p_added[i] = True
        for i in range(start, n):
            if p_status[i] == _RUNNING:
                ok, p_error[i] = _peak_error(
                    p_xi[i],
                    p_yi[i],
                    roi_dx,
                    roi_dy,
                    x_data,
                    f_data,
                    bg_data,
                    bg_counts,
                )
        _do_fit(
            n,
            p_n,
            p_x,
            p_y,
            p_bg,
            p_xi,
            p_yi,
            p_status,
            p_lambda,
            p_error,
            p_iter,
            p_added,
            p_psf,
            sigma,
            roi,
            xoff,
            roi_dx,
            roi_dy,
            x_data,
            variance,
            use_variance,
            f_data,
            bg_data,
            bg_counts,
            _TOLERANCE,
            ex,
            dex,
            ey,
            dey,
            psf,
        )
        keep = p_status[:n] != _ERROR
        n = _compact(
            keep,
            n,
            p_n,
            p_x,
            p_y,
            p_bg,
            p_xi,
            p_yi,
            p_status,
            p_lambda,
            p_error,
            p_iter,
            p_added,
            p_psf,
        )
        _significance(
            n,
            p_n,
            p_xi,
            p_yi,
            p_psf,
            roi_dx,
            roi_dy,
            x_data,
            variance,
            use_variance,
            significance,
        )
        removed = _prune(
            n,
            p_n,
            p_x,
            p_y,
            p_status,
            significance,
            sigma,
            threshold,
            neighborhood,
        )
        if removed > 0:
            for i in range(n):
                if p_status[i] == _ERROR and p_added[i]:
                    _add_peak(
                        p_n[i],
                        p_bg[i],
                        p_xi[i],
                        p_yi[i],
                        p_psf[i],
                        roi_dx,
                        roi_dy,
                        variance,
                        use_variance,
                        f_data,
                        bg_data,
                        bg_counts,
                        -1,
                    )
                    p_added[i] = False
            keep = p_status[:n] != _ERROR
            n = _compact(
                keep,
                n,
                p_n,
                p_x,
                p_y,
                p_bg,
                p_xi,
                p_yi,
                p_status,
                p_lambda,
                p_error,
                p_iter,
                p_added,
                p_psf,
            )
        if _n_running(n, p_status) > 0:
            _do_fit(
                n,
                p_n,
                p_x,
                p_y,
                p_bg,
                p_xi,
                p_yi,
                p_status,
                p_lambda,
                p_error,
                p_iter,
                p_added,
                p_psf,
                sigma,
                roi,
                xoff,
                roi_dx,
                roi_dy,
                x_data,
                variance,
                use_variance,
                f_data,
                bg_data,
                bg_counts,
                _TOLERANCE,
                ex,
                dex,
                ey,
                dey,
                psf,
            )
            keep = p_status[:n] != _ERROR
            n = _compact(
                keep,
                n,
                p_n,
                p_x,
                p_y,
                p_bg,
                p_xi,
                p_yi,
                p_status,
                p_lambda,
                p_error,
                p_iter,
                p_added,
                p_psf,
            )

    # peaks that did not converge are dropped (removeRunningPeaks)
    for i in range(n):
        if p_status[i] == _RUNNING and p_added[i]:
            _add_peak(
                p_n[i],
                p_bg[i],
                p_xi[i],
                p_yi[i],
                p_psf[i],
                roi_dx,
                roi_dy,
                variance,
                use_variance,
                f_data,
                bg_data,
                bg_counts,
                -1,
            )
            p_added[i] = False
    keep = p_status[:n] == _CONVERGED
    n = _compact(
        keep,
        n,
        p_n,
        p_x,
        p_y,
        p_bg,
        p_xi,
        p_yi,
        p_status,
        p_lambda,
        p_error,
        p_iter,
        p_added,
        p_psf,
    )
    _significance(
        n,
        p_n,
        p_xi,
        p_yi,
        p_psf,
        roi_dx,
        roi_dy,
        x_data,
        variance,
        use_variance,
        significance,
    )
    # the final deviance of each region, with every peak in place
    for i in range(n):
        ok, p_error[i] = _peak_error(
            p_xi[i],
            p_yi[i],
            roi_dx,
            roi_dy,
            x_data,
            f_data,
            bg_data,
            bg_counts,
        )
    crlb = np.full((n, 4), np.nan)
    n_emitters = np.ones(n, dtype=np.int64)
    if n > 0:
        neighbor_distance = _NEIGHBOR_DISTANCE * sigma
        cells, counts, order, nx, ny, x0, y0 = _build_grid(
            p_x, p_y, n, neighbor_distance
        )
        out_var = np.empty(4)
        for i in range(n):
            n_emitters[i] = _local_crlb(
                i,
                n,
                p_n,
                p_x,
                p_y,
                p_xi,
                p_yi,
                sigma,
                roi,
                roi_dx,
                roi_dy,
                f_data,
                bg_data,
                bg_counts,
                counts,
                order,
                nx,
                ny,
                x0,
                y0,
                neighbor_distance,
                _MAX_NEIGHBORS,
                out_var,
            )
            for p in range(4):
                crlb[i, p] = out_var[p]
    n_pixels = np.full(n, n_roi, dtype=np.int64)
    return FrameResult(
        p_x[:n].copy(),
        p_y[:n].copy(),
        p_n[:n].copy(),
        p_bg[:n].copy(),
        significance[:n].copy(),
        p_error[:n].copy(),
        p_iter[:n].copy(),
        n_pixels,
        crlb,
        n_emitters,
    )


# ----------------------------------------------------------------------
# Peaks of the first cycle, and single emitters with a free width: the
# PSF width of a movie is the median width of its isolated emitters.
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True)
def find_peaks(
    frame, peak_mask, sigma, threshold, peak_radius, background_sigma
):
    """The peaks the first cycle of :func:`analyze_frame` finds.

    Parameters
    ----------
    frame : np.ndarray
        Padded frame in photons, as for :func:`analyze_frame`.
    peak_mask : np.ndarray
        Where peaks may be found.
    sigma : float
        PSF standard deviation (camera pixels) the residual is smoothed
        with.
    threshold : float
        Peak threshold in units of the signal-to-noise ratio.
    peak_radius : float
        Radius (camera pixels) within which a peak is the maximum.
    background_sigma : float
        Width (camera pixels) of the background low pass.

    Returns
    -------
    x, y : np.ndarray
        Integer pixel positions of the peaks in the padded frame.
    """
    height, width = frame.shape
    roi = int(_ROI_SIZE_FACTOR * sigma)
    margin = int(roi / 2 + 2)
    image = np.empty((height, width))
    for y in range(height):
        for x in range(width):
            image[y, x] = max(frame[y, x], 1.0)
    background = np.empty((height, width))
    tmp = np.empty((height, width))
    size = min(height, width)
    _convolve(
        image,
        _gaussian_kernel(background_sigma, size, False),
        background,
        tmp,
    )
    fg_sigma = math.sqrt(sigma * sigma + 1.0 / 12.0)
    work = np.empty((height, width))
    for y in range(height):
        for x in range(width):
            work[y, x] = image[y, x] - background[y, x]
    fg = np.empty((height, width))
    _convolve(work, _gaussian_kernel(fg_sigma, size, False), fg, tmp)
    var = np.empty((height, width))
    _convolve(background, _gaussian_kernel(fg_sigma, size, True), var, tmp)
    for y in range(height):
        for x in range(width):
            v = max(var[y, x], _VARIANCE_FLOOR)
            work[y, x] = fg[y, x] / math.sqrt(v) if peak_mask[y, x] else 0.0
    taken = np.zeros((height, width), dtype=np.int64)
    out_x = np.empty(height * width // 4 + 16, dtype=np.int64)
    out_y = np.empty(height * width // 4 + 16, dtype=np.int64)
    n = _find_maxima(
        work, threshold, peak_radius, margin + 1, taken, out_x, out_y
    )
    return out_x[:n].copy(), out_y[:n].copy()


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _profile_sigma(center, sigma, size, e, de, ds):
    """Gaussian integrated over the pixels ``0 .. size - 1``, and its
    derivatives with respect to ``center`` and to ``sigma``."""
    inv = _INV_SQRT2 / sigma
    norm = _INV_SQRT2PI / sigma
    norm_s = _INV_SQRT2PI / (sigma * sigma)
    inv2 = 0.5 / (sigma * sigma)
    for c in range(size):
        a_plus = c - center + 0.5
        a_minus = c - center - 0.5
        g_plus = math.exp(-a_plus * a_plus * inv2)
        g_minus = math.exp(-a_minus * a_minus * inv2)
        e[c] = 0.5 * (math.erf(a_plus * inv) - math.erf(a_minus * inv))
        de[c] = -norm * (g_plus - g_minus)
        ds[c] = -norm_s * (a_plus * g_plus - a_minus * g_minus)


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_free_sigma(
    spots,
    index,
    variance,
    use_variance,
    theta,
    ex,
    dex,
    dsx,
    ey,
    dey,
    dsy,
    d,
    hess,
    grad,
):
    """Poisson deviance, gradient and Hessian of the pixel-integrated
    Gaussian ``[photons, x, y, sigma, bg]``."""
    box = spots.shape[1]
    for p in range(5):
        if not np.isfinite(theta[p]):
            return np.inf, False
        grad[p] = 0.0
        for q in range(5):
            hess[p, q] = 0.0
    photons = theta[0]
    sigma = theta[3]
    if not sigma > 0.05:
        return np.inf, False
    _profile_sigma(theta[1], sigma, box, ex, dex, dsx)
    _profile_sigma(theta[2], sigma, box, ey, dey, dsy)
    chi_square = 0.0
    for j in range(box):
        for i in range(box):
            g = ex[i] * ey[j]
            value = photons * g + theta[4]
            d[0] = g
            d[1] = photons * dex[i] * ey[j]
            d[2] = photons * ex[i] * dey[j]
            d[3] = photons * (dsx[i] * ey[j] + ex[i] * dsy[j])
            d[4] = 1.0
            var = variance[index, j, i] if use_variance else 0.0
            contrib, weight, factor, ok = _estimator_terms(
                True, value, spots[index, j, i], var
            )
            if not ok:
                return np.inf, False
            chi_square += contrib
            for p in range(5):
                grad[p] += d[p] * factor
                wp = weight * d[p]
                for q in range(p, 5):
                    hess[p, q] += wp * d[q]
    for p in range(5):
        for q in range(p):
            hess[p, q] = hess[q, p]
    return chi_square, True


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _fit_free_sigma_spots(
    spots,
    variance,
    use_variance,
    init,
    tolerance,
    max_iterations,
    thetas,
    chi_squares,
    states,
    iterations,
):
    """Levenberg-Marquardt fits of :func:`_accumulate_free_sigma`, the
    driver of ``picasso.fitting.gaussfit``."""
    n_params = 5
    box = spots.shape[1]
    theta = np.empty(n_params)
    theta_previous = np.empty(n_params)
    grad = np.zeros(n_params)
    grad_ok = np.zeros(n_params)
    delta = np.empty(n_params)
    scaling = np.zeros(n_params)
    hess = np.zeros((n_params, n_params))
    hess_ok = np.zeros((n_params, n_params))
    hess_damped = np.empty((n_params, n_params))
    indxc = np.empty(n_params, dtype=np.int32)
    indxr = np.empty(n_params, dtype=np.int32)
    ipiv = np.empty(n_params, dtype=np.int32)
    ex = np.empty(box)
    dex = np.empty(box)
    dsx = np.empty(box)
    ey = np.empty(box)
    dey = np.empty(box)
    dsy = np.empty(box)
    d = np.empty(n_params)
    for index in range(len(spots)):
        for p in range(n_params):
            theta[p] = init[index, p]
            scaling[p] = 0.0
        chi_square, ok = _accumulate_free_sigma(
            spots,
            index,
            variance,
            use_variance,
            theta,
            ex,
            dex,
            dsx,
            ey,
            dey,
            dsy,
            d,
            hess,
            grad,
        )
        n_iterations = 0
        state = FIT_STATE_CONVERGED
        if not ok:
            state = FIT_STATE_NEG_CURVATURE_MLE
        else:
            for p in range(n_params):
                grad_ok[p] = grad[p]
                for q in range(n_params):
                    hess_ok[p, q] = hess[p, q]
            lam = _LAMBDA_INITIAL
            previous_chi_square = chi_square
            for iteration in range(max_iterations):
                if not _lm_solve_step(
                    hess_ok,
                    grad_ok,
                    scaling,
                    lam,
                    hess_damped,
                    delta,
                    n_params,
                    indxc,
                    indxr,
                    ipiv,
                ):
                    state = FIT_STATE_SINGULAR_HESSIAN
                    break
                for p in range(n_params):
                    theta_previous[p] = theta[p]
                    theta[p] += delta[p]
                new_chi_square, ok = _accumulate_free_sigma(
                    spots,
                    index,
                    variance,
                    use_variance,
                    theta,
                    ex,
                    dex,
                    dsx,
                    ey,
                    dey,
                    dsy,
                    d,
                    hess,
                    grad,
                )
                n_iterations = iteration + 1
                if not ok:
                    chi_square, lam, state = _lm_reject_step(
                        theta,
                        theta_previous,
                        previous_chi_square,
                        lam,
                        state,
                        iteration,
                        max_iterations,
                        n_params,
                    )
                    continue
                chi_square, previous_chi_square, lam, state, converged = (
                    _lm_accept_or_reject(
                        theta,
                        theta_previous,
                        grad,
                        grad_ok,
                        hess,
                        hess_ok,
                        n_params,
                        new_chi_square,
                        previous_chi_square,
                        lam,
                        tolerance,
                        iteration,
                        max_iterations,
                        state,
                    )
                )
                if converged:
                    break
        _lm_finalize(
            theta,
            chi_square,
            state,
            n_iterations,
            True,
            n_params,
            thetas,
            chi_squares,
            states,
            iterations,
            index,
        )


def fit_free_sigma(
    spots: np.ndarray,
    variance: np.ndarray | None = None,
    tolerance: float = 1e-5,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit single emitters with a pixel-integrated Gaussian of free width.

    The maximum-likelihood fit ``[photons, x, y, sigma, bg]`` on each
    spot, seeded at the background-subtracted center of mass with a
    width of one pixel.

    Parameters
    ----------
    spots : np.ndarray
        ``(n_spots, box, box)`` spots in photons, at least zero.
    variance : np.ndarray, optional
        Per-pixel sCMOS readout variance in photons squared, laid out like
        ``spots``. Default is None.
    tolerance : float, optional
        Relative convergence tolerance of the deviance. Default is 1e-5.
    max_iterations : int, optional
        Most iterations per spot. Default is 100.

    Returns
    -------
    thetas : np.ndarray
        ``(n_spots, 5)`` parameters, NaN where the fit failed; x and y are
        box pixel coordinates.
    deviance : np.ndarray
        ``(n_spots,)`` Poisson deviance at the optimum.
    """
    spots = np.ascontiguousarray(spots, dtype=np.float32)
    n, box, _ = spots.shape
    background = spots.reshape(n, -1).min(axis=1).astype(np.float64)
    weights = np.clip(spots - background[:, None, None], 0, None).astype(
        np.float64
    )
    total = weights.sum(axis=(1, 2))
    grid = np.arange(box, dtype=np.float64)
    center = 0.5 * (box - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = (weights.sum(axis=1) * grid).sum(axis=1) / total
        cy = (weights.sum(axis=2) * grid).sum(axis=1) / total
    cx = np.where(np.isfinite(cx), cx, center)
    cy = np.where(np.isfinite(cy), cy, center)
    init = np.column_stack(
        [np.maximum(total, 1.0), cx, cy, np.ones(n), background]
    )
    if variance is None:
        variance = np.zeros((1, 1, 1), dtype=np.float32)
        use_variance = False
    else:
        variance = np.ascontiguousarray(variance, dtype=np.float32)
        use_variance = True
    thetas = np.full((n, 5), np.nan)
    chi_squares = np.full(n, np.inf)
    states = np.zeros(n, dtype=np.int32)
    iterations = np.zeros(n, dtype=np.int32)
    if n:
        _fit_free_sigma_spots(
            spots,
            variance,
            use_variance,
            init,
            float(tolerance),
            int(max_iterations),
            thetas,
            chi_squares,
            states,
            iterations,
        )
    return thetas, chi_squares
