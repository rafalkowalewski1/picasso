"""
    picasso.render
    ~~~~~~~~~~~~~~

    Render single molecule localizations to a super-resolution image

    :author: Joerg Schnitzbauer, 2015
"""
import numpy as _np
import numba as _numba
import scipy.signal as _signal


def render(locs, info, oversampling=1, viewport=None, blur_method=None):
    if viewport is None:
        viewport = [(0, 0), (info[0]['Width'], info[0]['Height'])]
    (y_min, x_min), (y_max, x_max) = viewport
    n_pixel_x = int(_np.round(oversampling * (y_max - y_min)))
    n_pixel_y = int(_np.round(oversampling * (x_max - x_min)))
    image = _np.zeros((n_pixel_x, n_pixel_y), dtype=_np.float32)
    x = locs.x
    y = locs.y
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    if _np.sum(in_view) == 0:
        return image
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    if blur_method is None:
        x = _np.int32(x)
        y = _np.int32(y)
        return _fill(image, x, y)
    elif blur_method == 'convolve':
        x = _np.int32(x)
        y = _np.int32(y)
        image = _fill(image, x, y)
        lpy = locs.lpy
        lpx = locs.lpx
        lpy = lpy[in_view]
        lpx = lpx[in_view]
        lpy = oversampling * _np.median(lpy)
        lpx = oversampling * _np.median(lpx)
        kernel_height = 10 * int(_np.round(lpy)) + 1
        kernel_width = 10 * int(_np.round(lpx)) + 1
        kernel_y = _signal.gaussian(kernel_height, lpy)
        kernel_x = _signal.gaussian(kernel_width, lpx)
        kernel = _np.outer(kernel_y, kernel_x)
        image = _signal.fftconvolve(image, kernel, mode='same')
        image = len(locs) * image / image.sum()
        return image
    elif blur_method == 'gaussian':
        lpy = locs.lpy
        lpx = locs.lpx
        lpy = oversampling * lpy[in_view]
        lpx = oversampling * lpx[in_view]
        return _fill_gaussians(image, x, y, lpx, lpy)
    else:
        raise Exception('blur_method not understood.')


@_numba.jit(nopython=True)
def _fill(image, x, y):
    for i, j in zip(x, y):
        image[j, i] += 1
    return image


@_numba.jit(nopython=True)
def _fill_gaussians(image, x, y, lpx, lpy):
    Y, X = image.shape
    for x_, y_, lpx_, lpy_ in zip(x, y, lpx, lpy):
        lpy_3 = 3 * lpy_
        i_min = _np.int32(y_ - lpy_3)
        if i_min < 0:
            i_min = 0
        i_max = _np.int32(y_ + lpy_3 + 1)
        if i_max > Y:
            i_max = Y
        lpx_3 = 3 * lpx_
        j_min = _np.int32(x_ - lpx_3)
        if j_min < 0:
            j_min = 0
        j_max = _np.int32(x_ + lpx_3) + 1
        if j_max > X:
            j_max = X
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                image[i, j] += _np.exp(-((j - x_)**2/(2 * lpx_**2) + (i - y_)**2/(2 * lpy_**2))) / (2 * _np.pi * lpx_ * lpy_)
    return image
