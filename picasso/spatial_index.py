"""Multi-resolution spatial index for fast viewport rendering.

Built once when a channel is loaded; queried per redraw to skip the
O(N) viewport scan that ``picasso.render._render_setup`` would otherwise
perform on every pan/zoom.

The pyramid stores three grid resolutions sharing a single permutation
sorted by Morton (Z-order) at the finest level. Because Z-order is
hierarchical, each coarser block at level L corresponds to a contiguous
range in the same sorted permutation -- so all levels reuse one ``perm``
array (~4 N bytes) rather than one per level.

Circular picks query the same pyramid (``query_circle``): the block
sizes do not depend on the pick size, so the index built at load time
serves every pick diameter, whereas the single-resolution
``get_index_blocks`` of :mod:`picasso.postprocess` has to be rebuilt
(sorting and copying the whole DataFrame) whenever the pick size
changes; it remains the fallback where no pyramid could be built.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import h5py
import numba
import numpy as np
import pandas as pd

from . import lib


_log = logging.getLogger(__name__)

#: HDF5 group holding a persisted pyramid, see ``save_render_index``.
RENDER_INDEX_GROUP = "render_index"
#: Files with fewer localizations get no persisted pyramid: theirs
#: builds in milliseconds and the 4 bytes per row would be a large part
#: of a small file.
PERSIST_MIN_LOCS = 100_000
_RENDER_INDEX_VERSION = 1


# Target upper bound on blocks per viewport edge at the chosen level.
# Tunable; ~64 keeps the inner gather loop tight while still letting the
# finest level cover small zoomed-in viewports.
_TARGET_BLOCKS_PER_EDGE = 64

# Viewport-to-FOV area ratio at/above which ``query_viewport`` bypasses
# the pyramid and returns ``None``. The caller then renders the full
# locs DataFrame and lets the renderer's vectorised ``in_view`` mask do
# the filtering -- avoiding a pandas ``iloc`` copy of nearly all rows,
# which dominates redraw cost at full-FOV (see ``query_viewport``).
_BYPASS_COVERAGE_RATIO = 0.1


@dataclass
class RenderIndexPyramid:
    """Multi-resolution spatial index over a single locs DataFrame.

    Attributes
    ----------
    perm : IntArray1D, shape (N,), dtype uint32
        ``perm[i]`` is the original-locs index at sort position ``i``,
        where the sort key is the Morton code of ``(x // base, y // base)``
        at the finest level.
    block_sizes : tuple[float, ...]
        Block side lengths in camera pixels, ascending. ``block_sizes[0]``
        is the finest level.
    block_starts, block_ends : list[IntArray2D]
        Per level, a ``(K_L, L_L)`` uint32 grid where
        ``perm[block_starts[i, j]:block_ends[i, j]]`` are the
        original-locs indices in block ``(i, j)``.
    width, height : float
        FOV size copied from ``info``, used by the query to clip block
        rectangles.
    """

    perm: lib.IntArray1D
    block_sizes: tuple[float, ...]
    block_starts: list[lib.IntArray2D]
    block_ends: list[lib.IntArray2D]
    width: float
    height: float


def _base_block_size(width: float, height: float) -> float:
    """Pick the finest block size based on FOV.

    Targets ~256k blocks at the finest level for the common 512x512 -
    1024x1024 SMLM FOVs. Floor of 1.0 -- sub-pixel blocks would mostly
    hold a single loc each and waste grid memory.
    """
    return float(max(1.0, np.ceil(np.sqrt(width * height / 256_000.0))))


@numba.njit(cache=True)
def _morton_encode_2d(x: lib.IntArray1D, y: lib.IntArray1D) -> lib.IntArray1D:
    """Interleave bits of ``(x, y)`` into a Morton (Z-order) key.

    ``x`` and ``y`` are 32-bit unsigned block coordinates; the returned
    key is uint64. Inputs above 2**16 are still handled because the
    masks below interleave the full 32-bit input -- but typical SMLM
    grids stay well below that.
    """
    n = x.shape[0]
    out = np.empty(n, dtype=np.uint64)
    M0 = np.uint64(0x0000FFFF0000FFFF)
    M1 = np.uint64(0x00FF00FF00FF00FF)
    M2 = np.uint64(0x0F0F0F0F0F0F0F0F)
    M3 = np.uint64(0x3333333333333333)
    M4 = np.uint64(0x5555555555555555)
    one = np.uint64(1)
    for i in range(n):
        xi = np.uint64(x[i])
        yi = np.uint64(y[i])
        xi = (xi | (xi << np.uint64(16))) & M0
        xi = (xi | (xi << np.uint64(8))) & M1
        xi = (xi | (xi << np.uint64(4))) & M2
        xi = (xi | (xi << np.uint64(2))) & M3
        xi = (xi | (xi << one)) & M4
        yi = (yi | (yi << np.uint64(16))) & M0
        yi = (yi | (yi << np.uint64(8))) & M1
        yi = (yi | (yi << np.uint64(4))) & M2
        yi = (yi | (yi << np.uint64(2))) & M3
        yi = (yi | (yi << one)) & M4
        out[i] = xi | (yi << one)
    return out


@numba.njit(cache=True)
def _fill_blocks_from_sorted(
    bx: lib.IntArray1D,
    by: lib.IntArray1D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
) -> None:
    """Fill ``block_starts``/``block_ends`` by single linear scan.

    Expects ``bx``/``by`` to be the block coordinates of each loc in the
    pyramid sort order; because the sort is by Morton at the finest
    level, locs sharing a block at *any* level form one contiguous run.
    """
    n = bx.shape[0]
    if n == 0:
        return
    cur_bx = bx[0]
    cur_by = by[0]
    block_starts[cur_by, cur_bx] = 0
    for k in range(1, n):
        if bx[k] != cur_bx or by[k] != cur_by:
            block_ends[cur_by, cur_bx] = k
            cur_bx = bx[k]
            cur_by = by[k]
            block_starts[cur_by, cur_bx] = k
    block_ends[cur_by, cur_bx] = n


def build_render_index(
    locs: pd.DataFrame,
    info: list[dict],
    n_levels: int = 3,
) -> RenderIndexPyramid | None:
    """Build the pyramid for one channel's locs.

    Parameters
    ----------
    locs : pd.DataFrame
        The localizations to index, with ``x`` and ``y`` columns.
    info : list of dicts
        Localizations metadata; "Width" and "Height" are required.
    n_levels : int, optional
        Number of pyramid levels, each with blocks 4x larger than the last.
        Default 3.

    Returns
    -------
    pyramid : RenderIndexPyramid or None
        ``None`` if required metadata is missing -- callers should fall back
        to the existing brute-force viewport filter in that case.
    """
    width = lib.get_from_metadata(info, "Width")
    height = lib.get_from_metadata(info, "Height")
    if width is None or height is None:
        return None
    width = float(width)
    height = float(height)

    base = _base_block_size(width, height)
    block_sizes = tuple(base * (4**lvl) for lvl in range(n_levels))

    n = len(locs)
    if n == 0:
        block_starts = []
        block_ends = []
        for size in block_sizes:
            K = max(1, int(np.ceil(height / size)))
            L = max(1, int(np.ceil(width / size)))
            block_starts.append(np.zeros((K, L), dtype=np.uint32))
            block_ends.append(np.zeros((K, L), dtype=np.uint32))
        return RenderIndexPyramid(
            perm=np.empty(0, dtype=np.uint32),
            block_sizes=block_sizes,
            block_starts=block_starts,
            block_ends=block_ends,
            width=width,
            height=height,
        )

    x = locs["x"].to_numpy()
    y = locs["y"].to_numpy()

    # Block coords at the finest level, clipped to the grid. Out-of-FOV
    # locs are pinned to the boundary so they stay queryable -- matches
    # the existing renderer, which just doesn't draw them.
    n_blocks_x0 = max(1, int(np.ceil(width / base)))
    n_blocks_y0 = max(1, int(np.ceil(height / base)))
    bx0 = np.clip(np.floor(x / base), 0, n_blocks_x0 - 1).astype(np.uint32)
    by0 = np.clip(np.floor(y / base), 0, n_blocks_y0 - 1).astype(np.uint32)

    # Sort by Morton at finest level -> hierarchical contiguity.
    keys = _morton_encode_2d(bx0, by0)
    perm = np.argsort(keys, kind="stable").astype(np.uint32)

    block_starts = []
    block_ends = []
    for size in block_sizes:
        L = max(1, int(np.ceil(width / size)))
        K = max(1, int(np.ceil(height / size)))
        bx_lvl = np.clip(np.floor(x[perm] / size), 0, L - 1).astype(np.uint32)
        by_lvl = np.clip(np.floor(y[perm] / size), 0, K - 1).astype(np.uint32)
        bs = np.zeros((K, L), dtype=np.uint32)
        be = np.zeros((K, L), dtype=np.uint32)
        _fill_blocks_from_sorted(bx_lvl, by_lvl, bs, be)
        block_starts.append(bs)
        block_ends.append(be)

    return RenderIndexPyramid(
        perm=perm,
        block_sizes=block_sizes,
        block_starts=block_starts,
        block_ends=block_ends,
        width=width,
        height=height,
    )


def _select_level(pyramid: RenderIndexPyramid, viewport: tuple) -> int:
    """Pick the smallest level whose blocks per viewport edge <= target.

    Walking from finest to coarsest means we pick the finest level that
    keeps block iteration bounded -- which also minimizes the gathered
    locs count (more blocks per coarse cell at coarser levels).
    """
    (y_min, x_min), (y_max, x_max) = viewport
    vp_dim = max(x_max - x_min, y_max - y_min)
    for lvl, size in enumerate(pyramid.block_sizes):
        if vp_dim / size <= _TARGET_BLOCKS_PER_EDGE:
            return lvl
    return len(pyramid.block_sizes) - 1


@numba.njit(cache=True)
def _gather_blocks(
    perm: lib.IntArray1D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    cy_min: int,
    cy_max: int,
    cx_min: int,
    cx_max: int,
) -> lib.IntArray1D:
    """Collect original-locs indices from all blocks in the rectangle."""
    total = 0
    for y in range(cy_min, cy_max + 1):
        for x in range(cx_min, cx_max + 1):
            total += block_ends[y, x] - block_starts[y, x]
    out = np.empty(total, dtype=np.uint32)
    pos = 0
    for y in range(cy_min, cy_max + 1):
        for x in range(cx_min, cx_max + 1):
            s = block_starts[y, x]
            e = block_ends[y, x]
            for k in range(s, e):
                out[pos] = perm[k]
                pos += 1
    return out


def query_viewport(
    pyramid: RenderIndexPyramid,
    viewport: tuple,
) -> lib.IntArray1D | None:
    """Indices into the original locs DataFrame for locs in the viewport.

    The returned set is a superset of the strictly-inside locs: a block
    at the viewport edge contributes all of its locs (the renderer's
    own ``in_view`` test inside ``_render_setup`` then prunes the
    overspill, on a tiny array).

    Returns ``None`` when the viewport covers (most of) the FOV --
    above ``_BYPASS_COVERAGE_RATIO`` of the FOV area, or fully
    enclosing it. In that regime gathering ~N indices and copying the
    DataFrame via ``iloc`` costs more than letting the renderer scan
    the full locs with its vectorised ``in_view`` mask. The caller
    treats ``None`` as "no pre-filter, use the full locs".

    Parameters
    ----------
    pyramid : RenderIndexPyramid
        The index built by :func:`build_render_index`.
    viewport : tuple
        ``((y_min, x_min), (y_max, x_max))`` in camera pixels.

    Returns
    -------
    indices : lib.IntArray1D or None
        Positions into the original locs DataFrame, or ``None`` for a
        (near-)full-FOV viewport, as described above.
    """
    (y_min, x_min), (y_max, x_max) = viewport
    # Bypass for (near-)full-FOV viewports -- see module-level constant.
    if (
        x_min <= 0.0
        and y_min <= 0.0
        and x_max >= pyramid.width
        and y_max >= pyramid.height
    ):
        return None
    fov_area = pyramid.width * pyramid.height
    if fov_area > 0.0:
        cx0 = max(0.0, x_min)
        cy0 = max(0.0, y_min)
        cx1 = min(pyramid.width, x_max)
        cy1 = min(pyramid.height, y_max)
        clipped_area = max(0.0, cx1 - cx0) * max(0.0, cy1 - cy0)
        if clipped_area / fov_area >= _BYPASS_COVERAGE_RATIO:
            return None

    if pyramid.perm.shape[0] == 0:
        return np.empty(0, dtype=np.uint32)

    return query_rect(pyramid, viewport)


def query_rect(pyramid: RenderIndexPyramid, rect: tuple) -> lib.IntArray1D:
    """Indices into the original locs DataFrame for locs in ``rect``.

    Unlike ``query_viewport`` there is no full-FOV bypass: the result is
    always an index array, a superset of the locs strictly inside the
    rectangle (whole blocks at its edges are included).

    Parameters
    ----------
    pyramid : RenderIndexPyramid
        The index built by :func:`build_render_index`.
    rect : tuple
        ``((y_min, x_min), (y_max, x_max))`` in camera pixels.

    Returns
    -------
    indices : lib.IntArray1D
        Positions into the original locs DataFrame.
    """
    (y_min, x_min), (y_max, x_max) = rect
    if pyramid.perm.shape[0] == 0:
        return np.empty(0, dtype=np.uint32)
    lvl = _select_level(pyramid, rect)
    size = pyramid.block_sizes[lvl]
    bs = pyramid.block_starts[lvl]
    be = pyramid.block_ends[lvl]
    K, L = bs.shape

    cx_min = int(np.floor(x_min / size))
    cy_min = int(np.floor(y_min / size))
    # x_max/y_max are exclusive in the existing renderer (strict ``<``),
    # so a value landing exactly on a block boundary belongs to the
    # previous block.
    cx_max = int(np.floor((x_max - 1e-9) / size))
    cy_max = int(np.floor((y_max - 1e-9) / size))
    cx_min = max(0, cx_min)
    cy_min = max(0, cy_min)
    cx_max = min(L - 1, cx_max)
    cy_max = min(K - 1, cy_max)
    if cx_min > cx_max or cy_min > cy_max:
        return np.empty(0, dtype=np.uint32)

    return _gather_blocks(pyramid.perm, bs, be, cy_min, cy_max, cx_min, cx_max)


@numba.njit(cache=True)
def _filter_circle(
    indices: lib.IntArray1D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    cx: float,
    cy: float,
    r2: float,
) -> lib.IntArray1D:
    """Keep the indices whose coordinates lie strictly within the
    circle (squared radius ``r2``), as ``lib.is_loc_at_numba`` does."""
    keep = np.empty(indices.shape[0], dtype=np.uint32)
    n = 0
    for k in range(indices.shape[0]):
        i = indices[k]
        dx = x[i] - cx
        dy = y[i] - cy
        if dx * dx + dy * dy < r2:
            keep[n] = i
            n += 1
    return keep[:n]


def query_circle(
    pyramid: RenderIndexPyramid,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    cx: float,
    cy: float,
    radius: float,
) -> lib.IntArray1D:
    """Indices into the original locs DataFrame for locs within a
    circular pick, the way ``picasso.postprocess.picked_locs`` selects
    them (``dx**2 + dy**2 < radius**2``).

    The blocks overlapping the circle's bounding box are gathered from
    the pyramid and the distance test is applied to those locs only.

    Parameters
    ----------
    pyramid : RenderIndexPyramid
        The index built by :func:`build_render_index` for ``x``, ``y``.
    x, y : lib.FloatArray1D
        Coordinates of all the localizations the pyramid indexes (the
        DataFrame's columns), in camera pixels.
    cx, cy : float
        Center of the pick in camera pixels.
    radius : float
        Radius of the pick in camera pixels.

    Returns
    -------
    indices : lib.IntArray1D
        Positions into the original locs DataFrame, in the pyramid's
        (Morton) order.
    """
    rect = ((cy - radius, cx - radius), (cy + radius, cx + radius))
    indices = query_rect(pyramid, rect)
    return _filter_circle(indices, x, y, float(cx), float(cy), radius**2)


# ---------------------------------------------------------------------------
# Persistence: the pyramid stored in the localizations' HDF5 file
# ---------------------------------------------------------------------------


def save_render_index(
    hdf_file: h5py.File, pyramid: RenderIndexPyramid
) -> None:
    """Write ``pyramid`` into an open HDF5 file as the group
    ``/render_index``: the permutation and every level's block tables
    as datasets, the block sizes, field size and row count as
    attributes. Older Picasso versions read only ``/locs`` and
    ``/metadata`` and are unaffected by the group.

    Parameters
    ----------
    hdf_file : h5py.File
        The localizations file, open for writing.
    pyramid : RenderIndexPyramid
        The index of the ``/locs`` rows of that file, in their order.
    """
    if RENDER_INDEX_GROUP in hdf_file:
        del hdf_file[RENDER_INDEX_GROUP]
    group = hdf_file.create_group(RENDER_INDEX_GROUP)
    group.attrs["version"] = _RENDER_INDEX_VERSION
    group.attrs["n"] = int(pyramid.perm.shape[0])
    group.attrs["width"] = float(pyramid.width)
    group.attrs["height"] = float(pyramid.height)
    group.attrs["block_sizes"] = np.asarray(
        pyramid.block_sizes, dtype=np.float64
    )
    group.create_dataset("perm", data=pyramid.perm)
    for lvl, (bs, be) in enumerate(
        zip(pyramid.block_starts, pyramid.block_ends)
    ):
        group.create_dataset(f"block_starts_{lvl}", data=bs)
        group.create_dataset(f"block_ends_{lvl}", data=be)


def read_render_index(path: str) -> RenderIndexPyramid | None:
    """Read the pyramid stored by ``save_render_index`` in the
    localizations file ``path``; None if the file has none (or it
    cannot be read). The result is *unchecked*: use
    ``load_render_index`` to get one that is known to describe the
    localizations.

    Parameters
    ----------
    path : str
        The localizations HDF5 file.

    Returns
    -------
    pyramid : RenderIndexPyramid or None
    """
    try:
        with h5py.File(path, "r") as hdf_file:
            if RENDER_INDEX_GROUP not in hdf_file:
                return None
            group = hdf_file[RENDER_INDEX_GROUP]
            if int(group.attrs.get("version", 0)) != _RENDER_INDEX_VERSION:
                return None
            block_sizes = tuple(float(s) for s in group.attrs["block_sizes"])
            perm = group["perm"][()].astype(np.uint32, copy=False)
            block_starts = []
            block_ends = []
            for lvl in range(len(block_sizes)):
                block_starts.append(
                    group[f"block_starts_{lvl}"][()].astype(
                        np.uint32, copy=False
                    )
                )
                block_ends.append(
                    group[f"block_ends_{lvl}"][()].astype(
                        np.uint32, copy=False
                    )
                )
            return RenderIndexPyramid(
                perm=perm,
                block_sizes=block_sizes,
                block_starts=block_starts,
                block_ends=block_ends,
                width=float(group.attrs["width"]),
                height=float(group.attrs["height"]),
            )
    except (OSError, KeyError, ValueError, TypeError):
        return None


@numba.njit(cache=True)
def _is_permutation(perm: lib.IntArray1D, n: int) -> bool:
    """Whether ``perm`` lists every index below ``n`` exactly once."""
    if perm.shape[0] != n:
        return False
    seen = np.zeros(n, dtype=np.uint8)
    for k in range(n):
        p = perm[k]
        if p >= n or seen[p]:
            return False
        seen[p] = 1
    return True


@numba.njit(cache=True)
def _blocks_hold_their_locs(
    perm: lib.IntArray1D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    size: float,
) -> bool:
    """Whether every block's range lists only localizations whose
    (clipped) block coordinates are that block, and the ranges cover
    all ``perm`` entries. This is the correctness criterion of the
    index: as long as it holds, every query is right."""
    n = perm.shape[0]
    K, L = block_starts.shape
    total = 0
    for i in range(K):
        for j in range(L):
            s = block_starts[i, j]
            e = block_ends[i, j]
            if e < s or e > n:
                return False
            total += e - s
            for k in range(s, e):
                p = perm[k]
                bx = int(np.floor(x[p] / size))
                by = int(np.floor(y[p] / size))
                bx = min(max(bx, 0), L - 1)
                by = min(max(by, 0), K - 1)
                if bx != j or by != i:
                    return False
    return total == n


def validate_render_index(
    pyramid: RenderIndexPyramid, locs: pd.DataFrame, info: list[dict]
) -> bool:
    """Whether ``pyramid`` correctly indexes ``locs``.

    Checked against the index's own correctness criterion rather than a
    checksum: the permutation covers every row exactly once and every
    block, at every level, holds only rows whose coordinates fall in it
    (one pass per level). Any edit of the file that changed, dropped,
    added or reordered coordinates fails; an edit that leaves the
    index correct (say, other columns) passes, which is what matters.

    Parameters
    ----------
    pyramid : RenderIndexPyramid
        A pyramid, e.g. read from the file by ``read_render_index``.
    locs : pd.DataFrame
        The localizations it claims to index, in file order.
    info : list of dicts
        Their metadata (the field size must match the pyramid's).

    Returns
    -------
    valid : bool
    """
    width = lib.get_from_metadata(info, "Width")
    height = lib.get_from_metadata(info, "Height")
    if width is None or height is None:
        return False
    if float(width) != pyramid.width or float(height) != pyramid.height:
        return False
    n = len(locs)
    if not _is_permutation(pyramid.perm, n):
        return False
    if len(pyramid.block_sizes) != len(pyramid.block_starts) or len(
        pyramid.block_starts
    ) != len(pyramid.block_ends):
        return False
    if n == 0:
        return True
    x = locs["x"].to_numpy()
    y = locs["y"].to_numpy()
    for size, bs, be in zip(
        pyramid.block_sizes, pyramid.block_starts, pyramid.block_ends
    ):
        K = max(1, int(np.ceil(pyramid.height / size)))
        L = max(1, int(np.ceil(pyramid.width / size)))
        if bs.shape != (K, L) or be.shape != (K, L):
            return False
        if not _blocks_hold_their_locs(
            pyramid.perm, bs, be, x, y, float(size)
        ):
            return False
    return True


def load_render_index(
    path: str, locs: pd.DataFrame, info: list[dict]
) -> RenderIndexPyramid | None:
    """The pyramid stored in ``path`` if it (still) describes ``locs``,
    else None -- the caller then builds one with
    ``build_render_index``. A stored index that fails the check (the
    file was edited without ``picasso.io.save_locs``) is reported in
    the log at INFO level.

    Parameters
    ----------
    path : str
        The localizations HDF5 file.
    locs : pd.DataFrame
        The localizations loaded from it.
    info : list of dicts
        Their metadata.

    Returns
    -------
    pyramid : RenderIndexPyramid or None
    """
    pyramid = read_render_index(path)
    if pyramid is None:
        return None
    if not validate_render_index(pyramid, locs, info):
        _log.info(
            "The render index stored in %s does not match its localizations "
            "(the file was modified without picasso.io.save_locs); it is "
            "rebuilt.",
            path,
        )
        return None
    return pyramid
