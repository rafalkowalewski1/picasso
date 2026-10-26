"""
picasso._ground_truth
~~~~~~~~~~~~~~~~~~~~~

Compare localizations with the ground truth of a simulated movie, and
simulate movies with a known ground truth.

Private: these are the tools of the multi-emitter fitting benchmark
(``benchmarks/multi_emitter``) and of the tests that recover simulated
emitters. Nothing here is part of the public API.

Everything works in Picasso's frame: lengths in nm, pixel centers on
integer camera pixels (pixel ``i`` spans ``[i - 0.5, i + 0.5)``), frames
counted from 0. A ground truth is a localization table with the columns
``frame``, ``x`` and ``y`` (and optionally ``z`` and ``photons``), e.g.,
the one Picasso: Simulate saves next to its movies (see
``simulate.ground_truth_locs``).

Matching
    :func:`match` pairs localizations with ground-truth emitters of the
    same frame one to one, within a tolerance, minimizing the summed
    distance (the Hungarian algorithm, as the SMLM challenges do).
    :func:`metrics` turns the pairs into the detection and accuracy
    figures (Jaccard index, recall, precision, RMSE, ...) and
    :func:`gt_distances` into a per-localization column.
Movies
    :func:`render_movie` turns a table of emitters into a movie with
    Simulate's noise model: each photon lands at a normally distributed
    position, the photons are binned into the camera pixels, and a
    Poisson background is added. :func:`paint_emitters`,
    :func:`pair_emitters` and :func:`random_emitters` build such tables.
SMLM Challenge 2016
    :func:`load_smlm_challenge_ground_truth` reads the ground truth of
    its simulated datasets (Sage et al., Nat. Methods 16, 387-395,
    2019).

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import simulate


class Matches(NamedTuple):
    """Pairs of localizations and ground-truth emitters, see
    :func:`match`."""

    #: Row positions (not index labels) of the matched localizations.
    loc_index: np.ndarray
    #: Row positions of the ground-truth emitters they are paired with.
    truth_index: np.ndarray
    #: Lateral distance of each pair in nm.
    distance: np.ndarray


def _candidate_pairs(
    locs: pd.DataFrame,
    truth: pd.DataFrame,
    tolerance: float,
    z_tolerance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (localization, emitter) pair of the same frame closer than
    ``tolerance`` (and, with ``z_tolerance``, closer than that in z)."""
    # The frame becomes a third coordinate, spaced further apart than the
    # tolerance, so that one tree query finds the pairs of every frame.
    spacing = 2.0 * tolerance + 1.0
    points = []
    for table in (locs, truth):
        points.append(
            np.column_stack(
                [
                    table["x"].to_numpy(np.float64),
                    table["y"].to_numpy(np.float64),
                    table["frame"].to_numpy(np.float64) * spacing,
                ]
            )
        )
    pairs = cKDTree(points[0]).sparse_distance_matrix(
        cKDTree(points[1]), tolerance, output_type="ndarray"
    )
    i, j, d = pairs["i"], pairs["j"], pairs["v"]
    if z_tolerance is not None:
        dz = locs["z"].to_numpy()[i] - truth["z"].to_numpy()[j]
        keep = np.abs(dz) <= z_tolerance
        i, j, d = i[keep], j[keep], d[keep]
    return i, j, d


def match(
    locs: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    tolerance: float,
    z_tolerance: float | None = None,
) -> Matches:
    """Pair localizations with the ground-truth emitters they found.

    Within each frame, every localization is paired with at most one
    emitter and vice versa. Only pairs closer than ``tolerance`` qualify,
    and of all the assignments that pair as many as possible, the one
    with the smallest summed distance is taken (the Hungarian algorithm,
    solved for each cluster of competing candidates).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations, with ``frame``, ``x`` and ``y`` (nm) and, with
        ``z_tolerance``, ``z`` (nm).
    truth : pd.DataFrame
        The ground truth, with the same columns.
    tolerance : float
        Largest lateral distance (nm) of a pair.
    z_tolerance : float, optional
        Largest axial distance (nm) of a pair. None (the default) ignores
        z.

    Returns
    -------
    matches : Matches
        The pairs, ordered by localization.
    """
    n_locs = len(locs)
    n_truth = len(truth)
    empty = Matches(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
    )
    if n_locs == 0 or n_truth == 0:
        return empty
    i, j, d = _candidate_pairs(locs, truth, tolerance, z_tolerance)
    if len(i) == 0:
        return empty
    # Clusters of localizations and emitters linked by candidate pairs;
    # each is an independent assignment problem.
    graph = coo_matrix(
        (np.ones(len(i)), (i, n_locs + j)),
        shape=(n_locs + n_truth, n_locs + n_truth),
    )
    _, labels = connected_components(graph, directed=False)
    cluster = labels[i]
    pairs_per_cluster = np.bincount(cluster, minlength=labels.max() + 1)

    loc_index = []
    truth_index = []
    distance = []
    # The most common case by far: one localization, one emitter.
    alone = pairs_per_cluster[cluster] == 1
    loc_index.append(i[alone])
    truth_index.append(j[alone])
    distance.append(d[alone])

    shared = ~alone
    if np.any(shared):
        order = np.argsort(cluster[shared], kind="stable")
        si, sj, sd = i[shared][order], j[shared][order], d[shared][order]
        sc = cluster[shared][order]
        bounds = np.flatnonzero(np.diff(sc)) + 1
        for ci, cj, cd in zip(
            np.split(si, bounds), np.split(sj, bounds), np.split(sd, bounds)
        ):
            rows, row_of = np.unique(ci, return_inverse=True)
            cols, col_of = np.unique(cj, return_inverse=True)
            # A forbidden pair costs more than any assignment of allowed
            # ones, so the assignment pairs as many as it can first.
            forbidden = float(tolerance) * (len(rows) + len(cols) + 1)
            cost = np.full((len(rows), len(cols)), forbidden)
            cost[row_of, col_of] = cd
            r, c = linear_sum_assignment(cost)
            allowed = cost[r, c] < forbidden
            loc_index.append(rows[r[allowed]])
            truth_index.append(cols[c[allowed]])
            distance.append(cost[r[allowed], c[allowed]])

    loc_index = np.concatenate(loc_index)
    order = np.argsort(loc_index, kind="stable")
    return Matches(
        loc_index[order],
        np.concatenate(truth_index)[order],
        np.concatenate(distance)[order],
    )


def gt_distances(n_locs: int, matches: Matches) -> np.ndarray:
    """Distance (nm) of each localization to the ground-truth emitter it
    is paired with, -1 for those paired with none (false positives).

    Parameters
    ----------
    n_locs : int
        Number of localizations that were matched.
    matches : Matches
        The result of :func:`match`.

    Returns
    -------
    gt_dist : np.ndarray
        ``(n_locs,)`` float32. -1 rather than NaN, which
        ``lib.ensure_sanity`` would drop when saving.
    """
    gt_dist = np.full(n_locs, -1.0, dtype=np.float32)
    gt_dist[matches.loc_index] = matches.distance
    return gt_dist


def metrics(
    locs: pd.DataFrame,
    truth: pd.DataFrame,
    matches: Matches,
) -> dict:
    """Detection and accuracy of localizations against the ground truth.

    Parameters
    ----------
    locs, truth : pd.DataFrame
        The localizations and the ground truth that were matched.
    matches : Matches
        The result of :func:`match`.

    Returns
    -------
    metrics : dict
        ``n_locs``, ``n_truth``, true positives ``tp``, false positives
        ``fp``, false negatives ``fn``, ``recall`` (tp / n_truth),
        ``precision`` (tp / n_locs), ``jaccard`` (tp / (tp + fp + fn)),
        the root mean squared lateral error ``rmse_lateral`` (nm, of the
        radial distance), ``rmse_x``, ``rmse_y``, the mean errors
        ``bias_x`` and ``bias_y`` (nm), with ``z`` in both tables
        ``rmse_z`` and ``bias_z``, and with ``photons`` in both the
        median ratio of the fitted to the true photons
        ``photon_ratio``. ``efficiency`` combines the Jaccard index and
        the lateral RMSE as the SMLM challenge 2016 did, ``100 -
        sqrt((100 - jaccard_percent)^2 + rmse_lateral^2)``. The errors
        are NaN without pairs.
    """
    n_locs = len(locs)
    n_truth = len(truth)
    tp = len(matches.loc_index)
    fp = n_locs - tp
    fn = n_truth - tp
    result = {
        "n_locs": n_locs,
        "n_truth": n_truth,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / n_truth if n_truth else np.nan,
        "precision": tp / n_locs if n_locs else np.nan,
        "jaccard": tp / (tp + fp + fn) if tp + fp + fn else np.nan,
    }
    li, ti = matches.loc_index, matches.truth_index
    with np.errstate(invalid="ignore"):
        dx = (
            locs["x"].to_numpy(np.float64)[li]
            - truth["x"].to_numpy(np.float64)[ti]
        )
        dy = (
            locs["y"].to_numpy(np.float64)[li]
            - truth["y"].to_numpy(np.float64)[ti]
        )
        mean = np.mean if tp else (lambda _: np.nan)
        result["rmse_lateral"] = float(np.sqrt(mean(dx**2 + dy**2)))
        result["rmse_x"] = float(np.sqrt(mean(dx**2)))
        result["rmse_y"] = float(np.sqrt(mean(dy**2)))
        result["bias_x"] = float(mean(dx))
        result["bias_y"] = float(mean(dy))
        if "z" in locs.columns and "z" in truth.columns:
            dz = (
                locs["z"].to_numpy(np.float64)[li]
                - truth["z"].to_numpy(np.float64)[ti]
            )
            result["rmse_z"] = float(np.sqrt(mean(dz**2)))
            result["bias_z"] = float(mean(dz))
        if "photons" in locs.columns and "photons" in truth.columns:
            true_photons = truth["photons"].to_numpy(np.float64)[ti]
            ratio = locs["photons"].to_numpy(np.float64)[li] / true_photons
            ratio = ratio[np.isfinite(ratio)]
            result["photon_ratio"] = (
                float(np.median(ratio)) if len(ratio) else np.nan
            )
    jaccard_percent = 100.0 * result["jaccard"]
    result["efficiency"] = 100.0 - np.sqrt(
        (100.0 - jaccard_percent) ** 2 + result["rmse_lateral"] ** 2
    )
    return result


# ---------------------------------------------------------------------------
# Simulated movies with a known ground truth
# ---------------------------------------------------------------------------


def render_movie(
    emitters: pd.DataFrame,
    *,
    n_frames: int,
    width: int,
    height: int,
    pixelsize: float,
    sigma: float,
    background: float,
    rng: np.random.Generator,
    baseline: float = 0.0,
) -> np.ndarray:
    """Simulate a movie of emitters with Picasso: Simulate's noise model.

    Each emitter sends its photons to positions drawn from a normal
    distribution around it; the photons are binned into the camera
    pixels, which makes the expected image the pixel-integrated Gaussian,
    and every pixel gets a Poisson background. The movie is in photons,
    so localize it with a ``Sensitivity`` and ``Gain`` of 1.

    Parameters
    ----------
    emitters : pd.DataFrame
        One row per emitter and frame: ``frame``, ``x`` and ``y`` (nm,
        Picasso's frame) and ``photons``, the number of photons emitted
        in that frame. Optional ``sx`` and ``sy`` (nm) give the emitter
        its own PSF sigma in x and y, e.g., for an astigmatic or
        mismatched PSF.
    n_frames, width, height : int
        Size of the movie, in frames and camera pixels.
    pixelsize : float
        Camera pixel size in nm.
    sigma : float
        PSF standard deviation in nm, for the emitters without ``sx`` and
        ``sy``.
    background : float
        Mean background in photons per pixel and frame.
    rng : np.random.Generator
        Random number generator.
    baseline : float, optional
        Camera offset added to every pixel, in counts. Default is 0.

    Returns
    -------
    movie : np.ndarray
        ``(n_frames, height, width)`` uint16 movie.
    """
    movie = rng.poisson(background, (n_frames, height, width)).astype(
        np.float64
    )
    if len(emitters):
        frame = emitters["frame"].to_numpy(np.int64)
        photons = np.rint(emitters["photons"].to_numpy(np.float64)).astype(
            np.int64
        )
        sx = (
            emitters["sx"].to_numpy(np.float64)
            if "sx" in emitters.columns
            else np.full(len(emitters), float(sigma))
        )
        sy = (
            emitters["sy"].to_numpy(np.float64)
            if "sy" in emitters.columns
            else np.full(len(emitters), float(sigma))
        )
        n = np.maximum(photons, 0)
        source = np.repeat(np.arange(len(emitters)), n)
        # positions in camera pixels, pixel centers on integers
        px = emitters["x"].to_numpy(np.float64)[source] / pixelsize
        py = emitters["y"].to_numpy(np.float64)[source] / pixelsize
        px = px + rng.standard_normal(len(source)) * sx[source] / pixelsize
        py = py + rng.standard_normal(len(source)) * sy[source] / pixelsize
        column = np.floor(px + 0.5).astype(np.int64)
        row = np.floor(py + 0.5).astype(np.int64)
        inside = (column >= 0) & (column < width) & (row >= 0) & (row < height)
        flat = (
            frame[source][inside] * (height * width)
            + row[inside] * width
            + column[inside]
        )
        counts = np.bincount(flat, minlength=n_frames * height * width)
        movie += counts.reshape(n_frames, height, width)
    movie += baseline
    return np.clip(np.rint(movie), 0, 2**16 - 1).astype(np.uint16)


def random_emitters(
    rng: np.random.Generator,
    *,
    n_frames: int,
    density: float,
    width: int,
    height: int,
    pixelsize: float,
    photons: float | tuple[float, float],
    margin: float = 0.0,
) -> pd.DataFrame:
    """Emitters placed uniformly at random, independently in every frame.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n_frames : int
        Number of frames.
    density : float
        Mean number of emitters per square micrometer and frame; the count
        per frame is Poisson distributed.
    width, height : int
        Size of the movie in camera pixels.
    pixelsize : float
        Camera pixel size in nm.
    photons : float or (float, float)
        Photons per emitter, or the mean and standard deviation of a
        normal distribution they are drawn from (floored at 1).
    margin : float, optional
        Distance (nm) from the edges of the field of view kept free of
        emitters. Default is 0.

    Returns
    -------
    emitters : pd.DataFrame
        ``frame``, ``x``, ``y`` (nm) and ``photons``, sorted by frame.
    """
    x_max = width * pixelsize
    y_max = height * pixelsize
    area_um2 = (x_max - 2 * margin) * (y_max - 2 * margin) / 1e6
    counts = rng.poisson(density * area_um2, n_frames)
    n = int(counts.sum())
    frame = np.repeat(np.arange(n_frames), counts)
    # pixel 0 spans [-0.5, 0.5) px
    x = rng.uniform(margin, x_max - margin, n) - 0.5 * pixelsize
    y = rng.uniform(margin, y_max - margin, n) - 0.5 * pixelsize
    return pd.DataFrame(
        {
            "frame": frame.astype(np.uint32),
            "x": x,
            "y": y,
            "photons": _draw_photons(rng, photons, n),
        }
    )


def _draw_photons(
    rng: np.random.Generator, photons: float | tuple[float, float], n: int
) -> np.ndarray:
    """``n`` photon counts, fixed or normally distributed."""
    if np.ndim(photons) == 0:
        return np.full(n, float(photons))
    mean, std = photons
    return np.maximum(rng.normal(mean, std, n), 1.0)


def pair_emitters(
    rng: np.random.Generator,
    *,
    n_frames: int,
    separation: float,
    width: int,
    height: int,
    pixelsize: float,
    spacing: float,
    photons: float | tuple[float, float],
    photon_ratio: float = 1.0,
) -> pd.DataFrame:
    """Pairs of emitters (or single emitters) on a grid.

    The pairs sit on a square grid, ``spacing`` apart, so that they do not
    overlap each other. Every frame, each pair gets a new random position
    within one pixel of its grid point and a new random orientation.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n_frames : int
        Number of frames.
    separation : float
        Distance between the two emitters of a pair in nm. 0 places single
        emitters instead.
    width, height : int
        Size of the movie in camera pixels.
    pixelsize : float
        Camera pixel size in nm.
    spacing : float
        Distance of the grid points in nm.
    photons : float or (float, float)
        Photons of the first emitter of a pair, or the mean and standard
        deviation of a normal distribution they are drawn from.
    photon_ratio : float, optional
        Photons of the second emitter relative to the first. Default is 1.

    Returns
    -------
    emitters : pd.DataFrame
        ``frame``, ``x``, ``y`` (nm), ``photons`` and ``pair``, the index
        of the pair, the same for both emitters, sorted by frame.
    """
    grid_x = np.arange(spacing, width * pixelsize - spacing + 1, spacing)
    grid_y = np.arange(spacing, height * pixelsize - spacing + 1, spacing)
    cx, cy = np.meshgrid(grid_x, grid_y)
    cx = cx.ravel()
    cy = cy.ravel()
    n_pairs = len(cx)
    frame = np.repeat(np.arange(n_frames), n_pairs)
    center_x = (
        np.tile(cx, n_frames)
        + rng.uniform(-0.5, 0.5, n_frames * n_pairs) * pixelsize
        - 0.5 * pixelsize
    )
    center_y = (
        np.tile(cy, n_frames)
        + rng.uniform(-0.5, 0.5, n_frames * n_pairs) * pixelsize
        - 0.5 * pixelsize
    )
    pair = np.arange(n_frames * n_pairs)
    first = _draw_photons(rng, photons, n_frames * n_pairs)
    if separation == 0:
        table = {
            "frame": frame,
            "x": center_x,
            "y": center_y,
            "photons": first,
            "pair": pair,
        }
    else:
        angle = rng.uniform(0, np.pi, n_frames * n_pairs)
        half_x = 0.5 * separation * np.cos(angle)
        half_y = 0.5 * separation * np.sin(angle)
        table = {
            "frame": np.concatenate([frame, frame]),
            "x": np.concatenate([center_x - half_x, center_x + half_x]),
            "y": np.concatenate([center_y - half_y, center_y + half_y]),
            "photons": np.concatenate([first, first * photon_ratio]),
            "pair": np.concatenate([pair, pair]),
        }
    emitters = pd.DataFrame(table)
    emitters["frame"] = emitters["frame"].astype(np.uint32)
    return emitters.sort_values(
        ["frame", "pair"], kind="stable", ignore_index=True
    )


def paint_emitters(
    *,
    n_frames: int,
    site_density: float,
    width: int,
    height: int,
    pixelsize: float,
    mean_bright: float,
    mean_dark: float,
    photon_rate: float,
    photon_rate_std: float = 0.0,
    integration_time: float = 100.0,
    margin: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Binding sites with DNA-PAINT kinetics, dense if asked to.

    Sites are placed uniformly at random and blink with the kinetics of
    Picasso: Simulate (:func:`picasso.simulate.distribute_photons`):
    exponentially distributed bright and dark times, so binding starts
    and ends within a frame and the photons of the first and last frame of
    an event are only a fraction of the photon rate. On average a fraction
    ``mean_bright / (mean_bright + mean_dark)`` of the sites emits in a
    frame, which with ``site_density`` sets the density of emitters.

    Parameters
    ----------
    n_frames : int
        Number of frames.
    site_density : float
        Binding sites per square micrometer.
    width, height : int
        Size of the movie in camera pixels.
    pixelsize : float
        Camera pixel size in nm.
    mean_bright, mean_dark : float
        Mean bright and dark times in ms.
    photon_rate : float
        Photons per ms of an emitting site, so ``photon_rate *
        integration_time`` per full frame. Set explicitly: Simulate's
        default changed with the brightness fix of Oct 2026.
    photon_rate_std : float, optional
        Standard deviation of the photon rate between binding events.
        Default is 0.
    integration_time : float, optional
        Exposure time of a frame in ms. Default is 100.
    margin : float, optional
        Distance (nm) from the edges kept free of sites. Default is 0.
    seed : int, optional
        Seed of NumPy's global random state, which Simulate's kinetics
        draw from. Default is 0.

    Returns
    -------
    emitters : pd.DataFrame
        ``frame``, ``x``, ``y`` (nm), ``photons`` and ``site``, sorted by
        frame.
    """
    np.random.seed(seed)
    x_max = width * pixelsize
    y_max = height * pixelsize
    area_um2 = (x_max - 2 * margin) * (y_max - 2 * margin) / 1e6
    n_sites = np.random.poisson(site_density * area_um2)
    x = np.random.uniform(margin, x_max - margin, n_sites) - 0.5 * pixelsize
    y = np.random.uniform(margin, y_max - margin, n_sites) - 0.5 * pixelsize
    frames = []
    sites = []
    photons = []
    for site in range(n_sites):
        photonsinframe, _, _ = simulate.distribute_photons(
            None,
            integration_time,
            n_frames,
            mean_dark,
            mean_bright,
            photon_rate,
            photon_rate_std,
            np.inf,
        )
        emitting = np.flatnonzero(photonsinframe > 0)
        frames.append(emitting)
        sites.append(np.full(len(emitting), site))
        photons.append(photonsinframe[emitting])
    frame = np.concatenate(frames) if frames else np.array([], dtype=int)
    site = np.concatenate(sites) if sites else np.array([], dtype=int)
    emitters = pd.DataFrame(
        {
            "frame": frame.astype(np.uint32),
            "x": x[site],
            "y": y[site],
            "photons": (
                np.concatenate(photons).astype(np.float64)
                if photons
                else np.array([], dtype=np.float64)
            ),
            "site": site,
        }
    )
    return emitters.sort_values(
        ["frame", "site"], kind="stable", ignore_index=True
    )


# ---------------------------------------------------------------------------
# SMLM Challenge 2016
# ---------------------------------------------------------------------------


def load_smlm_challenge_ground_truth(
    path: str, *, pixelsize: float = 100.0
) -> pd.DataFrame:
    """Read the ground truth of an SMLM Challenge 2016 dataset.

    The challenge's ``activations.csv`` lists one emitter per row: an
    index, the frame (counted from 1), x, y and z in nm with the origin at
    the corner of the first pixel, and the number of photons. Picasso
    counts frames from 0 and puts the origin at the center of the first
    pixel.

    Parameters
    ----------
    path : str
        Path of ``activations.csv``.
    pixelsize : float, optional
        Camera pixel size in nm, 100 for every dataset of the challenge.

    Returns
    -------
    truth : pd.DataFrame
        ``frame``, ``x``, ``y``, ``z`` (nm) and ``photons`` in Picasso's
        frame, sorted by frame.
    """
    # The files start with the header "Ground-truth,frame,xnano,ynano,
    # znano,intensity" and pad the values with spaces.
    table = pd.read_csv(path, skipinitialspace=True)
    table.columns = range(table.shape[1])
    truth = pd.DataFrame(
        {
            "frame": (table[1].to_numpy() - 1).astype(np.uint32),
            "x": table[2].to_numpy(np.float64) - 0.5 * pixelsize,
            "y": table[3].to_numpy(np.float64) - 0.5 * pixelsize,
            "z": table[4].to_numpy(np.float64),
            "photons": table[5].to_numpy(np.float64),
        }
    )
    return truth.sort_values("frame", kind="stable", ignore_index=True)
