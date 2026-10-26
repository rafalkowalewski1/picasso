"""Test the multi-emitter fitting - ``localize.localize_multi_emitter``,
``localize.MultiEmitterParameters`` and ``picasso.fitting.residualfit``.

The fit is a port of 3D-DAOSTORM from storm-analysis, so the first test
holds it to the reference implementation's output on a stored movie. The
others check what the benchmark of the port measured on a small scale:
recovery of close emitters, no false splits of single ones, calibrated
uncertainties, and the conventions every Picasso fit follows.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.special import erf

from picasso import localize
from picasso.fitting import residualfit

from tests._progress_tracking import RecordingProgress, assert_phases_complete

PIXELSIZE = 130.0
SIGMA = 130.0  # nm, one camera pixel
CAMERA_INFO = {
    "Baseline": 0,
    "Sensitivity": 1,
    "Gain": 1,
    "Pixelsize": PIXELSIZE,
}
FIXED_SIGMA = localize.MultiEmitterParameters(psf_sigma=SIGMA)


def _movie(frames: np.ndarray):
    info = [
        {
            "Frames": len(frames),
            "Height": frames.shape[1],
            "Width": frames.shape[2],
            "Pixelsize": PIXELSIZE,
        }
    ]
    return localize._InMemoryMovie(frames, info), info


def _render(
    emitters,
    n_frames,
    size=48,
    background=20.0,
    seed=0,
    sigma=SIGMA,
    baseline=0.0,
):
    """A movie in photons: each emitter's Gaussian integrated over the
    pixels (pixel centers on integers), plus a flat background, with
    Poisson noise and an optional camera baseline."""
    edges = np.arange(size + 1) - 0.5
    expected = np.full((n_frames, size, size), float(background))
    s = sigma / PIXELSIZE * np.sqrt(2.0)
    for frame, x, y, photons in zip(
        emitters["frame"], emitters["x"], emitters["y"], emitters["photons"]
    ):
        ex = np.diff(erf((edges - x / PIXELSIZE) / s)) / 2.0
        ey = np.diff(erf((edges - y / PIXELSIZE) / s)) / 2.0
        expected[int(frame)] += photons * np.outer(ey, ex)
    movie = np.random.default_rng(seed).poisson(expected) + baseline
    return np.clip(movie, 0, 2**16 - 1).astype(np.uint16)


def _grid(n_frames, spacing, photons, size=48, seed=0):
    """Single emitters on a grid ``spacing`` nm apart, each placed at
    random within a pixel of its grid point in every frame."""
    rng = np.random.default_rng(seed)
    points = np.arange(spacing, size * PIXELSIZE - spacing + 1, spacing)
    gx, gy = (a.ravel() for a in np.meshgrid(points, points))
    n = len(gx)
    jitter = rng.uniform(-0.5, 0.5, (2, n_frames * n)) * PIXELSIZE
    return pd.DataFrame(
        {
            "frame": np.repeat(np.arange(n_frames), n).astype(np.uint32),
            "x": np.tile(gx, n_frames) + jitter[0],
            "y": np.tile(gy, n_frames) + jitter[1],
            "photons": float(photons),
        }
    )


def _match(locs, truth, tolerance):
    """One-to-one pairs of localizations and true emitters of the same
    frame closer than ``tolerance`` (nm), with the smallest summed
    distance. Returns the localization and truth indices."""
    loc_index, truth_index = [], []
    for frame in np.unique(truth["frame"]):
        li = np.flatnonzero(locs["frame"].to_numpy() == frame)
        ti = np.flatnonzero(truth["frame"].to_numpy() == frame)
        if not len(li) or not len(ti):
            continue
        a = locs[["x", "y"]].to_numpy(float)[li]
        b = truth[["x", "y"]].to_numpy(float)[ti]
        d = np.hypot(*(a[:, None] - b[None]).transpose(2, 0, 1))
        cost = np.where(d < tolerance, d, 1e9)
        rows, cols = linear_sum_assignment(cost)
        ok = cost[rows, cols] < tolerance
        loc_index += list(li[rows[ok]])
        truth_index += list(ti[cols[ok]])
    return np.array(loc_index, dtype=int), np.array(truth_index, dtype=int)


def _scores(locs, truth, tolerance):
    """Recall, precision, false positives and lateral RMSE (nm)."""
    li, ti = _match(locs, truth, tolerance)
    d2 = (locs["x"].to_numpy()[li] - truth["x"].to_numpy()[ti]) ** 2 + (
        locs["y"].to_numpy()[li] - truth["y"].to_numpy()[ti]
    ) ** 2
    return {
        "recall": len(li) / max(len(truth), 1),
        "precision": len(li) / max(len(locs), 1),
        "fp": len(locs) - len(li),
        "rmse_lateral": float(np.sqrt(d2.mean())) if len(li) else np.nan,
    }


def _emitters(rows) -> pd.DataFrame:
    """(frame, x px, y px, photons) rows as an emitter table in nm."""
    frame, x, y, photons = np.array(rows, dtype=float).T
    return pd.DataFrame(
        {
            "frame": frame.astype(np.uint32),
            "x": x * PIXELSIZE,
            "y": y * PIXELSIZE,
            "photons": photons,
        }
    )


def _fit(frames, parameters=FIXED_SIGMA, **kwargs):
    movie, info = _movie(frames)
    kwargs.setdefault("camera_info", CAMERA_INFO)
    kwargs.setdefault("parallel", False)
    return localize.localize_multi_emitter(
        movie, info, parameters=parameters, **kwargs
    )


@pytest.fixture(scope="module")
def singles():
    """Bright single emitters on a grid, 1500 nm apart, over 20 frames,
    and their ground truth."""
    emitters = _grid(20, spacing=1500.0, photons=2000.0, seed=1)
    return _render(emitters, 20, seed=2), emitters


# ----------------------------------------------------------------------
# The port against storm-analysis
# ----------------------------------------------------------------------


def test_reproduces_storm_analysis():
    """storm-analysis v2.4.1 (3D-DAOSTORM, 2dfixed, MLE) on
    tests/data/multi_emitter_movie.npy: three single emitters and two
    pairs (2 and 3 PSF sigma apart) in each of four frames. The port
    finds the same emitters at the same positions; it integrates the PSF
    over the pixels where storm-analysis samples it at their centers,
    with the width of the sampled Gaussian matched."""
    frames = np.load("tests/data/multi_emitter_movie.npy")
    reference = pd.read_csv(
        "tests/data/multi_emitter_storm_analysis.csv", comment="#"
    )
    reference = pd.DataFrame(
        {
            "frame": reference["frame"],
            "x": reference["x"] * PIXELSIZE,
            "y": reference["y"] * PIXELSIZE,
        }
    )
    # storm-analysis ran at its threshold of 6
    locs, _ = _fit(
        frames,
        parameters=localize.MultiEmitterParameters(
            psf_sigma=SIGMA, snr_threshold=6.0
        ),
    )
    assert len(locs) == len(reference)
    li, ti = _match(locs, reference, tolerance=10.0)
    assert len(li) == len(reference)
    distance = np.hypot(
        locs["x"].to_numpy()[li] - reference["x"].to_numpy()[ti],
        locs["y"].to_numpy()[li] - reference["y"].to_numpy()[ti],
    )
    assert distance.mean() < 1.0  # nm
    assert distance.max() < 5.0


# ----------------------------------------------------------------------
# Recovery, false splits and uncertainties
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_emitters", [1, 2, 3, 4])
def test_recovers_close_emitters(n_emitters):
    """1 to 4 emitters, neighbors 2.5 PSF sigma apart, are found;
    single-emitter fitting would merge them. Four in a square of 2.5
    PSF sigma are at the limit: over 40 frames, 91-98 % are found
    (seeds 0-5)."""
    rng = np.random.default_rng(n_emitters)
    rows = []
    n_frames = 40
    # the corners of a regular polygon with sides of 2.5 px
    radius = 0.0 if n_emitters == 1 else 2.5 / (2 * np.sin(np.pi / n_emitters))
    for frame in range(n_frames):
        angle = rng.uniform(0, 2 * np.pi)
        for k in range(n_emitters):
            a = angle + 2 * np.pi * k / n_emitters
            rows.append(
                (
                    frame,
                    24 + radius * np.cos(a),
                    24 + radius * np.sin(a),
                    3000,
                )
            )
    truth = _emitters(rows)
    locs, _ = _fit(_render(truth, n_frames, seed=n_emitters))
    result = _scores(locs, truth, tolerance=60.0)
    assert result["recall"] >= 0.9
    assert result["precision"] >= 0.85
    assert result["rmse_lateral"] < 25.0
    assert locs["n_emitters"].max() <= n_emitters + 1


def test_background_is_counted_once():
    """Two overlapping emitters share one background: each fits the
    background of the frame, not a fraction or a multiple of it (the
    ThunderSTORM multi-emitter model adds it once per emitter)."""
    rows = [(f, 22.0, 24.0, 3000) for f in range(10)]
    rows += [(f, 24.5, 24.0, 3000) for f in range(10)]
    locs, _ = _fit(_render(_emitters(rows), 10, background=50.0))
    assert len(locs) == 20
    assert locs["bg"].median() == pytest.approx(50.0, rel=0.05)
    assert locs["photons"].median() == pytest.approx(3000.0, rel=0.05)


def test_single_emitters_are_not_split(singles):
    frames, truth = singles
    locs, _ = _fit(frames)
    result = _scores(locs, truth, tolerance=100.0)
    assert result["recall"] == 1.0
    assert result["fp"] == 0
    assert (locs["n_emitters"] == 1).all()


def test_uncertainties_match_the_scatter(singles):
    frames, truth = singles
    locs, _ = _fit(frames)
    li, ti = _match(locs, truth, tolerance=100.0)
    zx = (locs["x"].to_numpy()[li] - truth["x"].to_numpy()[ti]) / locs[
        "lpx"
    ].to_numpy()[li]
    zy = (locs["y"].to_numpy()[li] - truth["y"].to_numpy()[ti]) / locs[
        "lpy"
    ].to_numpy()[li]
    assert np.std(np.r_[zx, zy]) == pytest.approx(1.0, abs=0.15)
    # the background a reduced chi-square of 1 implies
    assert locs["reduced_chi_square"].median() == pytest.approx(1.0, abs=0.1)


def test_no_emitters():
    frames = _render(_emitters(np.zeros((0, 4))), 3)
    locs, info = _fit(frames)
    assert len(locs) == 0
    assert {"frame", "x", "y", "lpx", "lpy"} <= set(locs.columns)


# ----------------------------------------------------------------------
# Conventions
# ----------------------------------------------------------------------


def test_columns_and_dtypes(singles):
    frames, _ = singles
    locs, info = _fit(frames)
    assert list(locs.columns) == [
        "frame",
        "x",
        "y",
        "photons",
        "sx",
        "sy",
        "bg",
        "lpx",
        "lpy",
        "photons_unc",
        "bg_unc",
        "log_likelihood",
        "iterations",
        "reduced_chi_square",
        "significance",
        "n_emitters",
    ]
    assert locs["frame"].dtype == np.uint32
    assert locs["x"].dtype == np.float32
    assert (np.diff(locs["frame"].to_numpy().astype(int)) >= 0).all()
    np.testing.assert_allclose(locs["sx"], SIGMA, rtol=1e-6)
    assert np.isfinite(locs.to_numpy(dtype=float)).all()
    # every new column is documented and offered in the save dialog
    assert {"significance", "n_emitters"} <= set(
        localize.LOCALIZATION_COLUMNS["Multi-emitter only"]
    )


def test_parallel_equals_serial(singles):
    frames, _ = singles
    serial, _ = _fit(frames, parallel=False)
    parallel, _ = _fit(frames, parallel=True)
    pd.testing.assert_frame_equal(serial, parallel)


def test_metadata(singles):
    frames, _ = singles
    _, info = _fit(frames)
    entry = info[-1]
    assert entry["Step"] == "Fit multi-emitter"
    assert entry["PSF Sigma (nm)"] == pytest.approx(SIGMA)
    assert entry["PSF Sigma Estimated"] is False
    assert entry["SNR Threshold"] == residualfit.THRESHOLD
    assert entry["Pixelsize"] == PIXELSIZE
    assert info[0]["Frames"] == len(frames)
    # the settings read back reproduce the fit
    parameters = localize.MultiEmitterParameters.from_info(entry)
    assert parameters == FIXED_SIGMA


def test_estimates_the_psf_width(singles):
    frames, truth = singles
    locs, info = _fit(frames, parameters=localize.MultiEmitterParameters())
    assert info[-1]["PSF Sigma Estimated"] is True
    assert info[-1]["PSF Sigma (nm)"] == pytest.approx(SIGMA, rel=0.03)
    assert len(locs) == len(truth)


def test_too_few_emitters_to_estimate_the_width():
    frames = _render(_emitters([(0, 24, 24, 3000)]), 1)
    with pytest.raises(ValueError, match="psf_sigma"):
        _fit(frames, parameters=localize.MultiEmitterParameters())


def test_progress_phases(singles):
    frames, _ = singles
    tracker = RecordingProgress()
    _fit(frames, progress=tracker)
    assert_phases_complete(tracker, ["Fitting frames"])
    tracker = RecordingProgress()
    _fit(
        frames,
        parameters=localize.MultiEmitterParameters(),
        progress=tracker,
        parallel=True,
    )
    assert_phases_complete(
        tracker, ["Estimating the PSF sigma", "Fitting frames"]
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_abort(singles, parallel):
    frames, _ = singles
    locs, info = _fit(frames, parallel=parallel, abort_callback=lambda: True)
    assert locs is None
    assert info[-1]["Step"] == "Fit multi-emitter"


def test_roi_and_frame_bounds(singles):
    frames, truth = singles
    roi = [((0, 0), (24, 48))]  # the top half
    locs, info = _fit(frames, roi=roi, frame_bounds=(5, 9))
    assert set(locs["frame"]) <= set(range(5, 10))
    assert (locs["y"] < 24 * PIXELSIZE).all()
    assert (locs["roi_id"] == 0).all()
    assert info[-1]["ROI"] == roi
    assert info[-1]["Frame Bounds"] == (5, 9)
    in_roi = truth[
        (truth["frame"] >= 5)
        & (truth["frame"] <= 9)
        & (truth["y"] < 23 * PIXELSIZE)
    ]
    li, _ = _match(locs, in_roi, tolerance=100.0)
    assert len(li) == len(in_roi)


def test_seed_identification(singles):
    """Seeding the first cycle with Picasso's identification finds the
    same emitters."""
    frames, truth = singles
    locs, info = _fit(
        frames,
        seed_identification=localize.IdentificationParameters(
            method=localize.NetGradient(2000), box=7
        ),
    )
    li, _ = _match(locs, truth, tolerance=100.0)
    assert len(li) == len(truth) == len(locs)
    assert info[-1]["Seed Identification"]["Box Size"] == 7


def test_scmos_variance(singles):
    """A zero readout variance changes nothing; a large one widens the
    uncertainties."""
    frames, _ = singles
    shape = frames.shape[1:]
    calibration = {
        "offset": np.zeros(shape),
        "variance": np.zeros(shape),
    }
    plain, _ = _fit(frames)
    zero, info = _fit(frames, camera_calibration=calibration)
    pd.testing.assert_frame_equal(plain, zero)
    assert info[-1]["Camera offset source"] == "per-pixel map"
    noisy, _ = _fit(
        frames,
        camera_calibration={**calibration, "variance": np.full(shape, 50.0)},
    )
    assert noisy["lpx"].median() > 1.1 * plain["lpx"].median()


def test_em_gain_doubles_the_variances(singles):
    frames, _ = singles
    plain, _ = _fit(frames)
    em, _ = _fit(
        frames, camera_info={**CAMERA_INFO, "Gain": 100, "Sensitivity": 100}
    )
    np.testing.assert_allclose(em["x"], plain["x"], atol=1e-3)
    np.testing.assert_allclose(
        em["lpx"], np.sqrt(2.0) * plain["lpx"], rtol=1e-4
    )


def test_requires_the_pixel_size(singles):
    frames, _ = singles
    camera_info = {k: v for k, v in CAMERA_INFO.items() if k != "Pixelsize"}
    with pytest.raises(ValueError, match="Pixelsize"):
        _fit(frames, camera_info=camera_info)


def test_rejects_other_settings(singles):
    frames, _ = singles
    with pytest.raises(TypeError, match="MultiEmitterParameters"):
        _fit(frames, parameters={"snr_threshold": 6})


# ----------------------------------------------------------------------
# MultiEmitterParameters
# ----------------------------------------------------------------------


class TestMultiEmitterParameters:
    def test_defaults_are_3d_daostorm(self):
        # except the threshold, 6 in storm-analysis
        parameters = localize.MultiEmitterParameters()
        assert parameters.snr_threshold == 15.0
        assert parameters.psf_sigma is None
        assert parameters.max_cycles == 20
        assert parameters.peak_radius == 5.0
        assert parameters.background_sigma == 8.0

    def test_frozen_and_keyword_only(self):
        parameters = localize.MultiEmitterParameters()
        with pytest.raises(dataclasses.FrozenInstanceError):
            parameters.snr_threshold = 3.0
        with pytest.raises(TypeError):
            localize.MultiEmitterParameters(6.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"snr_threshold": 0},
            {"snr_threshold": -1.0},
            {"snr_threshold": np.inf},
            {"psf_sigma": 0.0},
            {"psf_sigma": "130"},
            {"max_cycles": 0},
            {"max_cycles": 2.5},
            {"max_cycles": True},
            {"peak_radius": -1},
            {"background_sigma": 0},
        ],
    )
    def test_rejects_invalid(self, kwargs):
        with pytest.raises(ValueError):
            localize.MultiEmitterParameters(**kwargs)

    def test_info_round_trip(self):
        parameters = localize.MultiEmitterParameters(
            snr_threshold=8, psf_sigma=140, max_cycles=10
        )
        info = parameters.to_info()
        assert info["PSF Sigma (nm)"] == 140.0
        assert localize.MultiEmitterParameters.from_info(info) == parameters
        assert "PSF Sigma (nm)" not in (
            localize.MultiEmitterParameters().to_info()
        )
        assert (
            localize.MultiEmitterParameters.from_info({})
            == localize.MultiEmitterParameters()
        )


# ----------------------------------------------------------------------
# residualfit
# ----------------------------------------------------------------------


def test_fit_free_sigma_recovers_the_width():
    emitters = _grid(1, spacing=1300.0, photons=3000.0, size=60, seed=3)
    movie = _render(
        emitters, 1, size=60, background=10.0, seed=4, sigma=1.3 * PIXELSIZE
    )
    spots = []
    for x, y in zip(emitters["x"], emitters["y"]):
        cx = int(round(x / PIXELSIZE))
        cy = int(round(y / PIXELSIZE))
        spots.append(movie[0, cy - 4 : cy + 5, cx - 4 : cx + 5])
    thetas, deviance = residualfit.fit_free_sigma(np.array(spots))
    assert np.median(thetas[:, 3]) == pytest.approx(1.3, rel=0.02)
    assert np.isfinite(deviance).all()


def test_padding_covers_the_region():
    for sigma in (0.8, 1.0, 1.3, 2.0):
        assert residualfit.padding(sigma) >= residualfit.roi_size(sigma) / 2


# ----------------------------------------------------------------------
# Derivatives against finite differences
# ----------------------------------------------------------------------


def test_profile_derivative():
    size = 8
    e = np.empty((1, size))
    de = np.empty((1, size))
    e_plus = np.empty((1, size))
    e_minus = np.empty((1, size))
    h = 1e-6
    residualfit._profile(3.3, 0, size, 1.2, e[0], de[0])
    residualfit._profile(3.3 + h, 0, size, 1.2, e_plus[0], de[0].copy())
    residualfit._profile(3.3 - h, 0, size, 1.2, e_minus[0], de[0].copy())
    np.testing.assert_allclose(
        de[0], (e_plus[0] - e_minus[0]) / (2 * h), atol=1e-7
    )
    # the profile integrates the unit Gaussian over the pixels, here
    # [-0.5, 7.5]
    from scipy.special import erf

    expected = 0.5 * (
        erf((7.5 - 3.3) / (np.sqrt(2) * 1.2))
        - erf((-0.5 - 3.3) / (np.sqrt(2) * 1.2))
    )
    assert e[0].sum() == pytest.approx(expected, rel=1e-9)


def _one_peak(photons, x, y, bg, data):
    """The fit image of one peak in an otherwise empty frame and its
    deviance, as ``residualfit`` holds them."""
    sigma = 1.0
    roi = residualfit.roi_size(sigma)
    xoff = 0.5 * roi - 1.0
    center = 0.5 * roi
    dx, dy = [], []
    for i in range(roi):
        for j in range(roi):
            if (j - center + 0.5) ** 2 + (i - center + 0.5) ** 2 <= (
                center * center
            ):
                dx.append(j)
                dy.append(i)
    roi_dx = np.array(dx, dtype=np.int64)
    roi_dy = np.array(dy, dtype=np.int64)
    xi = int(np.floor(x - xoff))
    yi = int(np.floor(y - xoff))
    ex, dex, ey, dey = (np.empty(roi) for _ in range(4))
    psf = np.empty(len(roi_dx))
    residualfit._peak_shape(
        x, y, xi, yi, sigma, roi_dx, roi_dy, ex, dex, ey, dey, psf
    )
    f_data = np.zeros_like(data)
    bg_data = np.zeros_like(data)
    bg_counts = np.zeros(data.shape, dtype=np.int64)
    residualfit._add_peak(
        photons,
        bg,
        xi,
        yi,
        psf,
        roi_dx,
        roi_dy,
        np.zeros((1, 1)),
        False,
        f_data,
        bg_data,
        bg_counts,
        1,
    )
    ok, error = residualfit._peak_error(
        xi, yi, roi_dx, roi_dy, data, f_data, bg_data, bg_counts
    )
    return (
        error,
        (xi, yi, roi_dx, roi_dy, ex, dex, ey, dey, f_data, bg_data, bg_counts),
    )


def test_deviance_gradient():
    """The gradient each peak's Levenberg-Marquardt step follows is the
    gradient of the Poisson deviance of its region."""
    rng = np.random.default_rng(4)
    data = rng.poisson(20, (24, 24)).astype(np.float64) + 1.0
    theta = np.array([1500.0, 11.3, 12.6, 21.0])
    error, (xi, yi, roi_dx, roi_dy, ex, dex, ey, dey, f, bg, counts) = (
        _one_peak(*theta, data)
    )
    jac = np.empty(4)
    hess = np.empty((4, 4))
    residualfit._jacobian_hessian(
        theta[0],
        xi,
        yi,
        roi_dx,
        roi_dy,
        ex,
        dex,
        ey,
        dey,
        data,
        f,
        bg,
        counts,
        jac,
        hess,
    )
    for p, h in enumerate((1e-3, 1e-6, 1e-6, 1e-5)):
        plus, minus = theta.copy(), theta.copy()
        plus[p] += h
        minus[p] -= h
        numeric = (_one_peak(*plus, data)[0] - _one_peak(*minus, data)[0]) / (
            2 * h
        )
        assert jac[p] == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_free_sigma_gradient():
    rng = np.random.default_rng(5)
    spots = rng.poisson(30, (1, 9, 9)).astype(np.float32)
    spots[0, 3:6, 3:6] += 400
    theta = np.array([2500.0, 4.2, 3.9, 1.1, 28.0])
    box = 9
    buffers = [np.empty(box) for _ in range(6)]
    d = np.empty(5)

    def chi_square(t):
        hess = np.empty((5, 5))
        grad = np.empty(5)
        value, ok = residualfit._accumulate_free_sigma(
            spots,
            0,
            np.zeros((1, 1, 1), np.float32),
            False,
            t,
            *buffers,
            d,
            hess,
            grad,
        )
        return value, grad

    _, grad = chi_square(theta)
    for p, h in enumerate((1e-2, 1e-5, 1e-5, 1e-6, 1e-4)):
        plus, minus = theta.copy(), theta.copy()
        plus[p] += h
        minus[p] -= h
        numeric = (chi_square(plus)[0] - chi_square(minus)[0]) / (2 * h)
        # the accumulators return minus half the gradient (the Gpufit
        # convention of picasso.fitting.gaussfit)
        assert -2 * grad[p] == pytest.approx(numeric, rel=1e-4, abs=1e-4)
