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
MODELS = list(residualfit.PSF_MODELS)


def _model(model):
    """``FIXED_SIGMA`` with the PSF model ``model``."""
    return dataclasses.replace(FIXED_SIGMA, psf_model=model)


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
    Poisson noise and an optional camera baseline. The emitters' PSF
    sigmas (nm) are ``sigma``, or their ``sx`` and ``sy`` columns."""
    edges = np.arange(size + 1) - 0.5
    expected = np.full((n_frames, size, size), float(background))
    n = len(emitters["x"])
    sigmas_x = emitters["sx"] if "sx" in emitters else np.full(n, sigma)
    sigmas_y = emitters["sy"] if "sy" in emitters else sigmas_x
    for frame, x, y, photons, sx, sy in zip(
        emitters["frame"],
        emitters["x"],
        emitters["y"],
        emitters["photons"],
        sigmas_x,
        sigmas_y,
    ):
        s_x = sx / PIXELSIZE * np.sqrt(2.0)
        s_y = sy / PIXELSIZE * np.sqrt(2.0)
        ex = np.diff(erf((edges - x / PIXELSIZE) / s_x)) / 2.0
        ey = np.diff(erf((edges - y / PIXELSIZE) / s_y)) / 2.0
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


@pytest.mark.parametrize(
    "model, storm_model", [("spherical", "2d"), ("elliptical", "3d")]
)
def test_reproduces_storm_analysis_with_fitted_sigmas(model, storm_model):
    """storm-analysis v2.4.1 (3D-DAOSTORM, models 2d and 3d, MLE) on
    tests/data/multi_emitter_widths_movie.npy: per frame, two round
    emitters narrower and wider than the nominal sigma, two astigmatic
    ones and a pair 2.7 sigma apart. The port finds the same emitters
    with the same positions, sigmas and photons; storm-analysis's sigmas
    are those of the sampled Gaussian, sqrt(s^2 + 1/12) for the pixel-
    integrated PSF of sigma s. Both models are as the reference fits
    them: the round one splits the astigmatic emitters in two, the
    elliptical one takes one of the pairs for a single wide emitter."""
    frames = np.load("tests/data/multi_emitter_widths_movie.npy")
    reference = pd.read_csv(
        f"tests/data/multi_emitter_widths_storm_{storm_model}.csv",
        comment="#",
    )
    reference = pd.DataFrame(
        {
            "frame": reference["frame"],
            "x": reference["x"] * PIXELSIZE,
            "y": reference["y"] * PIXELSIZE,
            "sx": np.sqrt(reference["xsigma"] ** 2 - 1 / 12) * PIXELSIZE,
            "sy": np.sqrt(reference["ysigma"] ** 2 - 1 / 12) * PIXELSIZE,
            "photons": reference["sum"],
        }
    )
    locs, _ = _fit(
        frames,
        parameters=localize.MultiEmitterParameters(
            psf_model=model, psf_sigma=SIGMA, snr_threshold=6.0
        ),
    )
    li, ti = _match(locs, reference, tolerance=30.0)
    # at a threshold of 6, one dim peak next to a pair (significance 6.6)
    # is kept by storm-analysis only
    assert len(li) >= len(reference) - 1
    assert len(locs) - len(li) <= 1
    distance = np.hypot(
        locs["x"].to_numpy()[li] - reference["x"].to_numpy()[ti],
        locs["y"].to_numpy()[li] - reference["y"].to_numpy()[ti],
    )
    assert np.median(distance) < 0.5  # nm
    for column in ("sx", "sy"):
        difference = (
            locs[column].to_numpy()[li] - reference[column].to_numpy()[ti]
        )
        assert np.median(np.abs(difference)) < 1.0  # nm
    ratio = (
        locs["photons"].to_numpy()[li] / reference["photons"].to_numpy()[ti]
    )
    assert np.median(np.abs(ratio - 1)) < 0.005


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


@pytest.mark.parametrize("model", MODELS)
def test_single_emitters_are_not_split(singles, model):
    frames, truth = singles
    locs, _ = _fit(frames, parameters=_model(model))
    result = _scores(locs, truth, tolerance=100.0)
    assert result["recall"] == 1.0
    assert result["fp"] == 0
    assert (locs["n_emitters"] == 1).all()


@pytest.mark.parametrize("model", MODELS)
def test_uncertainties_match_the_scatter(singles, model):
    frames, truth = singles
    locs, _ = _fit(frames, parameters=_model(model))
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


# ----------------------------------------------------------------------
# PSF models
# ----------------------------------------------------------------------


def _wide_singles(sx, sy):
    """Single emitters on a grid whose PSF sigmas (nm) differ from the
    nominal ``SIGMA``."""
    emitters = _grid(20, spacing=1500.0, photons=3000.0, seed=1)
    emitters = emitters.assign(sx=sx, sy=sy)
    return _render(emitters, 20, seed=2), emitters


@pytest.mark.parametrize(
    "model, sx, sy",
    [("spherical", 170.0, 170.0), ("elliptical", 170.0, 110.0)],
)
def test_fitted_sigmas_recover_the_psf(model, sx, sy):
    frames, truth = _wide_singles(sx, sy)
    locs, _ = _fit(frames, parameters=_model(model))
    result = _scores(locs, truth, tolerance=100.0)
    assert result["recall"] == 1.0
    assert result["fp"] == 0
    assert locs["sx"].median() == pytest.approx(sx, rel=0.02)
    assert locs["sy"].median() == pytest.approx(sy, rel=0.02)
    assert locs["photons"].median() == pytest.approx(3000, rel=0.03)
    # a fitted sigma stays within 0.5 to 5 times the nominal one
    for column in ("sx", "sy"):
        assert locs[column].between(0.5 * SIGMA, 5 * SIGMA).all()
    if model == "spherical":
        np.testing.assert_array_equal(locs["sx"], locs["sy"])
    # a fixed sigma narrower than the PSF loses photons to the background
    fixed, _ = _fit(frames)
    assert fixed["photons"].median() < 0.95 * locs["photons"].median()


def test_elliptical_fit_gives_z():
    """Astigmatic emitters at three z, rendered with the widths of the
    shared test calibration: z is recovered from the fitted sigmas, which
    the z fit is given as the widths of the sampled Gaussian the
    calibration records."""
    from tests.conftest import CALIB_3D

    emitters = _grid(20, spacing=1500.0, photons=5000.0, seed=1)
    column = np.round(emitters["x"] / 1500.0).astype(int) - 1
    z = np.array([-300.0, 0.0, 300.0])[column]
    blur = PIXELSIZE**2 / 12
    emitters = emitters.assign(
        sx=np.sqrt(np.polyval(CALIB_3D["X Coefficients"], z) ** 2 - blur),
        sy=np.sqrt(np.polyval(CALIB_3D["Y Coefficients"], z) ** 2 - blur),
        z=z * CALIB_3D["Magnification factor"],
    )
    frames = _render(emitters, 20, seed=2)
    tracker = RecordingProgress()
    locs, info = _fit(
        frames,
        parameters=_model("elliptical"),
        calibration_3d=dict(CALIB_3D),
        progress=tracker,
    )
    assert_phases_complete(tracker, ["Fitting frames", "Fitting z"])
    assert {"z", "d_zcalib", "lpz"} <= set(locs.columns)
    assert [entry.get("Step") for entry in info[-2:]] == [
        "Fit multi-emitter",
        "Fit 3D",
    ]
    li, ti = _match(locs, emitters, tolerance=100.0)
    assert len(li) == len(emitters)
    error = locs["z"].to_numpy()[li] - emitters["z"].to_numpy()[ti]
    for level in np.unique(emitters["z"]):
        at = emitters["z"].to_numpy()[ti] == level
        assert np.median(error[at]) == pytest.approx(0.0, abs=10.0)
    # the localizations keep the PSF sigmas
    assert locs["sx"].to_numpy()[li] == pytest.approx(
        emitters["sx"].to_numpy()[ti], rel=0.05
    )


@pytest.mark.parametrize("model", ["fixed", "spherical"])
def test_z_needs_the_elliptical_model(singles, model):
    from tests.conftest import CALIB_3D

    frames, _ = singles
    with pytest.raises(ValueError, match="elliptical"):
        _fit(frames, parameters=_model(model), calibration_3d=CALIB_3D)


def test_no_emitters():
    frames = _render(_emitters(np.zeros((0, 4))), 3)
    locs, info = _fit(frames)
    assert len(locs) == 0
    assert {"frame", "x", "y", "lpx", "lpy"} <= set(locs.columns)


# ----------------------------------------------------------------------
# Conventions
# ----------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_columns_and_dtypes(singles, model):
    frames, _ = singles
    locs, info = _fit(frames, parameters=_model(model))
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
    if model == "fixed":
        np.testing.assert_allclose(locs[["sx", "sy"]], SIGMA, rtol=1e-6)
    else:
        assert locs["sx"].median() == pytest.approx(SIGMA, rel=0.02)
    assert np.isfinite(locs.to_numpy(dtype=float)).all()
    # every new column is documented and offered in the save dialog
    assert {"significance", "n_emitters"} <= set(
        localize.LOCALIZATION_COLUMNS["Multi-emitter only"]
    )


@pytest.mark.parametrize("model", MODELS)
def test_parallel_equals_serial(singles, model):
    frames, _ = singles
    serial, _ = _fit(frames, parameters=_model(model), parallel=False)
    parallel, _ = _fit(frames, parameters=_model(model), parallel=True)
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


def test_estimates_the_psf_sigma(singles):
    frames, truth = singles
    tracker = RecordingProgress()
    reported = []

    def report(sigma):
        # the phases started so far tell when it came
        reported.append((sigma, [p["label"] for p in tracker.phases]))

    locs, info = _fit(
        frames,
        parameters=localize.MultiEmitterParameters(),
        progress=tracker,
        psf_sigma_callback=report,
    )
    assert info[-1]["PSF Sigma Estimated"] is True
    assert info[-1]["PSF Sigma (nm)"] == pytest.approx(SIGMA, rel=0.03)
    assert len(locs) == len(truth)
    # reported once, as soon as estimated, before the frames are fitted
    assert reported == [
        (info[-1]["PSF Sigma (nm)"], ["Estimating the PSF sigma"])
    ]


def test_reports_a_given_psf_sigma(singles):
    frames, _ = singles
    reported = []
    _fit(frames, psf_sigma_callback=reported.append)
    assert reported == [pytest.approx(SIGMA)]


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
        assert parameters.psf_model == "fixed"
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
            {"psf_model": "3d"},
            {"psf_model": None},
        ],
    )
    def test_rejects_invalid(self, kwargs):
        with pytest.raises(ValueError):
            localize.MultiEmitterParameters(**kwargs)

    def test_info_round_trip(self):
        parameters = localize.MultiEmitterParameters(
            psf_model="elliptical",
            snr_threshold=8,
            psf_sigma=140,
            max_cycles=10,
        )
        info = parameters.to_info()
        assert info["PSF Model"] == "elliptical"
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


@pytest.mark.parametrize("model", list(residualfit.PSF_MODELS))
def test_padding_covers_the_region(model):
    for sigma in (0.8, 1.0, 1.3, 2.0):
        assert residualfit.padding(sigma, model) >= (
            residualfit.roi_size(sigma, model) / 2
        )
    # storm-analysis's larger region where the sigma is fitted
    if model != "fixed":
        assert residualfit.roi_size(1.0, model) == int(1.5 * 8)


# ----------------------------------------------------------------------
# Derivatives against finite differences
# ----------------------------------------------------------------------


def test_profile_derivative():
    size = 8
    e, de, ds = np.empty(size), np.empty(size), np.empty(size)
    plus, minus, scratch = np.empty(size), np.empty(size), np.empty(size)
    h = 1e-6
    residualfit._profile(3.3, 0, size, 1.2, e, de, ds)
    residualfit._profile(3.3 + h, 0, size, 1.2, plus, scratch, scratch)
    residualfit._profile(3.3 - h, 0, size, 1.2, minus, scratch, scratch)
    np.testing.assert_allclose(de, (plus - minus) / (2 * h), atol=1e-7)
    residualfit._profile(3.3, 0, size, 1.2 + h, plus, scratch, scratch)
    residualfit._profile(3.3, 0, size, 1.2 - h, minus, scratch, scratch)
    np.testing.assert_allclose(ds, (plus - minus) / (2 * h), atol=1e-7)
    # the profile integrates the unit Gaussian over the pixels, here
    # [-0.5, 7.5]
    from scipy.special import erf

    expected = 0.5 * (
        erf((7.5 - 3.3) / (np.sqrt(2) * 1.2))
        - erf((-0.5 - 3.3) / (np.sqrt(2) * 1.2))
    )
    assert e.sum() == pytest.approx(expected, rel=1e-9)


def _one_peak(photons, x, y, bg, data, sx=1.0, sy=1.0):
    """The fit image of one peak in an otherwise empty frame and its
    deviance, as ``residualfit`` holds them."""
    roi = residualfit.roi_size(1.0)
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
    profiles = [np.empty(roi) for _ in range(6)]
    ex, dex, dsx, ey, dey, dsy = profiles
    psf = np.empty(len(roi_dx))
    residualfit._peak_shape(
        x, y, xi, yi, sx, sy, roi_dx, roi_dy, *profiles, psf
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
        (xi, yi, roi_dx, roi_dy, *profiles, f_data, bg_data, bg_counts),
    )


@pytest.mark.parametrize("n_widths", [0, 1, 2])
def test_deviance_gradient(n_widths):
    """The gradient each peak's Levenberg-Marquardt step follows is the
    gradient of the Poisson deviance of its region, the fitted sigmas
    included."""
    rng = np.random.default_rng(4)
    data = rng.poisson(20, (24, 24)).astype(np.float64) + 1.0
    theta = np.array([1500.0, 11.3, 12.6, 21.0, 1.1, 0.9])[: 4 + n_widths]

    def deviance(theta, peak=False):
        sx = theta[4] if n_widths else 1.0
        sy = theta[5] if n_widths == 2 else sx
        result = _one_peak(*theta[:4], data, sx=sx, sy=sy)
        return result if peak else result[0]

    _, (xi, yi, roi_dx, roi_dy, *rest) = deviance(theta, peak=True)
    profiles, (f, bg, counts) = rest[:6], rest[6:]
    jac = np.empty(4 + n_widths)
    hess = np.empty((4 + n_widths, 4 + n_widths))
    residualfit._jacobian_hessian(
        theta[0],
        xi,
        yi,
        roi_dx,
        roi_dy,
        *profiles,
        data,
        f,
        bg,
        counts,
        jac,
        hess,
    )
    for p, h in enumerate((1e-3, 1e-6, 1e-6, 1e-5, 1e-6, 1e-6)[: len(theta)]):
        plus, minus = theta.copy(), theta.copy()
        plus[p] += h
        minus[p] -= h
        numeric = (deviance(plus) - deviance(minus)) / (2 * h)
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
