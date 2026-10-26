"""Test ``picasso._ground_truth`` - matching localizations to a simulated
ground truth, the metrics computed from the pairs, and the simulated
movies of the multi-emitter fitting benchmark.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso import _ground_truth as gt
from picasso import localize


def _table(rows, columns=("frame", "x", "y")) -> pd.DataFrame:
    return pd.DataFrame(np.array(rows, dtype=float), columns=list(columns))


class TestMatch:
    def test_pairs_within_the_tolerance_and_frame_only(self):
        truth = _table([(0, 0, 0), (0, 500, 0), (1, 0, 0)])
        locs = _table([(0, 10, 0), (1, 0, 30), (2, 0, 0), (0, 700, 0)])
        matches = gt.match(locs, truth, tolerance=50)
        assert matches.loc_index.tolist() == [0, 1]
        assert matches.truth_index.tolist() == [0, 2]
        np.testing.assert_allclose(matches.distance, [10, 30])

    def test_one_to_one_with_smallest_summed_distance(self):
        # Greedy nearest-first would pair loc 0 with truth 1 (distance 5)
        # and leave loc 1 unpaired; the optimal assignment pairs both.
        truth = _table([(0, 0, 0), (0, 40, 0)])
        locs = _table([(0, 35, 0), (0, 75, 0)])
        matches = gt.match(locs, truth, tolerance=40)
        assert sorted(zip(matches.loc_index, matches.truth_index)) == [
            (0, 0),
            (1, 1),
        ]

    def test_each_emitter_counts_once(self):
        truth = _table([(0, 0, 0)])
        locs = _table([(0, 5, 0), (0, -3, 0)])
        matches = gt.match(locs, truth, tolerance=50)
        assert matches.loc_index.tolist() == [1]
        assert matches.truth_index.tolist() == [0]

    def test_z_tolerance(self):
        truth = _table([(0, 0, 0, 0)], ("frame", "x", "y", "z"))
        locs = _table([(0, 5, 0, 300)], ("frame", "x", "y", "z"))
        assert len(gt.match(locs, truth, tolerance=50).loc_index) == 1
        assert not len(
            gt.match(locs, truth, tolerance=50, z_tolerance=200).loc_index
        )

    @pytest.mark.parametrize("n_locs, n_truth", [(0, 3), (3, 0), (0, 0)])
    def test_empty(self, n_locs, n_truth):
        locs = _table(np.zeros((n_locs, 3)))
        truth = _table(np.zeros((n_truth, 3)))
        matches = gt.match(locs, truth, tolerance=10)
        assert len(matches.loc_index) == len(matches.distance) == 0

    def test_dense_random_matches_brute_force(self):
        """The cluster-wise assignment equals one assignment over each
        whole frame."""
        from scipy.optimize import linear_sum_assignment

        rng = np.random.default_rng(3)
        rows_t, rows_l = [], []
        for frame in range(5):
            xy = rng.uniform(0, 2000, (40, 2))
            rows_t += [(frame, *p) for p in xy]
            rows_l += [(frame, *p) for p in xy + rng.normal(0, 40, xy.shape)]
        truth, locs = _table(rows_t), _table(rows_l)
        matches = gt.match(locs, truth, tolerance=60)
        total = 0.0
        n = 0
        for frame in range(5):
            lf = locs[locs.frame == frame][["x", "y"]].to_numpy()
            tf = truth[truth.frame == frame][["x", "y"]].to_numpy()
            d = np.hypot(*(lf[:, None] - tf[None]).transpose(2, 0, 1))
            big = 60 * 1000
            cost = np.where(d < 60, d, big)
            r, c = linear_sum_assignment(cost)
            ok = cost[r, c] < big
            total += cost[r, c][ok].sum()
            n += ok.sum()
        assert len(matches.loc_index) == n
        assert matches.distance.sum() == pytest.approx(total)


class TestMetrics:
    def test_counts_and_errors(self):
        truth = _table(
            [(0, 0, 0, 100), (0, 500, 0, 100), (1, 0, 0, 100)],
            ("frame", "x", "y", "photons"),
        )
        locs = _table(
            [(0, 3, -4, 80), (1, 0, 0, 120), (0, 900, 0, 50)],
            ("frame", "x", "y", "photons"),
        )
        matches = gt.match(locs, truth, tolerance=50)
        result = gt.metrics(locs, truth, matches)
        assert (result["tp"], result["fp"], result["fn"]) == (2, 1, 1)
        assert result["recall"] == pytest.approx(2 / 3)
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["jaccard"] == pytest.approx(2 / 4)
        assert result["rmse_lateral"] == pytest.approx(np.sqrt(25 / 2))
        assert result["bias_x"] == pytest.approx(1.5)
        assert result["bias_y"] == pytest.approx(-2.0)
        assert result["photon_ratio"] == pytest.approx(1.0)
        assert result["efficiency"] == pytest.approx(
            100 - np.hypot(50, np.sqrt(12.5))
        )

    def test_gt_distances(self):
        truth = _table([(0, 0, 0)])
        locs = _table([(0, 3, 4), (0, 900, 0)])
        matches = gt.match(locs, truth, tolerance=50)
        np.testing.assert_allclose(
            gt.gt_distances(len(locs), matches), [5, -1]
        )


class TestRenderMovie:
    def test_photons_land_where_the_emitter_is(self):
        rng = np.random.default_rng(0)
        emitters = pd.DataFrame(
            {"frame": [0, 1], "x": [500.0, 1230.0], "y": [800.0, 400.0]}
        )
        emitters["photons"] = 20000.0
        movie = gt.render_movie(
            emitters,
            n_frames=2,
            width=20,
            height=16,
            pixelsize=100.0,
            sigma=100.0,
            background=0.0,
            rng=rng,
        )
        assert movie.shape == (2, 16, 20) and movie.dtype == np.uint16
        for frame, (x, y) in enumerate([(5.0, 8.0), (12.3, 4.0)]):
            image = movie[frame].astype(float)
            rows, cols = np.indices(image.shape)
            # pixel centers on integers: the centroid is the position
            assert (image * cols).sum() / image.sum() == pytest.approx(
                x, abs=0.03
            )
            assert (image * rows).sum() / image.sum() == pytest.approx(
                y, abs=0.03
            )
            assert image.sum() == pytest.approx(20000, rel=0.01)

    def test_background_mean(self):
        movie = gt.render_movie(
            pd.DataFrame(columns=["frame", "x", "y", "photons"]),
            n_frames=3,
            width=50,
            height=50,
            pixelsize=100.0,
            sigma=100.0,
            background=12.0,
            rng=np.random.default_rng(1),
            baseline=100.0,
        )
        assert movie.mean() == pytest.approx(112.0, abs=0.3)

    def test_fitted_positions_match(self):
        """A movie of single emitters is localized at their ground
        truth: the generators and the matching agree with Picasso's
        frame."""
        rng = np.random.default_rng(2)
        pixelsize = 130.0
        emitters = gt.pair_emitters(
            rng,
            n_frames=4,
            separation=0,
            width=48,
            height=48,
            pixelsize=pixelsize,
            spacing=1500.0,
            photons=5000.0,
        )
        movie = gt.render_movie(
            emitters,
            n_frames=4,
            width=48,
            height=48,
            pixelsize=pixelsize,
            sigma=150.0,
            background=10.0,
            rng=rng,
        )
        info = [
            {
                "Frames": 4,
                "Width": 48,
                "Height": 48,
                "Pixelsize": pixelsize,
                "Baseline": 0,
                "Sensitivity": 1,
                "Gain": 1,
            }
        ]
        locs = localize.localize_frames(
            movie,
            info,
            identification_parameters=localize.IdentificationParameters(
                method=localize.NetGradient(3000), box=7
            ),
            fitting_method="gaussmle",
            parallel=False,
        )
        matches = gt.match(locs, emitters, tolerance=100)
        result = gt.metrics(locs, emitters, matches)
        assert result["recall"] == 1.0 and result["precision"] == 1.0
        assert abs(result["bias_x"]) < 2 and abs(result["bias_y"]) < 2
        assert result["rmse_lateral"] < 10


class TestEmitterTables:
    def test_pairs_are_separated(self):
        emitters = gt.pair_emitters(
            np.random.default_rng(0),
            n_frames=3,
            separation=80.0,
            width=40,
            height=40,
            pixelsize=100.0,
            spacing=1000.0,
            photons=(1000.0, 100.0),
            photon_ratio=0.5,
        )
        assert len(emitters) % 2 == 0
        for _, pair in emitters.groupby("pair"):
            assert len(pair) == 2 and pair.frame.nunique() == 1
            assert np.hypot(
                *np.diff(pair[["x", "y"]].to_numpy(), axis=0)[0]
            ) == (pytest.approx(80.0))
            first, second = pair["photons"].to_numpy()
            assert second == pytest.approx(0.5 * first)

    def test_random_density(self):
        emitters = gt.random_emitters(
            np.random.default_rng(0),
            n_frames=200,
            density=2.0,
            width=50,
            height=50,
            pixelsize=100.0,
            photons=1000.0,
        )
        # 25 um^2 at 2 per um^2
        assert len(emitters) / 200 == pytest.approx(50, rel=0.05)
        assert emitters.x.min() >= -50 and emitters.x.max() < 4950

    def test_paint_duty_cycle_and_partial_frames(self):
        emitters = gt.paint_emitters(
            n_frames=2000,
            site_density=10.0,
            width=20,
            height=20,
            pixelsize=100.0,
            mean_bright=300.0,
            mean_dark=2700.0,
            photon_rate=10.0,
            seed=1,
        )
        n_sites = emitters.site.nunique()
        # about 10 % of the sites emit in a frame
        per_frame = len(emitters) / 2000 / n_sites
        assert 0.08 < per_frame < 0.14
        # binding starts and ends within frames: fewer photons than a
        # full frame's 1000 (on average) are common
        assert (emitters.photons < 500).mean() > 0.1


def test_challenge_ground_truth_reader(tmp_path):
    path = tmp_path / "activations.csv"
    path.write_text(
        "Ground-truth,frame,xnano,ynano,znano,intensity \n"
        "    1,  12, 43.325, 839.830, 340.704, 2747.034 \n"
        "    2,   1, 39.005, 850.199, 331.856, 1357.536 \n"
    )
    truth = gt.load_smlm_challenge_ground_truth(str(path))
    assert truth.frame.tolist() == [0, 11]
    np.testing.assert_allclose(truth.x, [39.005 - 50, 43.325 - 50])
    np.testing.assert_allclose(truth.y, [850.199 - 50, 839.830 - 50])
    np.testing.assert_allclose(truth.z, [331.856, 340.704])
    np.testing.assert_allclose(truth.photons, [1357.536, 2747.034])
