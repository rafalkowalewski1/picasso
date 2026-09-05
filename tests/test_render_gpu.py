"""Tests for the wgpu splat backend.

These self-skip when ``wgpu`` is not installed or no GPU adapter is
available.
"""

import os

import numpy as np
import pandas as pd
import pytest

wgpu = pytest.importorskip("wgpu")

from picasso import render  # noqa: E402
from picasso.render import backend as backend_mod  # noqa: E402
from picasso.render.backend import SplatBackendError  # noqa: E402

INFO = [{"Height": 64, "Width": 64, "Pixelsize": 130}]
KWARGS = dict(
    disp_px_size=13.0,  # oversampling 10
    viewport=((3.0, 2.0), (61.0, 62.5)),
    min_blur_width=0.1,
    ang=None,
)


@pytest.fixture(scope="module")
def gpu():
    try:
        from picasso.render.gpu import WgpuBackend

        backend = WgpuBackend()
    except SplatBackendError as error:
        pytest.skip(f"no usable GPU adapter: {error}")
    yield backend
    backend.close()


@pytest.fixture(scope="module")
def locs():
    rng = np.random.default_rng(42)
    n = 150_000
    return pd.DataFrame(
        {
            "x": rng.uniform(-2, 66, n).astype(np.float32),
            "y": rng.uniform(-2, 66, n).astype(np.float32),
            "lpx": rng.uniform(0.02, 0.5, n).astype(np.float32),
            "lpy": rng.uniform(0.02, 0.5, n).astype(np.float32),
        }
    )


def _columns(locs, blur):
    return [render._extract_render_columns(locs, blur, None)]


def _cpu(locs, blur):
    return backend_mod._cpu_backend().render_channels(
        _columns(locs, blur), [INFO], blur_method=blur, **KWARGS
    )


class TestWgpuParity:
    def test_gaussian_matches_cpu(self, gpu, locs):
        ((n_cpu, img_cpu),) = _cpu(locs, "gaussian")
        ((n_gpu, img_gpu),) = gpu.render_channels(
            _columns(locs, "gaussian"),
            [INFO],
            blur_method="gaussian",
            **KWARGS,
        )
        assert n_gpu == n_cpu
        assert img_gpu.shape == img_cpu.shape
        assert img_gpu.dtype == np.float32
        # f32 arithmetic + Q16 fixed point vs the CPU's f64: bounded
        # by ~1e-3 relative on meaningful pixels (measured ~4e-4 of
        # the image maximum), intensity conserved
        np.testing.assert_allclose(
            img_gpu, img_cpu, rtol=5e-3, atol=2e-3 * img_cpu.max()
        )
        assert abs(img_gpu.sum() - img_cpu.sum()) < 1e-3 * img_cpu.sum()

    def test_hist_matches_cpu(self, gpu, locs):
        ((n_cpu, img_cpu),) = _cpu(locs, None)
        ((n_gpu, img_gpu),) = gpu.render_channels(
            _columns(locs, None), [INFO], blur_method=None, **KWARGS
        )
        assert n_gpu == n_cpu
        assert img_gpu.sum() == img_cpu.sum()  # counts conserved exactly
        # locs within float rounding of a pixel boundary may land in
        # the neighbor pixel (f32 vs f64 coordinate transform)
        diff = np.abs(img_gpu - img_cpu)
        assert diff.max() <= 1
        assert (diff > 0).mean() < 1e-4

    def test_multi_channel_order(self, gpu, locs):
        channels = [locs.iloc[i::3] for i in range(3)]
        columns = [
            render._extract_render_columns(c, "gaussian", None)
            for c in channels
        ]
        gpu_out = gpu.render_channels(
            columns, [INFO] * 3, blur_method="gaussian", **KWARGS
        )
        for channel, (n, image) in zip(channels, gpu_out):
            ((n_cpu, img_cpu),) = _cpu(channel, "gaussian")
            assert n == n_cpu
            np.testing.assert_allclose(
                image, img_cpu, rtol=5e-3, atol=2e-3 * img_cpu.max()
            )


class TestWgpuScope:
    @pytest.mark.parametrize("blur", ["convolve", "smooth", "gaussian_iso"])
    def test_unsupported_blur_raises(self, gpu, locs, blur):
        with pytest.raises(SplatBackendError):
            gpu.render_channels(
                _columns(locs, blur), [INFO], blur_method=blur, **KWARGS
            )

    def test_rotation_raises(self, gpu, locs):
        kwargs = dict(KWARGS, ang=(0.5, 0.0, 0.0))
        with pytest.raises(SplatBackendError):
            gpu.render_channels(
                _columns(locs, "gaussian"),
                [INFO],
                blur_method="gaussian",
                **kwargs,
            )

    def test_extreme_density_overflow_is_detected(self, gpu):
        # a million locs exactly on one pixel center (viewport x_min=2,
        # oversampling 10: x=32.05 -> display 300.5) with tiny
        # precision: the pixel sum exceeds the fixed-point range at
        # every retry scale, so the backend must refuse (the caller
        # then renders on CPU)
        n = 1_000_000
        dense = pd.DataFrame(
            {
                "x": np.full(n, 32.05, np.float32),
                "y": np.full(n, 32.05, np.float32),
                "lpx": np.full(n, 2e-3, np.float32),
                "lpy": np.full(n, 2e-3, np.float32),
            }
        )
        kwargs = dict(KWARGS, min_blur_width=0.0)
        with pytest.raises(SplatBackendError):
            gpu.render_channels(
                _columns(dense, "gaussian"),
                [INFO],
                blur_method="gaussian",
                **kwargs,
            )
        # the sticky scale must not poison later ordinary renders
        gpu._scale = 65536.0

    def test_scale_adapts_and_recovers_density(self, gpu):
        # dense but representable (pixel sum ~3e6 needs a scale coarser
        # than Q16 but well above the minimum): the backend retries
        # coarser and still conserves intensity
        n = 200_000
        dense = pd.DataFrame(
            {
                "x": np.full(n, 32.05, np.float32),
                "y": np.full(n, 32.05, np.float32),
                "lpx": np.full(n, 0.01, np.float32),
                "lpy": np.full(n, 0.01, np.float32),
            }
        )
        kwargs = dict(KWARGS, min_blur_width=0.0)
        ((n_cpu, img_cpu),) = backend_mod._cpu_backend().render_channels(
            _columns(dense, "gaussian"),
            [INFO],
            blur_method="gaussian",
            **kwargs,
        )
        ((n_gpu, img_gpu),) = gpu.render_channels(
            _columns(dense, "gaussian"),
            [INFO],
            blur_method="gaussian",
            **kwargs,
        )
        assert n_gpu == n_cpu
        assert gpu._scale < 65536.0  # adapted to the density
        assert abs(img_gpu.sum() - img_cpu.sum()) < 1e-2 * img_cpu.sum()
        gpu._scale = 65536.0


class TestSeamOptIn:
    def test_env_var_selects_gpu_and_falls_back(self, gpu, locs, monkeypatch):
        from picasso.render.gpu import WgpuBackend

        monkeypatch.setenv("PICASSO_GPU_SPLAT", "1")
        monkeypatch.setattr(backend_mod, "_gpu_singleton", gpu)
        monkeypatch.setattr(backend_mod, "_gpu_unavailable", False)
        assert isinstance(backend_mod._get_backend(), WgpuBackend)
        # supported: rendered on the GPU
        ((n, _),) = render._render_channels(
            [locs], [INFO], blur_method="gaussian", **KWARGS
        )
        # unsupported: silently rendered on the CPU instead
        ((n_conv, img_conv),) = render._render_channels(
            [locs], [INFO], blur_method="convolve", **KWARGS
        )
        ((n_ref, img_ref),) = _cpu(locs, "convolve")
        assert n == n_ref and n_conv == n_ref
        np.testing.assert_array_equal(img_conv, img_ref)

    def test_without_env_var_cpu_is_selected(self, monkeypatch):
        monkeypatch.delenv("PICASSO_GPU_SPLAT", raising=False)
        from picasso.render.splat import CpuBackend

        assert isinstance(backend_mod._get_backend(), CpuBackend)
