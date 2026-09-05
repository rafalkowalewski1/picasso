"""
picasso.render.backend
~~~~~~~~~~~~~~~~~~~~~~

Contract between scene composition and the raw splat stage: a splat
backend turns per-channel localization columns into raw grayscale
images. ``splat.CpuBackend`` is the reference implementation and the
universal fallback; a GPU backend implements the same contract.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import abc
import logging
import os
import threading
from typing import Literal, TYPE_CHECKING

from .. import lib

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation

    from .splat import _RenderColumns


class SplatBackendError(Exception):
    """A splat backend failed to initialize or render.

    Raising this (rather than crashing) lets ``scene._render_channels``
    retry the request on the CPU backend.
    """


class SplatBackend(abc.ABC):
    """Renders raw per-channel grayscale images from localization
    columns — the seam between scene composition and the splat stage.

    The interface is whole-request (all channels at once) rather than
    per-channel on purpose: a GPU backend keeps channel buffers resident
    and loops channels on-device with a single readback, while the CPU
    backend schedules row chunks across channels through one thread
    pool. A per-channel interface would forbid both optimizations.

    Every implementation must honor three contracts:

    - **Parity**: per channel, ``(n locs in view, float image)`` equal
      to the CPU reference within the golden-scene tolerances; splatting
      is additive, so total intensity is conserved.
    - **Thread safety**: ``render_channels`` may be called concurrently
      (the asynchronous GUI render worker and a synchronous render such
      as the rotation window can overlap); implementations must
      serialize or isolate shared resources internally.
    - **Failure**: raise ``SplatBackendError`` on any initialization or
      render failure instead of crashing; the caller then re-renders
      the request on the CPU backend.
    """

    #: short identifier used in logs and settings
    name: str = "abstract"
    #: True when uploads persist across renders (GPU): callers then hand
    #: over whole channels instead of viewport slices, so the resident
    #: buffers are reused and nothing is transferred per view
    persistent_uploads: bool = False

    @abc.abstractmethod
    def render_channels(
        self,
        columns: list[_RenderColumns],
        info: list[list[dict]],
        *,
        disp_px_size: float,
        viewport: tuple[tuple[float, float], tuple[float, float]] | None,
        blur_method: (
            Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
        ),
        min_blur_width: float,
        ang: tuple | Rotation | None,
    ) -> list[tuple[int, lib.FloatArray2D]]:
        """Render each channel's raw grayscale image.

        Parameters
        ----------
        columns : list of splat._RenderColumns
            Localization columns, one per channel (already extracted,
            angle in radians and lpz fallback applied).
        info : list of list of dict
            Metadata, one entry per channel.
        disp_px_size, viewport, blur_method, min_blur_width, ang
            See ``splat.render``.

        Returns
        -------
        renderings : list of (int, lib.FloatArray2D)
            ``(number of locs in view, image)`` per channel, in input
            order.
        """

    def release_uploads(self) -> None:
        """Drop resident localization uploads (no-op by default); the
        GUI calls this when datasets are closed."""

    def close(self) -> None:
        """Release backend resources (no-op by default)."""


_cpu_singleton: SplatBackend | None = None
_gpu_singleton: SplatBackend | None = None
_gpu_unavailable = False
_singleton_lock = threading.Lock()  # GUI thread and render worker

_log = logging.getLogger(__name__)


def _cpu_backend() -> SplatBackend:
    """The process-wide CPU reference backend (also the fallback)."""
    global _cpu_singleton
    with _singleton_lock:
        if _cpu_singleton is None:
            from .splat import CpuBackend

            _cpu_singleton = CpuBackend()
        return _cpu_singleton


def _gpu_backend() -> SplatBackend | None:
    """The process-wide GPU backend, or None if it cannot start (the
    reason is logged once and never retried in this process)."""
    global _gpu_singleton, _gpu_unavailable
    with _singleton_lock:
        if _gpu_unavailable:
            return None
        if _gpu_singleton is None:
            try:
                from .gpu import WgpuBackend

                _gpu_singleton = WgpuBackend()
            except Exception as error:
                _log.warning("GPU splat backend unavailable: %s", error)
                _gpu_unavailable = True
                return None
        return _gpu_singleton


def release_uploads() -> None:
    """Drop the resident uploads of the GPU backend, if one is running
    (e.g. when the GUI closes a dataset)."""
    if _gpu_singleton is not None:
        _gpu_singleton.release_uploads()


def vram_budget_bytes() -> int | None:
    """GPU memory the backend may keep resident for uploads, from
    ``settings["Render"]["gpu"]["vram_budget_mb"]`` (read per render);
    None means unlimited (``0`` in the settings), invalid values mean
    ``lib.RENDER_VRAM_BUDGET_MB_DEFAULT``."""
    try:
        value = lib.io.load_user_settings()["Render"]["gpu"]["vram_budget_mb"]
    except Exception:
        value = lib.RENDER_VRAM_BUDGET_MB_DEFAULT
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
    ):
        value = lib.RENDER_VRAM_BUDGET_MB_DEFAULT
    return None if value == 0 else int(value * 2**20)


def _get_backend() -> SplatBackend:
    """The splat backend for the next render.

    The CPU reference implementation, unless the experimental
    ``PICASSO_GPU_SPLAT=1`` environment variable opts into the GPU
    backend (spike P1.3); the settings-driven choice via
    ``settings["Render"]["gpu"]`` replaces this in plan item P1.6.
    """
    if os.environ.get("PICASSO_GPU_SPLAT") == "1":
        backend = _gpu_backend()
        if backend is not None:
            return backend
    return _cpu_backend()
