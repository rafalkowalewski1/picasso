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

    def close(self) -> None:
        """Release backend resources (no-op by default)."""


_cpu_singleton: SplatBackend | None = None


def _cpu_backend() -> SplatBackend:
    """The process-wide CPU reference backend (also the fallback)."""
    global _cpu_singleton
    if _cpu_singleton is None:
        from .splat import CpuBackend

        _cpu_singleton = CpuBackend()
    return _cpu_singleton


def _get_backend() -> SplatBackend:
    """The splat backend for the next render.

    Always the CPU reference implementation for now; GPU selection via
    ``settings["Render"]["gpu"]`` plugs in here (plan item P1.6).
    """
    return _cpu_backend()
