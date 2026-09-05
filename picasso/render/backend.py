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
      to the CPU reference within the golden-scene tolerances
      (``tests/test_render_goldens.py``: raw intensities within 5e-3
      relative or 2e-3 of the image maximum, histogram counts exact up
      to pixel-boundary rounding, total intensity within 1e-3);
      splatting is additive, so total intensity is conserved. A
      backend need not be bit-reproducible from run to run (the GPU
      sums in hardware-dependent order), but its run-to-run spread
      must stay within the same tolerances.
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

    def describe(self) -> str:
        """Human-readable device description (for the GUI's info)."""
        return self.name

    def close(self) -> None:
        """Release backend resources (no-op by default)."""


_cpu_singleton: SplatBackend | None = None
_gpu_singleton: SplatBackend | None = None
_gpu_unavailable = False
_gpu_adapter: str | None = None  # preference the GPU singleton was built for
_singleton_lock = threading.Lock()  # GUI thread and render worker
_settings_cache = (None, None, None)  # (file mtime, loader, Render section)
_settings_lock = threading.Lock()

_log = logging.getLogger(__name__)


def _render_settings() -> dict:
    """The ``Render`` section of the user settings, re-read only when
    the settings file changed (a YAML parse per render is too slow to
    sit on every request). Keyed on the loader too, so a patched
    ``io.load_user_settings`` takes effect immediately."""
    global _settings_cache
    loader = lib.io.load_user_settings
    try:
        mtime = os.path.getmtime(lib.io._user_settings_filename())
    except OSError:
        mtime = None
    with _settings_lock:
        cached = _settings_cache
        if (
            cached[0] == mtime
            and cached[1] is loader
            and cached[2] is not None
        ):
            return cached[2]
        try:
            section = loader().get("Render", None)
        except Exception:
            section = None
        if not isinstance(section, dict):
            section = {}
        _settings_cache = (mtime, loader, section)
        return section


def gpu_settings() -> dict:
    """``settings["Render"]["gpu"]`` validated: ``enabled`` (``"auto"``,
    ``"on"`` or ``"off"``; YAML's bare ``on``/``off`` parse as booleans
    and are accepted), ``adapter`` (a non-empty string) and
    ``vram_budget_bytes`` (None = unlimited). Invalid values fall back
    to the ``lib.RENDER_GPU_*`` defaults."""
    raw = _render_settings().get("gpu", None)
    if not isinstance(raw, dict):
        raw = {}
    enabled = raw.get("enabled", lib.RENDER_GPU_ENABLED_DEFAULT)
    if isinstance(enabled, bool):
        enabled = "on" if enabled else "off"
    enabled = str(enabled).strip().lower()
    if enabled not in ("auto", "on", "off"):
        enabled = lib.RENDER_GPU_ENABLED_DEFAULT
    adapter = raw.get("adapter", lib.RENDER_GPU_ADAPTER_DEFAULT)
    if not isinstance(adapter, str) or not adapter.strip():
        adapter = lib.RENDER_GPU_ADAPTER_DEFAULT
    budget = raw.get("vram_budget_mb", lib.RENDER_VRAM_BUDGET_MB_DEFAULT)
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or budget < 0
    ):
        budget = lib.RENDER_VRAM_BUDGET_MB_DEFAULT
    return {
        "enabled": enabled,
        "adapter": adapter.strip(),
        "vram_budget_bytes": None if budget == 0 else int(budget * 2**20),
    }


def vram_budget_bytes() -> int | None:
    """GPU memory the backend may keep resident for uploads (see
    ``gpu_settings``); None means unlimited."""
    return gpu_settings()["vram_budget_bytes"]


def _cpu_backend() -> SplatBackend:
    """The process-wide CPU reference backend (also the fallback)."""
    global _cpu_singleton
    with _singleton_lock:
        if _cpu_singleton is None:
            from .splat import CpuBackend

            _cpu_singleton = CpuBackend()
        return _cpu_singleton


def _gpu_backend(adapter: str, warn: bool) -> SplatBackend | None:
    """The process-wide GPU backend for ``adapter``, or None if it
    cannot start: logged once per adapter preference — as a warning
    when the user asked for the GPU explicitly (``warn``), else as
    information — and not retried until the preference changes."""
    global _gpu_singleton, _gpu_unavailable, _gpu_adapter
    with _singleton_lock:
        if _gpu_adapter == adapter:
            if _gpu_singleton is not None:
                return _gpu_singleton
            if _gpu_unavailable:
                return None
        if _gpu_singleton is not None:  # the preference changed
            _gpu_singleton.close()
            _gpu_singleton = None
        _gpu_adapter = adapter
        try:
            from .gpu import WgpuBackend

            _gpu_singleton = WgpuBackend(adapter=adapter)
            _gpu_unavailable = False
        except Exception as error:
            (_log.warning if warn else _log.info)(
                "GPU rendering unavailable (%s); rendering on the CPU", error
            )
            _gpu_unavailable = True
            return None
        _log.info("Rendering on the GPU: %s", _gpu_singleton.describe())
        return _gpu_singleton


def _get_backend(n_locs: int | None = None) -> SplatBackend:
    """The splat backend for the next render, per
    ``settings["Render"]["gpu"]`` (``enabled``: auto/on/off, ``adapter``):
    the GPU when enabled and available, else the CPU reference. Requests
    of fewer than ``lib.RENDER_GPU_MIN_LOCS`` localizations (``n_locs``)
    stay on the CPU, which is faster for them than the GPU's fixed cost
    per render."""
    settings = gpu_settings()
    if settings["enabled"] == "off":
        return _cpu_backend()
    if n_locs is not None and n_locs < lib.RENDER_GPU_MIN_LOCS:
        return _cpu_backend()
    backend = _gpu_backend(
        settings["adapter"], warn=settings["enabled"] == "on"
    )
    return backend if backend is not None else _cpu_backend()


def describe_active() -> str:
    """Where large renders currently run, for the GUI's info dialog:
    ``"GPU (Apple M4 via Metal)"`` or ``"CPU (5 workers)"``."""
    backend = _get_backend()
    if backend.persistent_uploads:
        return f"GPU ({backend.describe()})"
    workers = lib.n_workers(
        lib.RENDER_CPU_UTILIZATION_DEFAULT, settings_section="Render"
    )
    return f"CPU ({workers} worker{'s' if workers != 1 else ''})"


def release_uploads() -> None:
    """Drop the resident uploads of the GPU backend, if one is running
    (e.g. when the GUI closes a dataset)."""
    if _gpu_singleton is not None:
        _gpu_singleton.release_uploads()
