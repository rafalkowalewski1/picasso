"""
picasso.render.gpu.backend_wgpu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Offscreen GPU splat backend on wgpu (WebGPU: Metal/D3D12/Vulkan).

Splatting runs in compute shaders — one thread per localization,
accumulating into a storage-buffer image with atomics — rather than
rasterized quads: on tile-based GPUs (Apple in particular) the
per-primitive cost of tens of millions of tiny quads dwarfs the actual
arithmetic, and measured compute is more than an order of magnitude
faster. Histogram counts use integer ``atomicAdd`` (bit-exact and
deterministic). Gaussian weights accumulate as Q16 fixed point through
the same integer ``atomicAdd`` — measured several times faster than a
float compare-exchange loop on clustered real data, and deterministic
because integer summation is order-independent; the cost is a bounded
quantization (contributions below ~1/131072 of a count are dropped and
each add rounds to 1/65536), far below display quantization. A
wrap-around of the 32-bit accumulator (astronomically dense views) is
detected in-shader and converts the request into a CPU fallback, so the
output is never silently wrong. The in-view count is tallied on the GPU
alongside the splat.

The arithmetic mirrors ``kernels._draw_gaussian_loc`` exactly (strict
in-view mask, truncation-based ±3σ footprint bounds, the same
normalized Gaussian at pixel centers), so both backends touch the
identical pixel set and differ only by float32-vs-float64 rounding.

Spike scope (P1.3): 2D ``gaussian`` and histogram (``blur_method`` in
``(None, "gaussian")``), no 3D rotation, no per-localization precision
ellipse angle. Anything else raises ``SplatBackendError`` and the
caller falls back to the CPU backend.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import threading
from typing import Literal, TYPE_CHECKING

import numpy as np
import wgpu

from ... import lib
from ..backend import SplatBackend, SplatBackendError

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation

    from ..splat import _RenderColumns

_WORKGROUP = 256
_MAX_SCALE = 65536.0  # Q16 fixed point at full precision
_MAX_WORKGROUPS = 65_535  # per dispatch dimension; grid-stride covers more

#: separable-Gaussian scratch size (display px); footprints wider than
#: this (deep zoom) fall back to evaluating exp per pixel
_MAX_GX = 64

_WGSL = (
    f"""
const WORKGROUP: u32 = {_WORKGROUP}u;
const MAX_GX: i32 = {_MAX_GX};
"""
    + """
struct Uniforms {
    os: f32,
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,
    min_blur: f32,
    npx_f: f32,
    npy_f: f32,
    n_locs: u32,
    npx: u32,
    npy: u32,
    scale: f32,  // fixed-point scale for Gaussian accumulation
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> image: array<atomic<u32>>;
// aux[0] = in-view count, aux[1] = fixed-point overflow flag
@group(0) @binding(2) var<storage, read_write> aux: array<atomic<u32>, 2>;
@group(1) @binding(0) var<storage, read> xs: array<f32>;
@group(1) @binding(1) var<storage, read> ys: array<f32>;
@group(1) @binding(2) var<storage, read> lpxs: array<f32>;
@group(1) @binding(3) var<storage, read> lpys: array<f32>;

// Gaussian weights accumulate as fixed-point u32 (plain atomicAdd is
// several times faster than a float compare-exchange loop, and integer
// summation is order-independent, hence deterministic). atomicAdd
// returns the previous value, so a wrap-around is detected and flagged;
// the Python side then retries at a coarser scale (or falls back to
// the CPU backend), so the output is never silently wrong.

var<workgroup> wg_count: atomic<u32>;

fn splat_fixed(idx: u32, w: f32) {
    let q = u32(round(w * u.scale));
    if (q == 0u) {
        return;
    }
    let old = atomicAdd(&image[idx], q);
    if (old > 4294967295u - q) {
        atomicStore(&aux[1], 1u);
    }
}

fn flush_count(local_index: u32, my_count: u32) {
    atomicAdd(&wg_count, my_count);
    workgroupBarrier();
    if (local_index == 0u) {
        atomicAdd(&aux[0], atomicLoad(&wg_count));
    }
}

@compute @workgroup_size(WORKGROUP)
fn cs_hist(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WORKGROUP;
    var my_count = 0u;
    for (var i = gid.x; i < u.n_locs; i += stride) {
        let x = xs[i];
        let y = ys[i];
        // strict in-view mask, identical to _render_setup's
        if (x <= u.x_min || x >= u.x_max || y <= u.y_min || y >= u.y_max) {
            continue;
        }
        my_count += 1u;
        let j = min(u32(trunc(u.os * (x - u.x_min))), u.npx - 1u);
        let ii = min(u32(trunc(u.os * (y - u.y_min))), u.npy - 1u);
        atomicAdd(&image[ii * u.npx + j], 1u);
    }
    flush_count(lid, my_count);
}

@compute @workgroup_size(WORKGROUP)
fn cs_gaussian(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WORKGROUP;
    var my_count = 0u;
    for (var i = gid.x; i < u.n_locs; i += stride) {
        let x = xs[i];
        let y = ys[i];
        if (x <= u.x_min || x >= u.x_max || y <= u.y_min || y >= u.y_max) {
            continue;
        }
        my_count += 1u;
        let xp = u.os * (x - u.x_min);
        let yp = u.os * (y - u.y_min);
        let sx = u.os * max(lpxs[i], u.min_blur);
        let sy = u.os * max(lpys[i], u.min_blur);
        // CPU footprint arithmetic: trunc toward zero, then clamp
        let j_min = max(0, i32(trunc(xp - 3.0 * sx)));
        let j_max = min(i32(u.npx), i32(trunc(xp + 3.0 * sx)) + 1);
        let i_min = max(0, i32(trunc(yp - 3.0 * sy)));
        let i_max = min(i32(u.npy), i32(trunc(yp + 3.0 * sy)) + 1);
        let nx = j_max - j_min;
        if (nx <= 0 || i_max <= i_min) {
            continue;
        }
        let inv_2sx2 = 1.0 / (2.0 * sx * sx);
        let inv_2sy2 = 1.0 / (2.0 * sy * sy);
        let norm = 1.0 / (6.283185307179586 * sx * sy);
        if (nx <= MAX_GX) {
            // separable kernel, as in kernels._draw_gaussian_loc:
            // one exp per row/column instead of one per pixel
            var gx: array<f32, MAX_GX>;
            for (var jj = 0; jj < nx; jj++) {
                let dx = f32(j_min + jj) + 0.5 - xp;
                gx[jj] = exp(-dx * dx * inv_2sx2);
            }
            for (var ii = i_min; ii < i_max; ii++) {
                let dy = f32(ii) + 0.5 - yp;
                let gy = norm * exp(-dy * dy * inv_2sy2);
                let row = u32(ii) * u.npx;
                for (var jj = 0; jj < nx; jj++) {
                    splat_fixed(row + u32(j_min + jj), gy * gx[jj]);
                }
            }
        } else {
            for (var ii = i_min; ii < i_max; ii++) {
                let dy = f32(ii) + 0.5 - yp;
                let gy = norm * exp(-dy * dy * inv_2sy2);
                let row = u32(ii) * u.npx;
                for (var jj = j_min; jj < j_max; jj++) {
                    let dx = f32(jj) + 0.5 - xp;
                    splat_fixed(
                        row + u32(jj), gy * exp(-dx * dx * inv_2sx2)
                    );
                }
            }
        }
    }
    flush_count(lid, my_count);
}
"""
)


class WgpuBackend(SplatBackend):
    """GPU splat backend computing offscreen through wgpu.

    Raises
    ------
    SplatBackendError
        If no suitable adapter/device is available (at construction) or
        a render request is outside the supported scope / fails.
    """

    name = "wgpu"

    def __init__(self):
        try:
            adapter = wgpu.gpu.request_adapter_sync(
                power_preference="high-performance"
            )
            # raise the storage-binding ceiling to what the adapter has;
            # channels are one f32 buffer per column, 4 bytes per loc
            wanted = (
                "max-storage-buffer-binding-size",
                "max-buffer-size",
            )
            limits = {
                key: adapter.limits[key]
                for key in wanted
                if key in adapter.limits
            }
            self._device = adapter.request_device_sync(required_limits=limits)
        except Exception as error:
            raise SplatBackendError(
                f"wgpu initialization failed: {error}"
            ) from error
        self._adapter_info = dict(adapter.info)
        self._max_channel_locs = (
            limits.get("max-storage-buffer-binding-size", 128 * 2**20) // 4
        )
        device = self._device
        shader = device.create_shader_module(code=_WGSL)
        # explicit shared layout so both pipelines accept the same bind
        # groups (auto layout would drop cs_hist's unused bindings)
        group0 = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
            ]
        )
        self._group1_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": binding,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": wgpu.BufferBindingType.read_only_storage
                    },
                }
                for binding in range(4)
            ]
        )
        layout = device.create_pipeline_layout(
            bind_group_layouts=[group0, self._group1_layout]
        )
        self._pipelines = {
            entry: device.create_compute_pipeline(
                layout=layout,
                compute={"module": shader, "entry_point": entry},
            )
            for entry in ("cs_hist", "cs_gaussian")
        }
        self._uniforms = device.create_buffer(
            size=48,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        # aux[0] = in-view count, aux[1] = fixed-point overflow flag
        self._aux = device.create_buffer(
            size=8,
            usage=(
                wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST
            ),
        )
        self._group0_layout = group0
        # one render at a time per device queue; the seam allows
        # concurrent callers (async worker + synchronous renders)
        self._lock = threading.Lock()
        self._image = None  # storage image buffer, reused across calls
        self._image_size = 0
        self._group0 = None
        # per-channel storage buffers cached by the identity of the
        # column arrays (a reference is kept, so ids stay valid); the
        # proper dataset-keyed invalidation is plan item P1.7
        self._buffer_cache = {}
        self._scale = _MAX_SCALE  # adapted per data, see _render

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
        """Render each channel offscreen on the GPU (see
        ``backend.SplatBackend.render_channels``)."""
        if blur_method not in (None, "gaussian"):
            raise SplatBackendError(
                f"blur_method '{blur_method}' not GPU-rendered yet"
            )
        if ang is not None or viewport is None:
            raise SplatBackendError("3D rotation not GPU-rendered yet")
        if any(c.angle is not None for c in columns):
            raise SplatBackendError(
                "per-localization angles not GPU-rendered yet"
            )
        if blur_method == "gaussian" and any(
            c.lpx is None or c.lpy is None for c in columns
        ):
            raise SplatBackendError("missing localization precision")
        if any(len(c) > self._max_channel_locs for c in columns):
            raise SplatBackendError(
                "channel exceeds the GPU storage-binding limit"
            )
        try:
            with self._lock:
                return self._render(
                    columns,
                    info,
                    disp_px_size=disp_px_size,
                    viewport=viewport,
                    blur_method=blur_method,
                    min_blur_width=min_blur_width,
                )
        except SplatBackendError:
            raise
        except Exception as error:
            raise SplatBackendError(f"GPU render failed: {error}") from error

    def _render(
        self,
        columns,
        info,
        *,
        disp_px_size,
        viewport,
        blur_method,
        min_blur_width,
    ):
        device = self._device
        (y_min, x_min), (y_max, x_max) = viewport
        pixelsize = lib.get_from_metadata(
            info[0], "Pixelsize", raise_error=True
        )
        oversampling = pixelsize / disp_px_size
        # image size exactly as _render_setup computes it (float64)
        n_pixel_y = int(np.ceil(oversampling * (y_max - y_min)))
        n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
        if n_pixel_x <= 0 or n_pixel_y <= 0:
            raise SplatBackendError("empty viewport")
        image_bytes = n_pixel_x * n_pixel_y * 4

        if self._image_size < image_bytes:
            self._image = device.create_buffer(
                size=image_bytes,
                usage=(
                    wgpu.BufferUsage.STORAGE
                    | wgpu.BufferUsage.COPY_SRC
                    | wgpu.BufferUsage.COPY_DST
                ),
            )
            self._image_size = image_bytes
            self._group0 = None
        if self._group0 is None:
            self._group0 = device.create_bind_group(
                layout=self._group0_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self._uniforms,
                            "offset": 0,
                            "size": 48,
                        },
                    },
                    {
                        "binding": 1,
                        "resource": {
                            "buffer": self._image,
                            "offset": 0,
                            "size": self._image_size,
                        },
                    },
                    {
                        "binding": 2,
                        "resource": {
                            "buffer": self._aux,
                            "offset": 0,
                            "size": 8,
                        },
                    },
                ],
            )

        uniforms_base = (
            oversampling,
            x_min,
            y_min,
            x_max,
            y_max,
            min_blur_width,
            float(n_pixel_x),
            float(n_pixel_y),
        )

        # the fixed-point scale adapts to the data: on accumulator
        # overflow the whole request is retried coarser, and the working
        # scale is remembered, so a stream of GUI renders converges once
        scale = self._scale if blur_method else 1.0
        while True:
            renderings, overflow, raw_max = self._render_once(
                columns,
                blur_method,
                uniforms_base,
                image_bytes,
                n_pixel_x,
                n_pixel_y,
                scale,
            )
            if not overflow:
                break
            scale /= 16.0
            if scale < 16.0:
                raise SplatBackendError(
                    "fixed-point accumulator overflow at minimum scale; "
                    "falling back to CPU rendering"
                )
        if blur_method:
            self._scale = scale
            # regrow towards full precision when plenty of headroom
            # remains (16x margin against densification between frames)
            if scale < _MAX_SCALE and raw_max * 64 < 2**31:
                self._scale = min(_MAX_SCALE, scale * 4.0)
        return renderings

    def _render_once(
        self,
        columns,
        blur_method,
        uniforms_base,
        image_bytes,
        n_pixel_x,
        n_pixel_y,
        scale,
    ):
        device = self._device
        pipeline = self._pipelines["cs_gaussian" if blur_method else "cs_hist"]
        readbacks = []
        encoder = device.create_command_encoder()
        for channel in columns:
            uniforms = np.zeros(12, dtype=np.float32)
            uniforms[:8] = uniforms_base
            uniforms[8:11].view(np.uint32)[:] = (
                len(channel),
                n_pixel_x,
                n_pixel_y,
            )
            uniforms[11] = scale
            device.queue.write_buffer(self._uniforms, 0, uniforms.tobytes())
            group1 = self._channel_bind_group(channel, blur_method)
            encoder.clear_buffer(self._image, 0, image_bytes)
            encoder.clear_buffer(self._aux, 0, 8)
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, self._group0)
            compute_pass.set_bind_group(1, group1)
            n_groups = min(
                _MAX_WORKGROUPS, -(-len(channel) // _WORKGROUP) or 1
            )
            compute_pass.dispatch_workgroups(n_groups)
            compute_pass.end()
            readback = device.create_buffer(
                size=image_bytes + 8,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            )
            encoder.copy_buffer_to_buffer(
                self._image, 0, readback, 0, image_bytes
            )
            encoder.copy_buffer_to_buffer(
                self._aux, 0, readback, image_bytes, 8
            )
            readbacks.append(readback)
            # uniforms are written per channel: flush the encoder so the
            # write lands before the next channel's dispatch
            device.queue.submit([encoder.finish()])
            encoder = device.create_command_encoder()
        device.queue.submit([encoder.finish()])

        renderings = []
        overflow_any = False
        raw_max = 0
        for readback in readbacks:
            readback.map_sync(wgpu.MapMode.READ)
            raw = np.frombuffer(readback.read_mapped(), dtype=np.uint32)
            n, overflow = (int(v) for v in raw[-2:])
            counts = raw[:-2]
            if blur_method:
                channel_max = int(counts.max(initial=0))
                raw_max = max(raw_max, channel_max)
                # fixed point back to float; the scale is a power of
                # two, so dividing in f32 is exact while the counts fit
                # its 24-bit mantissa, else go through f64
                if channel_max < 2**24:
                    image = counts.astype(np.float32) / np.float32(scale)
                else:
                    image = (counts.astype(np.float64) / scale).astype(
                        np.float32
                    )
            else:
                image = counts.astype(np.float32)
            image = image.reshape(n_pixel_y, n_pixel_x)
            readback.unmap()
            overflow_any = overflow_any or bool(overflow)
            renderings.append((n, image))
        return renderings, overflow_any, raw_max

    def _channel_bind_group(self, channel, blur_method):
        """Storage buffers + bind group for one channel, cached by
        array identity while the arrays stay alive."""
        if blur_method:
            arrays = (channel.x, channel.y, channel.lpx, channel.lpy)
        else:
            # histogram ignores precision; bind x twice as a placeholder
            arrays = (channel.x, channel.y, channel.x, channel.x)
        key = tuple(id(a) for a in arrays)
        cached = self._buffer_cache.get(key)
        if cached is not None:
            return cached[2]
        buffers = tuple(
            self._device.create_buffer_with_data(
                data=np.ascontiguousarray(a, dtype=np.float32),
                usage=wgpu.BufferUsage.STORAGE,
            )
            for a in arrays
        )
        group = self._device.create_bind_group(
            layout=self._group1_layout,
            entries=[
                {
                    "binding": binding,
                    "resource": {
                        "buffer": buffer,
                        "offset": 0,
                        "size": buffer.size,
                    },
                }
                for binding, buffer in enumerate(buffers)
            ],
        )
        self._buffer_cache[key] = (arrays, buffers, group)
        return group

    def clear_cache(self) -> None:
        """Drop cached channel buffers (frees GPU memory)."""
        for _, buffers, _ in self._buffer_cache.values():
            for buffer in buffers:
                buffer.destroy()
        self._buffer_cache.clear()

    def close(self) -> None:
        """Release the device and all cached resources."""
        self.clear_cache()
        self._image = None
        self._group0 = None
        self._device.destroy()
