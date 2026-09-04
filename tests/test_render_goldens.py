"""Golden-image regression tests for picasso.render.

Every scene in ``tests/_render_golden_scenes.py`` is rendered with the
current implementation and compared against the frozen arrays in
``tests/data/render_goldens.npz``. Unlike the property tests in
``test_render.py`` (mass conservation, shapes, rotation semantics),
these tests pin the exact pixel values, so kernel refactors have to
prove they leave the output unchanged.

Comparison policy: float arrays use a tight tolerance (rtol 1e-5) rather
than bit-exact equality -- ``exp`` differs at the last bit across libm
implementations and numba versions, and parallel splatting reorders
float summation; both effects sit far below 1e-5 while any real kernel
bug (centering, sigma scaling, normalization) shifts values at percent
level. uint8 RGB output may differ by at most 1 count per pixel (values
sitting exactly on a rounding boundary).

To intentionally change rendering output, regenerate the archive with
``python tests/regen_render_goldens.py`` and commit the npz diff.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
import pytest

from picasso import io

from tests._render_golden_scenes import (
    GOLDENS_PATH,
    KIND_KEYS,
    SCENES,
    prepare_inputs,
    run_scene,
)

REGEN_HINT = (
    "run 'python tests/regen_render_goldens.py' and commit the result if "
    "this rendering change is intended"
)


@pytest.fixture(scope="module")
def golden_data():
    try:
        return np.load(GOLDENS_PATH)
    except FileNotFoundError:
        pytest.fail(f"{GOLDENS_PATH} is missing; {REGEN_HINT}")


@pytest.fixture(scope="module")
def locs_data():
    return io.load_locs("./tests/data/testdata_locs.hdf5")


@pytest.fixture(scope="module")
def inputs(locs_data):
    return prepare_inputs(locs_data[0])


@pytest.fixture(scope="module")
def info(locs_data):
    return locs_data[1]


def test_goldens_match_scene_table(golden_data):
    """The archive holds exactly the arrays the scene table defines --
    catches stale goldens after the table changes."""
    expected = {
        f"{scene['name']}::{key}"
        for scene in SCENES
        for key in KIND_KEYS[scene["kind"]]
    }
    assert set(golden_data.files) == expected, REGEN_HINT


@pytest.mark.parametrize("scene", SCENES, ids=[s["name"] for s in SCENES])
def test_golden(scene, inputs, info, golden_data):
    out = run_scene(scene, inputs, info)
    for key, arr in out.items():
        golden = golden_data[f"{scene['name']}::{key}"]
        assert arr.shape == golden.shape, (
            f"{scene['name']}::{key} shape {arr.shape} != golden "
            f"{golden.shape}; {REGEN_HINT}"
        )
        if arr.dtype == np.uint8:
            diff = np.abs(arr.astype(np.int16) - golden.astype(np.int16))
            assert diff.max() <= 1, (
                f"{scene['name']}::{key}: uint8 output deviates by up to "
                f"{diff.max()} counts at {np.unravel_index(diff.argmax(), diff.shape)} "
                f"({(diff > 1).sum()} pixels beyond rounding noise); {REGEN_HINT}"
            )
        else:
            np.testing.assert_allclose(
                arr,
                golden,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"{scene['name']}::{key} deviates from golden; {REGEN_HINT}",
            )
