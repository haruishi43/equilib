#!/usr/bin/env python3

"""The class API caches the rotation-invariant grid prep; the func API does not.

This pins the invariant that the cache is a pure optimization: for every
transform, the cached (class) output must be *bitwise* identical to the
uncached (func) output across rotations, dtypes, and backends.
"""

from copy import deepcopy

import numpy as np

import pytest

import torch

from equilib import (
    Equi2Cube,
    Equi2Equi,
    Equi2Pers,
    Pers2Equi,
    equi2cube,
    equi2equi,
    equi2pers,
    pers2equi,
)
from equilib._cache import GRID_CACHE_MAXSIZE

ROTS = [
    {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    {"roll": 0.1, "pitch": 0.3, "yaw": -0.5},
    {"roll": -0.2, "pitch": 0.4, "yaw": 1.2},
]

# Each case shares one `config` dict between the class constructor and the func
# call. `call` holds extra per-call kwargs (pers2equi takes `fov_x` at runtime).
CASES = {
    "equi2pers": dict(
        cls=Equi2Pers,
        func=equi2pers,
        config=dict(height=32, width=48, fov_x=90.0),
        call=dict(),
        in_shape=(3, 64, 128),
    ),
    "equi2equi": dict(
        cls=Equi2Equi,
        func=equi2equi,
        config=dict(height=64, width=128),
        call=dict(),
        in_shape=(3, 64, 128),
    ),
    "equi2cube": dict(
        cls=Equi2Cube,
        func=equi2cube,
        config=dict(w_face=32, cube_format="dice"),
        call=dict(),
        in_shape=(3, 64, 128),
    ),
    "pers2equi": dict(
        cls=Pers2Equi,
        func=pers2equi,
        config=dict(height=64, width=128),
        call=dict(fov_x=90.0),
        in_shape=(3, 32, 48),
    ),
}


def _input(shape, dtype, backend):
    rng = np.random.default_rng(42)
    arr = rng.random(shape) * 255.0
    if backend == "numpy":
        return arr.astype(dtype)
    return torch.from_numpy(arr).to(dtype)


def _clone(inp):
    return inp.copy() if isinstance(inp, np.ndarray) else inp.clone()


def _equal(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return torch.equal(a, b)


@pytest.mark.parametrize("name", list(CASES.keys()))
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_cached_matches_pure(name: str, backend: str) -> None:
    case = CASES[name]
    dtypes = (
        [np.float32, np.uint8]
        if backend == "numpy"
        else [torch.float32, torch.uint8]
    )

    for dtype in dtypes:
        inp = _input(case["in_shape"], dtype, backend)
        obj = case["cls"](**case["config"])
        assert obj._cache == {}, f"{name}/{backend}: cache not empty on init"

        for rot in ROTS:
            cached = obj(_clone(inp), deepcopy(rot), **case["call"])
            pure = case["func"](
                _clone(inp),
                deepcopy(rot),
                **case["config"],
                **case["call"],
            )
            assert _equal(cached, pure), (
                f"{name}/{backend}/{dtype}: cached output differs from pure "
                f"output at rot={rot}"
            )

        assert len(obj._cache) >= 1, (
            f"{name}/{backend}: cache stayed empty after calls"
        )


def test_cache_is_bounded() -> None:
    # Feeding many distinct input shapes (here: batch sizes) produces many
    # distinct cache keys; the per-instance cache must not grow without bound.
    obj = Equi2Pers(height=16, width=24, fov_x=90.0)
    base = np.random.default_rng(0).random((1, 3, 32, 64)).astype(np.float32)

    for bs in range(1, GRID_CACHE_MAXSIZE + 6):
        equi = np.repeat(base, bs, axis=0)
        rots = [{"roll": 0.0, "pitch": 0.0, "yaw": 0.0}] * bs
        obj(equi, rots)

    assert len(obj._cache) <= GRID_CACHE_MAXSIZE
