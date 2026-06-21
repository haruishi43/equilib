#!/usr/bin/env python3

"""Is the rotation-invariant grid cache worth it?

The class API caches the rotation-invariant grid prep (`m`/`G`/`xyz`); the func
API recomputes it every call. This benchmark runs an identical workload through
both — many calls with a *varying* rotation but a *fixed* input shape/dtype —
so the only difference is prep recomputation. The per-call delta is exactly the
time the cache saves, and `1 - cached/pure` is prep's share of total runtime.

Standalone (not pytest):  uv run python benchmarks/cache_benchmark.py
"""

from math import sin
import statistics
import time

import numpy as np

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

N = 150  # timed iterations
WARMUP = 15


def _rot(i: int):
    # varies every call -> rotation/matmul/grid-sample rerun, but the grid
    # prep (shape/dtype fixed) is a cache hit on the class path.
    return {"roll": 0.0, "pitch": 0.4 * sin(i), "yaw": 0.05 * i}


def _bench(call):
    for i in range(WARMUP):
        call(i)
    times = []
    for i in range(N):
        t0 = time.perf_counter()
        call(i)
        times.append(time.perf_counter() - t0)
    # min = cleanest compute floor (least OS jitter); median = typical
    return min(times), statistics.median(times)


def _input(shape, backend):
    rng = np.random.default_rng(0)
    arr = (rng.random(shape) * 255.0).astype(np.float32)
    return arr if backend == "numpy" else torch.from_numpy(arr)


# name -> (class, func, ctor/func config, per-call extra kwargs, input shape)
CASES = {
    "equi2pers": dict(
        cls=Equi2Pers,
        func=equi2pers,
        config=dict(height=480, width=640, fov_x=90.0),
        call=dict(),
        in_shape=(3, 1024, 2048),
    ),
    "pers2equi": dict(
        cls=Pers2Equi,
        func=pers2equi,
        config=dict(height=512, width=1024),
        call=dict(fov_x=90.0),
        in_shape=(3, 480, 640),
    ),
    "equi2equi": dict(
        cls=Equi2Equi,
        func=equi2equi,
        config=dict(height=512, width=1024),
        call=dict(),
        in_shape=(3, 512, 1024),
    ),
    "equi2cube": dict(
        cls=Equi2Cube,
        func=equi2cube,
        config=dict(w_face=256, cube_format="dice"),
        call=dict(),
        in_shape=(3, 512, 1024),
    ),
}


def run(backend: str) -> None:
    print(f"\n=== backend: {backend}  (N={N} calls) ===")
    print(
        f"{'transform':<11} | {'pure':>7} {'cached':>7} {'saved':>6} "
        f"{'prep%':>6}  (min ms) | {'pure':>7} {'cached':>7} "
        f"{'prep%':>6}  (median ms)"
    )
    for name, c in CASES.items():
        inp = _input(c["in_shape"], backend)

        def pure_call(i, c=c, inp=inp):
            x = inp.copy() if backend == "numpy" else inp.clone()
            return c["func"](x, _rot(i), **c["config"], **c["call"])

        obj = c["cls"](**c["config"])

        def cached_call(i, obj=obj, c=c, inp=inp):
            x = inp.copy() if backend == "numpy" else inp.clone()
            return obj(x, _rot(i), **c["call"])

        p_min, p_med = (v * 1e3 for v in _bench(pure_call))
        c_min, c_med = (v * 1e3 for v in _bench(cached_call))
        min_pct = 100.0 * (p_min - c_min) / p_min
        med_pct = 100.0 * (p_med - c_med) / p_med
        print(
            f"{name:<11} | {p_min:>7.2f} {c_min:>7.2f} "
            f"{p_min - c_min:>6.2f} {min_pct:>5.1f}%           | "
            f"{p_med:>7.2f} {c_med:>7.2f} {med_pct:>5.1f}%"
        )


if __name__ == "__main__":
    torch.set_num_threads(torch.get_num_threads())
    run("numpy")
    run("torch")
