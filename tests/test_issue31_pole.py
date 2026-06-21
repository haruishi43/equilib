#!/usr/bin/env python3

"""Regression for issue #31 — interpolation artifact on pitch rotation.

The transforms that sample from an equirectangular source map each output
direction to a source pixel via `convert_grid`. Longitude is periodic (wrap),
but **latitude is not** — near the pole it must clamp. The original code wrapped
latitude with a modulo, folding the pole band onto the opposite edge and pulling
sky pixels into the floor.

This pins the fix at the `convert_grid` level (white-box, but it is exactly the
invariant that broke): sweeping latitude through both poles, the vertical pixel
coordinate must stay in range and monotonic — a wrap shows up as a ~h jump.
"""

import importlib

import numpy as np

import pytest

# transforms that sample from an equirectangular source (pers2equi/cube2equi
# sample from other sources and are unaffected)
TRANSFORMS = ["equi2equi", "equi2pers", "equi2cube"]


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("method", ["robust", "faster"])
def test_convert_grid_clamps_latitude(transform: str, method: str) -> None:
    mod = importlib.import_module(f"equilib.{transform}.numpy")
    h = w = 64

    # unit direction vectors sweeping latitude phi from one pole to the other
    # at a fixed longitude
    phis = np.linspace(-np.pi / 2 + 1e-4, np.pi / 2 - 1e-4, 400)
    dirs = np.stack([np.cos(phis), np.zeros_like(phis), np.sin(phis)], axis=-1)
    dirs = dirs[None, :, None, :].astype(np.float64)  # (b, H', W', 3)

    grid = np.asarray(mod.convert_grid(dirs, h_equi=h, w_equi=w, method=method))
    uj = grid[:, 0].ravel()  # vertical (latitude) source coordinate

    assert uj.min() >= 0.0 and uj.max() <= h - 1, (
        f"{transform}/{method}: uj outside [0, h-1] -> {uj.min()}..{uj.max()}"
    )
    # a wrapped latitude jumps by ~h near the pole; a clamped one stays smooth
    max_jump = float(np.abs(np.diff(uj)).max())
    assert max_jump < 2.0, (
        f"{transform}/{method}: latitude wrap discontinuity "
        f"(max adjacent jump {max_jump:.1f} px ~ h={h})"
    )
