#!/usr/bin/env python3

import torch


def create_coord(
    height: int,
    width: int,
) -> torch.Tensor:
    """Create mesh coordinate grid"""
    _xs = torch.linspace(0, width - 1, width)
    _ys = torch.linspace(0, height - 1, height)
    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    ys, xs = torch.meshgrid([_ys, _xs])
    zs = torch.ones_like(xs)
    coord = torch.stack((xs, ys, zs), dim=2)
    return coord
