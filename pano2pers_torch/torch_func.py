#!/usr/bin/env python3

import torch
import torch.nn.functional as F


def grid_sample(
    img: torch.tensor,
    grid: torch.tensor,
    device: torch.device = torch.device('cpu'),
    mode: str = 'bilinear',
) -> torch.tensor:
    # batch, c, h_out, w_out = grid.shape
    # grid = grid.view(batch, h_out, w_out, c)
    # grid shape: (batch, h_out, w_out, channel)
    img = img.expand(grid.size(0), -1, -1, -1)
    out = F.grid_sample(
        img,
        grid,
        mode=mode,
        align_corners=False,
    ).squeeze(0)
    return out