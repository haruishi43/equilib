#!/usr/bin/env python3

import torch
import torch.nn.functional as F


def grid_sample(
    img: torch.tensor,
    grid: torch.tensor,
    device: torch.device = torch.device('cpu'),
    mode: str = 'bilinear',
) -> torch.tensor:
    r"""Torch Grid Sample (default)
        Uses `torch.nn.functional.grid_sample`
        By far the best way to sample

    params:
        img: Tensor[B, C, H, W]  or Tensor[C, H, W]
        grid: Tensor[B, 2, H, W] or Tensor[2, H, W]
        device: torch.device
        mode: `bilinear` or `nearest`

    returns:
        out: Tensor[B, C, H, W] or Tensor[C, H, W]
            where H, W are grid size
    """
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
        grid = grid.unsqueeze(0)

    # grid in shape: (batch, channel, h_out, w_out)
    # grid out shape: (batch, h_out, w_out, channel)
    batch, channels, h_img, w_img = img.shape
    batch_, xy, h_out, w_out = grid.shape
    assert batch == batch_, "Batch size mismatch"

    # normalize grid -1~1
    grid = grid.permute(0, 2, 3, 1)
    norm_uj = 2 * (grid[:, :, :, 0] - h_img/2) / h_img
    norm_ui = 2 * (grid[:, :, :, 1] - w_img/2) / w_img
    # grid sample takes xy, not (height, width)
    grid[:, :, :, 0] = norm_ui
    grid[:, :, :, 1] = norm_uj

    out = F.grid_sample(
        img,
        grid,
        mode=mode,
        align_corners=True,
    )
    if out.shape[0] == 1:
        out = out.squeeze(0)
    return out
