#!/usr/bin/env python3

from typing import List

import torch

from equilib.common.torch_utils import get_device

from .interp import linear_interp


def interp2d(
    Q: List[torch.Tensor],
    dy: torch.Tensor,
    dx: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    r"""Naive Interpolation
    (y,x): target pixel
    mode: interpolation mode
    """
    q00, q10, q01, q11 = Q
    L = torch.tensor(1.0).to(get_device(Q))
    if mode == "bilinear":
        f0 = linear_interp(q00, q01, dx, L)
        f1 = linear_interp(q10, q11, dx, L)
        return linear_interp(f0, f1, dy, L)
    else:
        raise NotImplementedError


def grid_sample(
    img: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    r"""Torch Grid Sample (Custom)

    params:
    - img: Tensor[B, C, H, W]  or Tensor[C, H, W]
    - grid: Tensor[B, 2, H, W] or Tensor[2, H, W]
    - mode: `bilinear` or `nearest`

    returns:
    - out: Tensor[B, C, H, W] or Tensor[C, H, W]
        where H, W are grid size
    """
    assert len(img.shape) == len(grid.shape), "ERR: img and grid does not match"
    assert len(img.shape) > 2, "ERR: dim needs to be 3 or 4"
    if len(img.shape) == len(grid.shape) == 3:
        img = img.unsqueeze(0)
        grid = grid.unsqueeze(0)
    batches, channels, h_in, w_in = img.shape
    _, _, h_out, w_out = grid.shape

    assert get_device(img) == get_device(
        grid
    ), "ERR: img and grid does not match device " "{} vs {}".format(
        get_device(img), get_device(grid)
    )
    device = get_device(img)

    _dtype = img.dtype
    _max = torch.tensor(1.0, device=device)
    _min = torch.tensor(0.0, device=device)

    # Initialize output image
    out = torch.zeros(
        (batches, channels, h_out, w_out), dtype=_dtype, device=device
    )

    if mode == "bilinear":
        min_grid = torch.floor(grid).type(torch.int64)
        max_grid = min_grid + 1
        d_grid = grid - min_grid

        max_grid[:, 0, :, :] = torch.where(
            max_grid[:, 0, :, :] >= h_in,
            max_grid[:, 0, :, :] - h_in,
            max_grid[:, 0, :, :],
        )
        max_grid[:, 1, :, :] = torch.where(
            max_grid[:, 1, :, :] >= w_in,
            max_grid[:, 1, :, :] - w_in,
            max_grid[:, 1, :, :],
        )

        y_mins = min_grid[:, 0, :, :]
        x_mins = min_grid[:, 1, :, :]
        y_mins = y_mins.view(batches, -1)
        x_mins = x_mins.view(batches, -1)

        y_maxs = max_grid[:, 0, :, :]
        x_maxs = max_grid[:, 1, :, :]
        y_maxs = y_maxs.view(batches, -1)
        x_maxs = x_maxs.view(batches, -1)

        y_d = d_grid[:, 0, :, :]
        x_d = d_grid[:, 1, :, :]
        y_d = y_d.view(batches, -1)
        x_d = x_d.view(batches, -1)

        # Q00 = torch.zeros((batches, channels, y_mins.shape[-1]))
        # Q10 = torch.zeros((batches, channels, y_mins.shape[-1]))
        # Q01 = torch.zeros((batches, channels, y_mins.shape[-1]))
        # Q11 = torch.zeros((batches, channels, y_mins.shape[-1]))
        # FIXME: looping batch... slow...
        for b in range(batches):
            Q00 = img[b][:, y_mins[b], x_mins[b]]
            Q10 = img[b][:, y_maxs[b], x_mins[b]]
            Q01 = img[b][:, y_mins[b], x_maxs[b]]
            Q11 = img[b][:, y_maxs[b], x_maxs[b]]

            out[b] = interp2d(
                [Q00, Q10, Q01, Q11],
                y_d[b],
                x_d[b],
                mode=mode,
            ).view(channels, h_out, w_out)

        # NOTE: it is faster, but uses too much memory
        # def _flatten(t):
        #     return t.reshape((t.shape[0] * t.shape[1]))

        # # trying to reduce for loops
        # y_d = _flatten(y_d)
        # x_d = _flatten(x_d)
        # y_mins = _flatten(y_mins)
        # x_mins = _flatten(x_mins)
        # y_maxs = _flatten(y_maxs)
        # x_maxs = _flatten(x_maxs)

        # img = img.permute(1, 0, 2, 3)
        # img = img.view(channels, -1, h_in, w_in).squeeze(1)

        # out = interp2d(
        #     [
        #         img[:, y_mins, x_mins],
        #         img[:, y_maxs, x_mins],
        #         img[:, y_mins, x_maxs],
        #         img[:, y_maxs, x_maxs],
        #     ],
        #     y_d,
        #     x_d,
        #     mode=mode,
        # ).reshape(channels, batches, h_out, w_out).permute(1, 0, 2, 3)

    elif mode == "nearest":
        round_grid = torch.round(grid).type(torch.int64)
        y = round_grid[:, 0, :, :]
        x = round_grid[:, 1, :, :]
        y = y.view(batches, -1)
        x = x.view(batches, -1)

        for b in range(batches):
            out[b] = img[b][: y[b], x[b]]

    else:
        raise ValueError("{} is not available".format(mode))

    out = torch.where(out >= _max, _max, out)
    out = torch.where(out < _min, _min, out)
    out = out.reshape(batches, channels, h_out, w_out)
    out = out.type(_dtype)
    if out.shape[0] == 1:
        out = out.squeeze(0)
    return out
