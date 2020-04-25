#!/usr/bin/env python3

from typing import List

import torch
import torch.nn.functional as F


def linear_interp(v0, v1, d, l):
    r"""Basic Linear Interpolation
    """
    return v0*(1-d)/l + v1*d/l


def interp2d(
    Q: List[torch.tensor],
    dy: torch.tensor, dx: torch.tensor,
    mode: str = 'bilinear',
) -> torch.tensor:
    r"""Naive Interpolation
        (y,x): target pixel
        mode: interpolation mode
    """
    q00, q10, q01, q11 = Q
    if mode == 'bilinear':
        f0 = linear_interp(q00, q01, dx, 1)
        f1 = linear_interp(q10, q11, dx, 1)
        return linear_interp(f0, f1, dy, 1)
    elif mode == 'nearest':
        raise NotImplementedError
    else:
        raise NotImplementedError


def grid_sample(
    img: torch.tensor,
    grid: torch.tensor,
    device: torch.device = torch.device('cpu'),
    mode: str = 'bilinear',
) -> torch.tensor:
    r"""Torch Grid Sample
        Supports batch 
    """
    assert len(img.shape) == len(grid.shape), \
        "ERR: img and grid does not match"
    assert len(img.shape) > 2, \
        "ERR: dim needs to be 3 or 4"
    if len(img.shape) == len(grid.shape) == 3:
        img = img.unsqueeze(0)
        grid = grid.unsqueeze(0)
    batch, channels, h_in, w_in = img.shape
    _, _, h_out, w_out = grid.shape
    _dtype = img.dtype
    _max = torch.tensor(1., device=device)
    _min = torch.tensor(0., device=device)

    # Initialize output image
    out = torch.zeros(
        (batch, channels, h_out, w_out),
        dtype=_dtype,
        device=device
    )

    min_grid = torch.floor(grid).type(torch.int64)
    max_grid = min_grid + 1
    d_grid = grid - min_grid

    max_grid[:,0,:,:] = torch.where(
        max_grid[:,0,:,:] >= h_in,
        max_grid[:,0,:,:] - h_in,
        max_grid[:,0,:,:])
    max_grid[:,1,:,:] = torch.where(
        max_grid[:,1,:,:] >= w_in,
        max_grid[:,1,:,:] - w_in,
        max_grid[:,1,:,:])

    y_mins = min_grid[:,0,:,:]
    x_mins = min_grid[:,1,:,:]
    y_mins = y_mins.view(-1)
    x_mins = x_mins.view(-1)

    y_maxs = max_grid[:,0,:,:]
    x_maxs = max_grid[:,1,:,:]
    y_maxs = y_maxs.view(-1)
    x_maxs = x_maxs.view(-1)

    y_d = d_grid[:,0,:,:]
    x_d = d_grid[:,1,:,:]
    y_d = y_d.view(-1)
    x_d = x_d.view(-1)

    Q00 = img[:,:,y_mins,x_mins]
    Q10 = img[:,:,y_maxs,x_mins]
    Q01 = img[:,:,y_mins,x_maxs]
    Q11 = img[:,:,y_maxs,x_maxs]

    out = interp2d(
        [Q00, Q10, Q01, Q11],
        y_d, x_d,
        mode='bilinear')

    out = torch.where(out >= _max, _max, out)
    out = torch.where(out < _min, _min, out)
    out = out.reshape(batch, channels, h_out, w_out)
    out = out.type(_dtype)
    if out.shape[0] == 1:
        out = out.squeeze(0)
    return out