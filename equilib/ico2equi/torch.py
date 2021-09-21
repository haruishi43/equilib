#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import tensor
from torch import cos, sin, tan, arccos, sqrt, sum
from torch import deg2rad as radians

from equilib.grid_sample import torch_grid_sample
from equilib.torch_utils import (
    get_device,
    pi,
    calculate_tangent_angles
)

__all__ = ["convert2batches", "run"]


def ceil_max(a: torch.Tensor):
    return torch.sign(a)*torch.ceil(torch.abs(a))

def rodrigues(
    rot_vector: torch.Tensor,
    device: torch.device = torch.device("cpu")
    ):
    if len(rot_vector.shape) == 1:
        theta = torch.linalg.norm(rot_vector)
        i = torch.eye(3, device=device)
        if theta < 1e-9:
            return i
        r = rot_vector / theta
        rr = r.tile((3,1))*r.tile((3,1)).T
        rmap = tensor([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ], device=device)
        return (cos(theta)*i + (1-cos(theta))*rr + sin(theta)*rmap)
    else:
        R = torch.clone(rot_vector)
        r = tensor([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
        s = sqrt(sum(r**2)/4)
        c = (sum(torch.eye(3, device=device)*R)-1)/2
        c = torch.clip(c, -1, 1)
        theta_ = arccos(c)
        
        if c > 0:
            return torch.zeros(3, dtype=torch.float32, device=device)
        
        if s < 1e-5:
            t = (R[0,0]+1)/2
            r[0] = sqrt(torch.max([t, 0]))
            t = (R[1,1]+1)/2
            r[1] = sqrt(torch.max([t,0]))*ceil_max(R[0,1])
            t = (R[2,2]+1)/2
            r[2] = sqrt(torch.max([t,0]))*ceil_max(R[0,2])
            abs_r = torch.abs(r)
            abs_r -= abs_r[0]
            if (abs_r[1] > 0) and (abs_r[2] > 0) and (R[1,2] > 0 != r[1]*r[2]>0):
                r[2] = -r[2]
            theta_ /= torch.linalg.norm(r)
            r *= theta_
        else:
            vth = 1/(2*s) * theta_
            r *= vth
            
        return r.reshape(3,1)


def get_equirec(
    img: torch.Tensor, 
    fov_x: float,
    theta: int,
    phi: int,
    height: int,
    width: int,
    device: torch.device = torch.device("cpu")
):
    device = img.device
    _img = img
    _height = tensor(_img.shape[1], dtype=torch.float32, device=device)
    _width = tensor(_img.shape[2], dtype=torch.float32, device=device) 
    wFOV = tensor(fov_x, dtype=torch.float32, device=device)
    hFOV = _height / _width * fov_x

    w_len = tan(radians(wFOV / 2.0))
    h_len = tan(radians(hFOV / 2.0))
    #
    # THETA is left/right angle, PHI is up/down angle, both in degree
    #

    x,y = torch.meshgrid(
        torch.linspace(-180, 180,width, device=device),
        torch.linspace(90,-90,height, device=device)
    )
    
    x_map = cos(radians(x)) * cos(radians(y))
    y_map = sin(radians(x)) * cos(radians(y))
    z_map = sin(radians(y))

    xyz = torch.stack((x_map,y_map,z_map))
    xyz = torch.permute(xyz, (2,1,0))

    y_axis = tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_axis = tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    R1 = rodrigues(z_axis * radians(theta), device=device)
    R2 = rodrigues(torch.mv(R1, y_axis) * radians(-phi), device=device)

    R1 = torch.linalg.inv(R1)
    R2 = torch.linalg.inv(R2)

    xyz = xyz.reshape([height * width, 3]).T
    xyz = torch.mm(R2, xyz)
    xyz = torch.mm(R1, xyz).T

    xyz = xyz.reshape([height , width, 3])
    inverse_mask = torch.where(xyz[:,:,0]>0,1,0)
    
    xyz[:,:] = xyz[:,:]/xyz[:,:,0].unsqueeze(2).repeat(1,1,3)
    
    
    lon_map = torch.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
                &(xyz[:,:,2]<h_len),(xyz[:,:,1]+w_len)/2/w_len*_width,tensor(0, dtype=torch.float32, device=device))
    lat_map = torch.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
                &(xyz[:,:,2]<h_len),(-xyz[:,:,2]+h_len)/2/h_len*_height,tensor(0, dtype=torch.float32, device=device))
    mask = torch.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
                &(xyz[:,:,2]<h_len),tensor(1, dtype=torch.float32, device=device),tensor(0, dtype=torch.float32, device=device))

    # TODO: FIX shapes and pipeline in sourcecode

    out = torch.empty((1, 3, height, width), dtype=torch.float32, device=device)
    grid = torch.stack((lat_map, lon_map))
    grid = grid.unsqueeze(0)
    imgt = _img.unsqueeze(0)
    out = torch_grid_sample(imgt, grid, out, "bilinear")
    out = out.squeeze()
    
    mask = mask * inverse_mask
    mask = mask.unsqueeze(2).repeat(1,1,3)
    mask = torch.permute(mask, (2,0,1))
    persp = out * mask
    
    return persp, mask


def run(
    icomaps: List[torch.Tensor],
    height: int,
    width: int,
    fov_x: float,
    mode: str = 'list',
    backend: str = 'native'
) -> torch.Tensor:
    """Run Cube2Equi

    params:
    - icomaps (np.ndarray)
    - height, widht (int): output equirectangular image's size
    - mode (str)

    return:
    - equi (np.ndarray)

    NOTE: we assume that the input `horizon` is a 4 dim array

    """

    assert (
        len(icomaps[0].shape) == 4
    ), f"ERR: `horizon` should be 4-dim (b, c, h, w), but got {icomaps[0].shape}"

    icomaps_dtype = icomaps[0].dtype
    assert icomaps_dtype in (
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
    ), (
        f"ERR: input horizon has dtype of {icomaps_dtype}which is\n"
        f"incompatible: try {(torch.uint8, torch.float16, torch.float32, torch.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as horizon
    if icomaps[0].device.type == "cuda":
        dtype = torch.float32 if icomaps_dtype == torch.uint8 else icomaps_dtype
        assert dtype in (torch.float16, torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float16, torch.float32, torch.float64)}"
        )
    else:
        # NOTE: for cpu, it can't use half-precision
        dtype = torch.float32 if icomaps_dtype == torch.uint8 else icomaps_dtype
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float32, torch.float64)}"
        )
    if backend == "native" and icomaps_dtype == torch.uint8:
        # FIXME: hacky way of dealing with images that are uint8 when using
        # torch.grid_sample
        for i, ico in enumerate(icomaps):
            icomaps[i] = ico.type(torch.float32)

    bs, c = len(icomaps), icomaps[0].shape[1]
    device = get_device(icomaps[0])
    cpu_device = torch.device("cpu")

    # initialize output equi
    out_batch = torch.empty((bs, c, height, width), dtype=icomaps_dtype, device=device)
    subdivision_levels = [int(np.log(icomap.shape[0]/20)/np.log(4)) for icomap in icomaps]
    angles = calculate_tangent_angles(subdivision_levels, device=device)
    # torch routine
    zero = tensor(0, dtype=dtype, device=device)
    one = tensor(1, dtype=dtype, device=device)
    for bn, (imgs, angle) in enumerate(zip(icomaps, angles)):
        angle *= -1*180/torch.clone(pi).to(dtype=dtype, device=device) 
        out = torch.empty((c, height, width), dtype=dtype, device=device)
        merge_image = torch.zeros((c,height,width), dtype=dtype, device=device)
        merge_mask = torch.zeros((c,height,width), dtype=dtype, device=device)
        for img,[T,P] in zip(imgs, angle):
            img, mask = get_equirec(img,fov_x,T,P,height, width, device=device)
            merge_image += img
            merge_mask += mask
        merge_mask = torch.where(merge_mask==zero,one,merge_mask)
        out = np.divide(merge_image,merge_mask)

        out = (
            out.type(icomaps_dtype)
            if icomaps_dtype == torch.uint8
            else torch.clip(out, 0.0, 1.0)
        )
        out_batch[bn] = out

    return out_batch
