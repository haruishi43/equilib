#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Optional, Union

import collections

import numpy as np

from equilib.numpy_utils import calculate_tangent_angles

from equilib.grid_sample import numpy_grid_sample

__all__ = ["run"]


def dict2list(batch: List[Dict[int, np.array]]):
    output = []
    for d in batch:
        od = collections.OrderedDict(sorted(d.items()))
        output.extend(list(od.values()))
    return output


def ceil_max(a: np.array):
    """ Method for rounding digits

        For a > 0 returns higher, e.g. a = 1.1 output = 2
        For a < 0 returns lower, e.g. a = -0.1 output = 1
    """
    return np.sign(a)*np.ceil(np.abs(a))

def rodrigues(rot_vector: np.ndarray):
    """ Rodrigues transformation 
        https://docs.opencv.org/4.5.3/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
    """
    if rot_vector.shape == (3,):
        # do if input is a vector
        theta = np.linalg.norm(rot_vector)
        i = np.eye(3)
        if theta < 1e-9:
            return i
        r = rot_vector / theta
        rr = np.tile(r, (3,1))*np.tile(r, (3,1)).T
        rmap = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        return (np.cos(theta)*i + (1-np.cos(theta))*rr + np.sin(theta)*rmap).astype(np.float32)
    else:
        # do if vector is a matrix
        R = rot_vector
        r = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
        s = np.sqrt(np.sum(r**2)/4)
        c = (np.sum(np.eye(3)*R)-1)/2
        c = np.clip(c, -1, 1)
        theta_ = np.arccos(c)
        
        if c > 0:
            return np.zeros(3, np.float32)
        
        if s < 1e-5:
            t = (R[0,0]+1)/2
            r[0] = np.sqrt(np.max([t, 0]))
            t = (R[1,1]+1)/2
            r[1] = np.sqrt(np.max([t,0]))*ceil_max(R[0,1])
            t = (R[2,2]+1)/2
            r[2] = np.sqrt(np.max([t,0]))*ceil_max(R[0,2])
            abs_r = np.abs(r)
            abs_r -= abs_r[0]
            if (abs_r[1] > 0) and (abs_r[2] > 0) and (R[1,2] > 0 != r[1]*r[2]>0):
                r[2] = -r[2]
            theta_ /= np.linalg.norm(r)
            r *= theta_
        else:
            vth = 1/(2*s) * theta_
            r *= vth
            
        return r.reshape(3,1).astype(np.float32)


def get_equirec(
    img: np.ndarray, 
    fov_x: float,
    theta: int,
    phi: int,
    height: int,
    width: int,
    mode:str,
):
    _img = img
    _height, _width = _img.shape[1:]
    wFOV = fov_x
    THETA = theta
    PHI = phi
    hFOV = float(_height) / _width * fov_x

    w_len = np.tan(np.radians(wFOV / 2.0))
    h_len = np.tan(np.radians(hFOV / 2.0))
    #
    # THETA is left/right angle, PHI is up/down angle, both in degree
    #

    x,y = np.meshgrid(np.linspace(-180, 180,width),np.linspace(90,-90,height))
    
    x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
    y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
    z_map = np.sin(np.radians(y))

    xyz = np.stack((x_map,y_map,z_map),axis=2)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    R1 = rodrigues(z_axis * np.radians(THETA))
    R2 = rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)

    xyz = xyz.reshape([height * width, 3]).T
    xyz = np.dot(R2, xyz)
    xyz = np.dot(R1, xyz).T

    xyz = xyz.reshape([height , width, 3])
    inverse_mask = np.where(xyz[:,:,0]>0,1,0)

    xyz[:,:] = xyz[:,:]/np.repeat(xyz[:,:,0][:, :, np.newaxis], 3, axis=2)
    
    
    lon_map = np.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
                &(xyz[:,:,2]<h_len),(xyz[:,:,1]+w_len)/2/w_len*_width,0)
    lat_map = np.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
                &(xyz[:,:,2]<h_len),(-xyz[:,:,2]+h_len)/2/h_len*_height,0)
    mask = np.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
                &(xyz[:,:,2]<h_len),1,0)

    # TODO: FIX shapes and pipeline in sourcecode

    dtype = np.float32
    out = np.empty((1, 3, height, width), dtype=dtype)
    grid = np.stack((lat_map, lon_map), axis=0)
    grid = np.concatenate([grid[np.newaxis, ...]] * 1)
    imgt = np.concatenate([_img[np.newaxis, ...]] * 1)
    out = numpy_grid_sample(imgt, grid, out, mode=mode)
    out = out.squeeze()
    
    mask = mask * inverse_mask
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = np.transpose(mask, (2,0,1))
    persp = out * mask
    
    return persp, mask


def run(
    icomaps: List[np.ndarray],
    height: int,
    width: int,
    fov_x: float,
    mode: str
) -> np.ndarray:
    """Run Ico2Equi

    params:
    - icomaps (np.ndarray)
    - height, widht (int): output equirectangular image's size
    - fov_x (float): fov of horizontal axis in degrees
    - mode (str)

    returns:
    - equi (np.ndarray)

    NOTE: we assume that the input `horizon` is a 4 dim array

    """

    assert (
        len(icomaps) >= 1 and len(icomaps[0].shape)==4
    ), f"ERR: `icomaps` should be 4-dim (b, fn, c, h, w), but got {icomaps.shape}"

    icomaps_dtype = icomaps[0].dtype
    assert icomaps_dtype in (np.uint8, np.float32, np.float64), (
        f"ERR: input horizon has dtype of {icomaps_dtype}\n"
        f"which is incompatible: try {(np.uint8, np.float32, np.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as horizon
    dtype = (
        np.dtype(np.float32)
        if icomaps_dtype == np.dtype(np.uint8)
        else icomaps_dtype
    )
    assert dtype in (np.float32, np.float64), (
        f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
        f"try {(np.float32, np.float64)}"
    )

    bs,c = len(icomaps), icomaps[0].shape[1]

    # initialize output equi
    out_batch = np.empty((bs, c, height, width), dtype=icomaps_dtype)
    # calculate subdision level of input bathces
    subdivision_levels = [int(np.log(icomap.shape[0]/20)/np.log(4)) for icomap in icomaps]
    # calculate angles for target subdivion levels
    angles = calculate_tangent_angles(subdivision_levels)

    for bn, (imgs, angle) in enumerate(zip(icomaps, angles)):
        angle *= -1*180/np.pi
        out = np.empty((c, height, width), dtype=dtype)
        # merge_image is sum of reconstructed images
        # merge_mask is sum of latitude-longtitude masks
        merge_image = np.zeros((c,height,width))
        merge_mask = np.zeros((c,height,width))
        for img,[T,P] in zip(imgs, angle):
            img, mask = get_equirec(img,fov_x,T,P,height, width, mode=mode)
            merge_image += img
            merge_mask += mask
        # result image equals to dividing sum of images on sum of masks
        merge_mask = np.where(merge_mask==0,1,merge_mask)
        out = np.divide(merge_image,merge_mask)

        out = (
            out.astype(icomaps_dtype)
            if icomaps_dtype == np.dtype(np.uint8)
            else np.clip(out, 0.0, 1.0)
        )
        out_batch[bn] = out

    return out_batch
