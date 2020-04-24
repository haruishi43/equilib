#!/usr/bin/env python3

import os
import os.path as osp

from PIL import Image
import numpy as np

from naive import grid_sample as naive_grid_sample
from faster import grid_sample as faster_grid_sample
from utils import (
    create_coord, create_K, create_rot_mat,
    pixel_wise_rot,
)


if __name__ == "__main__":
    data_path = osp.join('..', 'data')
    pano_path = osp.join(data_path, 'pano2.png')

    pano_img = Image.open(pano_path)

    # Sometimes images are RGBA
    pano_img = pano_img.convert('RGB')
    pano = np.asarray(pano_img)

    pano = np.transpose(pano, (2,0,1))
    _, h_pano, w_pano = pano.shape
    print('panorama size:')
    print(h_pano, w_pano)
    
    # Variables:
    h_pers = 480
    w_pers = 640
    rot = [45, 0, 0]
    fov_x = 90

    coord = create_coord(h_pers, w_pers)
    K = create_K(h_pers, w_pers, fov_x)
    R = create_rot_mat(rot)

    K_inv = np.linalg.inv(K)
    R_inv = np.linalg.inv(R)
    coord = coord[:, :, :, np.newaxis]

    rot_coord = R_inv @ K_inv @ coord
    rot_coord = rot_coord.squeeze(3)

    a, b = pixel_wise_rot(rot_coord)

    ui = (a + np.pi) * w_pano / (2 * np.pi)
    uj = (b + np.pi / 2) * h_pano / np.pi

    ui = np.where(ui < 0, ui + w_pano, ui)
    ui = np.where(ui >= w_pano, ui - w_pano, ui)
    uj = np.where(uj < 0, uj + h_pano, uj)
    uj = np.where(uj >= h_pano, uj - h_pano, uj)

    # pano = pano / 255.  # scaling 0.0 - 1.0

    grid = np.stack((uj, ui), axis=0)
    # sampled = naive_grid_sample(pano, grid, mode='bilinear')
    sampled = faster_grid_sample(pano, grid, mode='bilinear')

    # after sample
    pers = np.transpose(sampled, (1,2,0))
    # pers = (pers * 255).astype(np.uint8)  # unscaling
    pers_img = Image.fromarray(pers)

    pers_path = osp.join(data_path, 'output_.jpg')
    pers_img.save(pers_path)