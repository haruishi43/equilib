#!/usr/bin/env python3

import os
import os.path as osp

import time

from PIL import Image
import numpy as np

from pano2pers_numpy import (
    naive_sample,
    faster_sample,
    utils
)


if __name__ == "__main__":
    data_path = osp.join('.', 'data')
    pano_path = osp.join(data_path, '8081_earthmap4k.jpg')

    tic = time.perf_counter()
    pano_img = Image.open(pano_path)

    # Sometimes images are RGBA
    pano_img = pano_img.convert('RGB')
    pano = np.asarray(pano_img)

    pano = np.transpose(pano, (2,0,1))

    toc = time.perf_counter()
    print(f"Process Pano: {toc - tic:0.4f} seconds")

    _, h_pano, w_pano = pano.shape
    print('panorama size:')
    print(h_pano, w_pano)
    
    # Variables:
    h_pers = 480
    w_pers = 640
    rot = {
        'roll': 0,
        'pitch': 0,
        'yaw': 0,
    }
    fov_x = 90

    tic = time.perf_counter()
    m = utils.create_coord(h_pers, w_pers)
    K = utils.create_K(h_pers, w_pers, fov_x)
    R = utils.create_rot_mat(**rot)
    toc = time.perf_counter()
    print(f"Process m, K, R: {toc - tic:0.4f} seconds")

    # m = P M
    # P = K [R^-1 | t] = K R^-1 (in this case...)
    # R K^-1 m = M

    tic = time.perf_counter()
    K_inv = np.linalg.inv(K)
    R_inv = np.linalg.inv(R)
    m = m[:, :, :, np.newaxis]
    toc = time.perf_counter()
    print(f"Take Inverse: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    M = R_inv @ K_inv @ m
    M = M.squeeze(3)
    toc = time.perf_counter()
    print(f"M = R^-1 K^-1 m: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    phi, theta = utils.pixel_wise_rot(M)
    toc = time.perf_counter()
    print(f"pixel_wise_rot: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    # this normalization is needed? how to represent it in equation...
    ui = (theta - np.pi) * w_pano / (2 * np.pi)
    uj = (phi - np.pi / 2) * h_pano / np.pi

    ui = np.where(ui < 0, ui + w_pano, ui)
    ui = np.where(ui >= w_pano, ui - w_pano, ui)
    uj = np.where(uj < 0, uj + h_pano, uj)
    uj = np.where(uj >= h_pano, uj - h_pano, uj)

    # pano = pano / 255.  # scaling 0.0 - 1.0

    grid = np.stack((uj, ui), axis=0)
    toc = time.perf_counter()
    print(f"preprocess grid: {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    sampled = naive_sample(pano, grid, mode='bilinear')
    toc = time.perf_counter()
    print(f"naive: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    sampled = faster_sample(pano, grid, mode='bilinear')
    toc = time.perf_counter()
    print(f"faster: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    # after sample
    pers = np.transpose(sampled, (1,2,0))
    # pers = (pers * 255).astype(np.uint8)  # unscaling
    pers_img = Image.fromarray(pers)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")

    pers_path = osp.join(data_path, 'output_numpy.jpg')
    pers_img.save(pers_path)