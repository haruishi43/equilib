#!/usr/bin/env python3

import os.path as osp
import time

import numpy as np
from PIL import Image

from panolib.pano2pers import NumpyPano2Pers


def test_numpy_single():
    data_path = osp.join('.', 'tests', 'data')
    result_path = osp.join('.', 'tests', 'results')
    pano_path = osp.join(data_path, 'test.jpg')

    tic = time.perf_counter()
    pano_img = Image.open(pano_path)
    # Sometimes images are RGBA
    pano_img = pano_img.convert('RGB')
    pano = np.asarray(pano_img)
    pano = np.transpose(pano, (2, 0, 1))
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

    pano2pers = NumpyPano2Pers(
        w_pano=w_pano,
        h_pano=h_pano,
        w_pers=w_pers,
        h_pers=h_pers,
        fov_x=fov_x
    )
    sampled = pano2pers(
        pano=pano,
        rot=rot,
        sampling_method="faster",
        mode="bilinear",
    )

    tic = time.perf_counter()
    pers = np.transpose(sampled, (1, 2, 0))
    pers_img = Image.fromarray(pers)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")

    pers_path = osp.join(result_path, 'output_numpy_single.jpg')
    pers_img.save(pers_path)
