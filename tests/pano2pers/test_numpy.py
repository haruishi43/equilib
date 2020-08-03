#!/usr/bin/env python3

import os.path as osp
import time

import numpy as np
from PIL import Image

from panolib.pano2pers import NumpyPano2Pers


def run(pano, rot):
    h_pano, w_pano = pano.shape[-2:]
    print('panorama size:')
    print(h_pano, w_pano)

    # Variables:
    h_pers = 480
    w_pers = 640
    fov_x = 90

    tic = time.perf_counter()
    pano2pers = NumpyPano2Pers(
        w_pers=w_pers,
        h_pers=h_pers,
        fov_x=fov_x
    )
    toc = time.perf_counter()
    print(f"Init Pano2Pers: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    sample = pano2pers(
        pano=pano,
        rot=rot,
        sampling_method="faster",
        mode="bilinear",
    )
    toc = time.perf_counter()
    print(f"Sample: {toc - tic:0.4f} seconds")

    return sample


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

    rot = {
        'roll': 0,
        'pitch': 0,
        'yaw': 0,
    }

    sample = run(pano, rot)

    tic = time.perf_counter()
    pers = np.transpose(sample, (1, 2, 0))
    pers_img = Image.fromarray(pers)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")

    pers_path = osp.join(result_path, 'output_numpy_single.jpg')
    pers_img.save(pers_path)


def test_numpy_batch():
    data_path = osp.join('.', 'tests', 'data')
    result_path = osp.join('.', 'tests', 'results')
    pano_path = osp.join(data_path, 'test.jpg')
    batch_size = 4

    tic = time.perf_counter()
    batched_pano = []
    for _ in range(batch_size):
        pano_img = Image.open(pano_path)
        # Sometimes images are RGBA
        pano_img = pano_img.convert('RGB')
        pano = np.asarray(pano_img)
        pano = np.transpose(pano, (2, 0, 1))
        batched_pano.append(pano)
    batched_pano = np.stack(batched_pano, axis=0)
    toc = time.perf_counter()
    print(f"Process Pano: {toc - tic:0.4f} seconds")

    batched_rot = []
    inc = np.pi/8
    for i in range(batch_size):
        rot = {
            'roll': 0,
            'pitch': i * inc,
            'yaw': 0,
        }
        batched_rot.append(rot)

    batched_sample = run(batched_pano, batched_rot)

    tic = time.perf_counter()
    batched_pers = []
    for i in range(batch_size):
        sample = batched_sample[i]
        pers = np.transpose(sample, (1, 2, 0))
        pers_img = Image.fromarray(pers)
        batched_pers.append(pers_img)
    toc = time.perf_counter()

    for i, pers in enumerate(batched_pers):
        pers_path = osp.join(result_path, f'output_numpy_batch_{i}.jpg')
        pers.save(pers_path)
