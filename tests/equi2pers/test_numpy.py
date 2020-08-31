#!/usr/bin/env python3

import os.path as osp
import time

import numpy as np
from PIL import Image

from equilib.equi2pers import NumpyEqui2Pers

SAMPLING_METHOD = 'faster'
SAMPLING_MODE = 'bilinear'


def run(equi, rot):
    h_equi, w_equi = equi.shape[-2:]
    print('equirectangular image size:')
    print(h_equi, w_equi)

    # Variables:
    h_pers = 480
    w_pers = 640
    fov_x = 90

    tic = time.perf_counter()
    equi2pers = NumpyEqui2Pers(
        w_pers=w_pers,
        h_pers=h_pers,
        fov_x=fov_x
    )
    toc = time.perf_counter()
    print(f"Init Equi2Pers: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    sample = equi2pers(
        equi=equi,
        rot=rot,
        sampling_method=SAMPLING_METHOD,
        mode=SAMPLING_MODE,
    )
    toc = time.perf_counter()
    print(f"Sample: {toc - tic:0.4f} seconds")

    return sample


def test_numpy_single():
    data_path = osp.join('.', 'tests', 'data')
    result_path = osp.join('.', 'tests', 'results')
    equi_path = osp.join(data_path, 'test.jpg')

    tic = time.perf_counter()
    equi_img = Image.open(equi_path)
    # Sometimes images are RGBA
    equi_img = equi_img.convert('RGB')
    equi = np.asarray(equi_img)
    equi = np.transpose(equi, (2, 0, 1))
    toc = time.perf_counter()
    print(f"Process Equirectangular Image: {toc - tic:0.4f} seconds")

    rot = {
        'roll': 0,
        'pitch': 0,
        'yaw': 0,
    }

    sample = run(equi, rot)

    tic = time.perf_counter()
    pers = np.transpose(sample, (1, 2, 0))
    pers_img = Image.fromarray(pers)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")

    pers_path = osp.join(result_path, 'equi2pers_numpy_single.jpg')
    pers_img.save(pers_path)


def test_numpy_batch():
    data_path = osp.join('.', 'tests', 'data')
    result_path = osp.join('.', 'tests', 'results')
    equi_path = osp.join(data_path, 'test.jpg')
    batch_size = 4

    tic = time.perf_counter()
    batched_equi = []
    for _ in range(batch_size):
        equi_img = Image.open(equi_path)
        # Sometimes images are RGBA
        equi_img = equi_img.convert('RGB')
        equi = np.asarray(equi_img)
        equi = np.transpose(equi, (2, 0, 1))
        batched_equi.append(equi)
    batched_equi = np.stack(batched_equi, axis=0)
    toc = time.perf_counter()
    print(f"Process Equirectangular Image: {toc - tic:0.4f} seconds")

    batched_rot = []
    inc = np.pi/8
    for i in range(batch_size):
        rot = {
            'roll': 0,
            'pitch': i * inc,
            'yaw': 0,
        }
        batched_rot.append(rot)

    batched_sample = run(batched_equi, batched_rot)

    tic = time.perf_counter()
    batched_pers = []
    for i in range(batch_size):
        sample = batched_sample[i]
        pers = np.transpose(sample, (1, 2, 0))
        pers_img = Image.fromarray(pers)
        batched_pers.append(pers_img)
    toc = time.perf_counter()

    for i, pers in enumerate(batched_pers):
        pers_path = osp.join(result_path, f'equi2pers_numpy_batch_{i}.jpg')
        pers.save(pers_path)
