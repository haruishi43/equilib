#!/usr/bin/env python3

import os.path as osp
import time

import numpy as np
from PIL import Image

from equilib.equi2cube import NumpyEqui2Cube


def run(equi, rot):
    h_equi, w_equi = equi.shape[-2:]
    print('equirectangular image size:')
    print(h_equi, w_equi)

    w_face = 256

    tic = time.perf_counter()
    equi2cube = NumpyEqui2Cube(
        w_face=w_face,
    )
    toc = time.perf_counter()
    print(f"Init Equi2Cube: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    sample = equi2cube(
        equi=equi,
        rot=rot,
        cube_format="dict",
        sampling_method="faster",
        mode="bilinear",
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
        'yaw': np.pi/4,
    }

    cube = run(equi, rot)

    tic = time.perf_counter()
    for k, c in cube.items():
        out = np.transpose(c, (1, 2, 0))
        out_img = Image.fromarray(out)
        out_path = osp.join(result_path, f'equi2cube_numpy_single_{k}.jpg')
        out_img.save(out_path)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")


# def test_numpy_batch():
#     data_path = osp.join('.', 'tests', 'data')
#     result_path = osp.join('.', 'tests', 'results')
#     equi_path = osp.join(data_path, 'test.jpg')
#     batch_size = 4

#     tic = time.perf_counter()
#     batched_equi = []
#     for _ in range(batch_size):
#         equi_img = Image.open(equi_path)
#         # Sometimes images are RGBA
#         equi_img = equi_img.convert('RGB')
#         equi = np.asarray(equi_img)
#         equi = np.transpose(equi, (2, 0, 1))
#         batched_equi.append(equi)
#     batched_equi = np.stack(batched_equi, axis=0)
#     toc = time.perf_counter()
#     print(f"Process Equirectangular Image: {toc - tic:0.4f} seconds")

#     batched_rot = []
#     inc = np.pi/8
#     for i in range(batch_size):
#         rot = {
#             'roll': 0,
#             'pitch': i * inc,
#             'yaw': 0,
#         }
#         batched_rot.append(rot)

#     batched_sample = run(batched_equi, batched_rot)

#     tic = time.perf_counter()
#     batched_out = []
#     for i in range(batch_size):
#         sample = batched_sample[i]
#         out = np.transpose(sample, (1, 2, 0))
#         out_img = Image.fromarray(out)
#         batched_out.append(out_img)
#     toc = time.perf_counter()

#     for i, out in enumerate(batched_out):
#         out_path = osp.join(result_path, f'equi2equi_numpy_batch_{i}.jpg')
#         out.save(out_path)
