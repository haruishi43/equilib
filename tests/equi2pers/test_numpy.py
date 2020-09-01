#!/usr/bin/env python3

import time
import os.path as osp

import numpy as np

from PIL import Image

from equilib.equi2pers import NumpyEqui2Pers

SAMPLING_METHOD = "faster"
SAMPLING_MODE = "bilinear"

WIDTH = 640
HEIGHT = 480
FOV = 90


def run(equi, rot):
    h_equi, w_equi = equi.shape[-2:]
    print("equirectangular image size:")
    print(h_equi, w_equi)

    tic = time.perf_counter()
    equi2pers = NumpyEqui2Pers(
        w_pers=WIDTH,
        h_pers=HEIGHT,
        fov_x=FOV,
    )
    toc = time.perf_counter()
    print("Init Equi2Pers: {:0.4f} seconds".format(toc - tic))

    tic = time.perf_counter()
    sample = equi2pers(
        equi=equi,
        rot=rot,
        sampling_method=SAMPLING_METHOD,
        mode=SAMPLING_MODE,
    )
    toc = time.perf_counter()
    print("Sample: {:0.4f} seconds".format(toc - tic))

    return sample


def test_numpy_single():
    data_path = osp.join(".", "tests", "data")
    result_path = osp.join(".", "tests", "results")
    equi_path = osp.join(data_path, "test.jpg")

    tic = time.perf_counter()
    equi_img = Image.open(equi_path)
    # Sometimes images are RGBA
    equi_img = equi_img.convert("RGB")
    equi = np.asarray(equi_img)
    equi = np.transpose(equi, (2, 0, 1))
    toc = time.perf_counter()
    print("Process Equirectangular Image: {:0.4f} seconds".format(toc - tic))

    rot = {
        "roll": 0,
        "pitch": 0,
        "yaw": 0,
    }

    sample = run(equi, rot)

    tic = time.perf_counter()
    pers = np.transpose(sample, (1, 2, 0))
    pers_img = Image.fromarray(pers)
    toc = time.perf_counter()
    print("post process: {:0.4f} seconds".format(toc - tic))

    pers_path = osp.join(result_path, "equi2pers_numpy_single.jpg")
    pers_img.save(pers_path)


def test_numpy_batch():
    data_path = osp.join(".", "tests", "data")
    result_path = osp.join(".", "tests", "results")
    equi_path = osp.join(data_path, "test.jpg")
    batch_size = 4

    tic = time.perf_counter()
    batched_equi = []
    for _ in range(batch_size):
        equi_img = Image.open(equi_path)
        # Sometimes images are RGBA
        equi_img = equi_img.convert("RGB")
        equi = np.asarray(equi_img)
        equi = np.transpose(equi, (2, 0, 1))
        batched_equi.append(equi)
    batched_equi = np.stack(batched_equi, axis=0)
    toc = time.perf_counter()
    print("Process Equirectangular Image: {:0.4f} seconds".format(toc - tic))

    batched_rot = []
    inc = np.pi / 8
    for i in range(batch_size):
        rot = {
            "roll": 0,
            "pitch": i * inc,
            "yaw": 0,
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
        pers_path = osp.join(
            result_path, "equi2pers_numpy_batch_{}.jpg".format(i)
        )
        pers.save(pers_path)
