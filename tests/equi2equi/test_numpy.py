#!/usr/bin/env python3

import time
import os.path as osp

import numpy as np

from PIL import Image

from equilib.equi2equi import NumpyEqui2Equi

WIDTH = 640
HEIGHT = 320


def run(equi, rot):
    h_equi, w_equi = equi.shape[-2:]
    print("equirectangular image size:")
    print(h_equi, w_equi)

    tic = time.perf_counter()
    equi2equi = NumpyEqui2Equi(
        w_out=WIDTH,
        h_out=HEIGHT,
    )
    toc = time.perf_counter()
    print("Init Equi2Equi: {:0.4f} seconds".format(toc - tic))

    tic = time.perf_counter()
    sample = equi2equi(
        src=equi,
        rot=rot,
        sampling_method="faster",
        mode="bilinear",
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
    out = np.transpose(sample, (1, 2, 0))
    out_img = Image.fromarray(out)
    toc = time.perf_counter()
    print("post process: {:0.4f} seconds".format(toc - tic))

    out_path = osp.join(result_path, "equi2equi_numpy_single.jpg")
    out_img.save(out_path)


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
    batched_out = []
    for i in range(batch_size):
        sample = batched_sample[i]
        out = np.transpose(sample, (1, 2, 0))
        out_img = Image.fromarray(out)
        batched_out.append(out_img)
    toc = time.perf_counter()

    for i, out in enumerate(batched_out):
        out_path = osp.join(
            result_path, "equi2equi_numpy_batch_{}.jpg".format(i)
        )
        out.save(out_path)
