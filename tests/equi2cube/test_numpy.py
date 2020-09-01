#!/usr/bin/env python3

import time
import os.path as osp

import numpy as np

from PIL import Image

from equilib.equi2cube import NumpyEqui2Cube

WIDTH = 256


def run(equi, rot, cube_format):
    h_equi, w_equi = equi.shape[-2:]
    print("equirectangular image size:")
    print(h_equi, w_equi)

    w_face = WIDTH

    tic = time.perf_counter()
    equi2cube = NumpyEqui2Cube(
        w_face=w_face,
    )
    toc = time.perf_counter()
    print("Init Equi2Cube: {:0.4f} seconds".format(toc - tic))

    tic = time.perf_counter()
    samples = equi2cube(
        equi=equi,
        rot=rot,
        cube_format=cube_format,
        sampling_method="faster",
        mode="bilinear",
    )
    toc = time.perf_counter()
    print("Sample: {:0.4f} seconds".format(toc - tic))

    return samples


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
        "roll": np.pi / 4,
        "pitch": 0,
        "yaw": 0,
    }

    cube_format = "dice"

    cube = run(equi, rot, cube_format=cube_format)

    tic = time.perf_counter()
    if cube_format == "dict":
        print("output: dict")
        for k, c in cube.items():
            out = np.transpose(c, (1, 2, 0))
            out_img = Image.fromarray(out)
            out_path = osp.join(
                result_path, "equi2cube_numpy_single_dict_{}.jpg".format(k)
            )
            out_img.save(out_path)
    elif cube_format == "list":
        print("output: list")
        for k, c in zip(["F", "R", "B", "L", "U", "D"], cube):
            out = np.transpose(c, (1, 2, 0))
            out_img = Image.fromarray(out)
            out_path = osp.join(
                result_path, "equi2cube_numpy_single_list_{}.jpg".format(k)
            )
            out_img.save(out_path)
    elif cube_format in ["horizon", "dice"]:
        print("output: {}".format(cube_format))
        out = np.transpose(cube, (1, 2, 0))
        out_img = Image.fromarray(out)
        out_path = osp.join(
            result_path, "equi2cube_numpy_single_{}.jpg".format(cube_format)
        )
        out_img.save(out_path)
    toc = time.perf_counter()
    print("post process: {:0.4f} seconds".format(toc - tic))


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

    cube_format = "dice"

    batched_cubes = run(batched_equi, batched_rot, cube_format=cube_format)

    tic = time.perf_counter()
    if cube_format == "dict":
        print("output: dict")
        for i in range(batch_size):
            for k, c in batched_cubes[i].items():
                out = np.transpose(c, (1, 2, 0))
                out_img = Image.fromarray(out)
                out_path = osp.join(
                    result_path,
                    "equi2cube_numpy_batched_dict_{}_{}.jpg".format(i, k),
                )
                out_img.save(out_path)
    elif cube_format == "list":
        print("output: list")
        for i in range(batch_size):
            for k, c in zip(["F", "R", "B", "L", "U", "D"], batched_cubes[i]):
                out = np.transpose(c, (1, 2, 0))
                out_img = Image.fromarray(out)
                out_path = osp.join(
                    result_path,
                    "equi2cube_numpy_batched_list_{}_{}.jpg".format(i, k),
                )
                out_img.save(out_path)
    elif cube_format in ["horizon", "dice"]:
        print("output: {}".format(cube_format))
        for i in range(batch_size):
            out = np.transpose(batched_cubes[i], (1, 2, 0))
            out_img = Image.fromarray(out)
            out_path = osp.join(
                result_path,
                "equi2cube_numpy_batched_{}_{}.jpg".format(cube_format, i),
            )
            out_img.save(out_path)
    toc = time.perf_counter()
    print("post process: {:0.4f} seconds".format(toc - tic))
