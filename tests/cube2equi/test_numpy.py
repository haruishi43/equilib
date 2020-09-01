#!/usr/bin/env python3

import time
import os.path as osp

import numpy as np

from PIL import Image

from equilib.cube2equi import NumpyCube2Equi

OUT_W = 480
OUT_H = 240


def run(cube, cube_format):
    print("Input is {}:".format(cube_format))
    if isinstance(cube, list):
        c = cube[0]  # get the first of the batch
        if isinstance(c, list):
            assert cube_format == "list"
            print("one: {}".format(c[0].shape))
        elif isinstance(c, dict):
            assert cube_format == "dict"
            print("one: {}".format(c["F"].shape))
        elif isinstance(c, np.ndarray):
            assert cube_format in ["horizon", "dice", "list"]
            # can be single list
            if cube_format == "list":
                print("one: {}".format(c.shape))
            else:
                print("one: {}".format(c[0].shape))
    elif isinstance(cube, dict):
        assert cube_format == "dict"
        print("one: {}".format(cube["F"].shape))
    else:
        assert cube_format in ["horizon", "dice"]
        print("one: {}".format(cube.shape))

    tic = time.perf_counter()
    cube2equi = NumpyCube2Equi(
        w_out=OUT_W,
        h_out=OUT_H,
    )
    toc = time.perf_counter()
    print("Init Cube2Equi: {:0.4f} seconds".format(toc - tic))

    tic = time.perf_counter()
    samples = cube2equi(
        cubemap=cube,
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

    cube_format = "dict"

    tic = time.perf_counter()
    if cube_format in ["horizon", "dice"]:
        img_path = osp.join(data_path, "test_{}.jpg".format(cube_format))
        cube = Image.open(img_path)
        cube = cube.convert("RGB")
        cube = np.asarray(cube)
        cube = np.transpose(cube, (2, 0, 1))
    elif cube_format in ["dict", "list"]:
        img_paths = osp.join(data_path, "test_dict_{k}.jpg")
        cube = {}
        for k in ["F", "R", "B", "L", "U", "D"]:
            face = Image.open(img_paths.format(cube_format=cube_format, k=k))
            face = face.convert("RGB")
            face = np.asarray(face)
            face = np.transpose(face, (2, 0, 1))
            cube[k] = face
        if cube_format == "list":
            cube = list(cube.values())
    else:
        raise ValueError
    toc = time.perf_counter()
    print("Process Cube Image: {:0.4f} seconds".format(toc - tic))

    equi = run(cube, cube_format=cube_format)

    tic = time.perf_counter()
    equi = np.transpose(equi, (1, 2, 0))
    equi_img = Image.fromarray(equi)
    out_path = osp.join(
        result_path, "cube2equi_numpy_single_{}.jpg".format(cube_format)
    )
    equi_img.save(out_path)
    toc = time.perf_counter()
    print("post process: {:0.4f} seconds".format(toc - tic))


def test_numpy_batch():
    data_path = osp.join(".", "tests", "data")
    result_path = osp.join(".", "tests", "results")
    batch_size = 4
    cube_format = "dict"

    tic = time.perf_counter()
    batched_cube = []
    for _ in range(batch_size):
        if cube_format in ["horizon", "dice"]:
            img_path = osp.join(data_path, "test_{}.jpg".format(cube_format))
            cube = Image.open(img_path)
            cube = cube.convert("RGB")
            cube = np.asarray(cube)
            cube = np.transpose(cube, (2, 0, 1))
        elif cube_format in ["dict", "list"]:
            img_paths = osp.join(data_path, "test_dict_{k}.jpg")
            cube = {}
            for k in ["F", "R", "B", "L", "U", "D"]:
                face = Image.open(
                    img_paths.format(cube_format=cube_format, k=k)
                )
                face = face.convert("RGB")
                face = np.asarray(face)
                face = np.transpose(face, (2, 0, 1))
                cube[k] = face
            if cube_format == "list":
                cube = list(cube.values())
        else:
            raise ValueError

        batched_cube.append(cube)
    toc = time.perf_counter()
    print("Process Cube Image: {:0.4f} seconds".format(toc - tic))

    batched_equi = run(batched_cube, cube_format=cube_format)

    tic = time.perf_counter()
    for i, equi in enumerate(batched_equi):
        equi = np.transpose(equi, (1, 2, 0))
        equi_img = Image.fromarray(equi)
        out_path = osp.join(
            result_path,
            "cube2equi_numpy_batched_{}_{}.jpg".format(cube_format, i),
        )
        equi_img.save(out_path)
    toc = time.perf_counter()
    print("post process: {:0.4f} seconds".format(toc - tic))
