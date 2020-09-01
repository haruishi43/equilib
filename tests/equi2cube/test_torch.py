#!/usr/bin/env python3

import time
import os.path as osp

import numpy as np

import torch

from PIL import Image

from torchvision import transforms

from equilib.equi2cube import TorchEqui2Cube

WIDTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(equi, rot, cube_format):
    h_equi, w_equi = equi.shape[-2:]
    print("equirectangular image size:")
    print(h_equi, w_equi)

    w_face = WIDTH

    tic = time.perf_counter()
    equi2cube = TorchEqui2Cube(
        w_face=w_face,
    )
    toc = time.perf_counter()
    print("Init Equi2Cube: {:0.4f} seconds".format(toc - tic))

    tic = time.perf_counter()
    samples = equi2cube(
        equi=equi,
        rot=rot,
        cube_format=cube_format,
        sampling_method="torch",
        mode="bilinear",
    )
    toc = time.perf_counter()
    print("Sample: {:0.4f} seconds".format(toc - tic))

    return samples


def test_torch_single():
    data_path = osp.join(".", "tests", "data")
    result_path = osp.join(".", "tests", "results")
    equi_path = osp.join(data_path, "test.jpg")
    device = DEVICE

    # Transforms
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    to_PIL = transforms.Compose(
        [
            transforms.ToPILImage(),
        ]
    )

    tic = time.perf_counter()
    equi_img = Image.open(equi_path)
    # Sometimes images are RGBA
    equi_img = equi_img.convert("RGB")
    equi = to_tensor(equi_img)
    equi = equi.to(device)
    toc = time.perf_counter()
    print("Process Equirectangular Image: {:0.4f} seconds".format(toc - tic))

    rot = {
        "roll": 0,
        "pitch": 0,
        "yaw": 0,
    }

    cube_format = "dice"

    cube = run(equi, rot, cube_format=cube_format)

    tic = time.perf_counter()
    if cube_format == "dict":
        print("output: dict")
        for k, c in cube.items():
            out = c.to("cpu")
            out_img = to_PIL(out)
            out_path = osp.join(
                result_path, "equi2cube_torch_single_dict_{}.jpg".format(k)
            )
            out_img.save(out_path)
    elif cube_format == "list":
        print("output: list")
        for k, c in zip(["F", "R", "B", "L", "U", "D"], cube):
            out = c.to("cpu")
            out_img = to_PIL(out)
            out_path = osp.join(
                result_path, "equi2cube_torch_single_list_{}.jpg".format(k)
            )
            out_img.save(out_path)
    elif cube_format in ["horizon", "dice"]:
        print("output: {}".format(cube_format))
        out = cube.to("cpu")
        out_img = to_PIL(out)
        out_path = osp.join(
            result_path, "equi2cube_torch_single_{}.jpg".format(cube_format)
        )
        out_img.save(out_path)
    toc = time.perf_counter()
    print("post process: {:0.4f} seconds".format(toc - tic))


def test_torch_batch():
    data_path = osp.join(".", "tests", "data")
    result_path = osp.join(".", "tests", "results")
    equi_path = osp.join(data_path, "test.jpg")
    batch_size = 4
    device = DEVICE

    # Transforms
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    to_PIL = transforms.Compose(
        [
            transforms.ToPILImage(),
        ]
    )

    tic = time.perf_counter()
    batched_equi = []
    for _ in range(batch_size):
        equi_img = Image.open(equi_path)
        # Sometimes images are RGBA
        equi = to_tensor(equi_img)
        batched_equi.append(equi)
    batched_equi = torch.stack(batched_equi, axis=0)
    batched_equi = batched_equi.to(device)
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

    cube_format = "dict"

    batched_cubes = run(batched_equi, batched_rot, cube_format=cube_format)

    tic = time.perf_counter()
    if cube_format == "dict":
        print("output: dict")
        for i in range(batch_size):
            for k, c in batched_cubes[i].items():
                out = c.to("cpu")
                out_img = to_PIL(out)
                out_path = osp.join(
                    result_path,
                    "equi2cube_torch_batched_dict_{}_{}.jpg".format(i, k),
                )
                out_img.save(out_path)
    elif cube_format == "list":
        print("output: list")
        for i in range(batch_size):
            for k, c in zip(["F", "R", "B", "L", "U", "D"], batched_cubes[i]):
                out = c.to("cpu")
                out_img = to_PIL(out)
                out_path = osp.join(
                    result_path,
                    "equi2cube_torch_batched_list_{}_{}.jpg".format(i, k),
                )
                out_img.save(out_path)
    elif cube_format in ["horizon", "dice"]:
        print("output: {}".format(cube_format))
        for i in range(batch_size):
            out = batched_cubes[i].to("cpu")
            out_img = to_PIL(out)
            out_path = osp.join(
                result_path,
                "equi2cube_torch_batched_{}_{}.jpg".format(cube_format, i),
            )
            out_img.save(out_path)
    toc = time.perf_counter()
    print("post process: {:0.4f} seconds".format(toc - tic))
