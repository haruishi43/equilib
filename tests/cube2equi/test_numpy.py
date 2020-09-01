#!/usr/bin/env python3

import os.path as osp
import time

import numpy as np
from PIL import Image

from equilib.cube2equi import NumpyCube2Equi

OUT_W = 4000
OUT_H = 2000


def run(cube, cube_format):
    print(f"Input is {cube_format}:")
    if isinstance(cube, list):
        c = cube[0]  # get the first of the batch
        if isinstance(c, list):
            assert cube_format == 'list'
            print(f'one: {c[0].shape}')
        elif isinstance(c, dict):
            assert cube_format == 'dict'
            print(f"one: {c['F'].shape}")
        elif isinstance(c, np.ndarray):
            assert cube_format in ['horizon', 'dice', 'list']
            # can be single list
            if cube_format == 'list':
                print(f'one: {c.shape}')
            else:
                print(f'one: {c[0].shape}')
    elif isinstance(cube, dict):
        assert cube_format == 'dict'
        print(f"one: {cube['F'].shape}")
    else:
        assert cube_format in ['horizon', 'dice']
        print(f'one: {cube.shape}')

    tic = time.perf_counter()
    cube2equi = NumpyCube2Equi(
        w_out=OUT_W,
        h_out=OUT_H,
    )
    toc = time.perf_counter()
    print(f"Init Cube2Equi: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    samples = cube2equi(
        cubemap=cube,
        cube_format=cube_format,
        sampling_method="faster",
        mode="bilinear",
    )
    toc = time.perf_counter()
    print(f"Sample: {toc - tic:0.4f} seconds")

    return samples


def test_numpy_single():
    data_path = osp.join('.', 'tests', 'data')
    result_path = osp.join('.', 'tests', 'results')

    cube_format = "dict"

    tic = time.perf_counter()
    if cube_format in ['horizon', 'dice']:
        img_path = osp.join(data_path, f'test_{cube_format}.jpg')
        cube = Image.open(img_path)
        cube = cube.convert('RGB')
        cube = np.asarray(cube)
        cube = np.transpose(cube, (2, 0, 1))
    elif cube_format in ['dict', 'list']:
        img_paths = osp.join(data_path, "test_dict_{k}.jpg")
        cube = {}
        for k in ['F', 'R', 'B', 'L', 'U', 'D']:
            face = Image.open(
                img_paths.format(cube_format=cube_format, k=k)
            )
            face = face.convert('RGB')
            face = np.asarray(face)
            face = np.transpose(face, (2, 0, 1))
            cube[k] = face
        if cube_format == 'list':
            cube = list(cube.values())
    else:
        raise ValueError
    toc = time.perf_counter()
    print(f"Process Cube Image: {toc - tic:0.4f} seconds")

    equi = run(cube, cube_format=cube_format)

    tic = time.perf_counter()
    equi = np.transpose(equi, (1, 2, 0))
    equi_img = Image.fromarray(equi)
    out_path = osp.join(
        result_path,
        f'cube2equi_numpy_single_{cube_format}.jpg'
    )
    equi_img.save(out_path)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")


def test_numpy_batch():
    data_path = osp.join('.', 'tests', 'data')
    result_path = osp.join('.', 'tests', 'results')
    batch_size = 4
    cube_format = "dict"

    tic = time.perf_counter()
    batched_cube = []
    for _ in range(batch_size):
        if cube_format in ['horizon', 'dice']:
            img_path = osp.join(data_path, f'test_{cube_format}.jpg')
            cube = Image.open(img_path)
            cube = cube.convert('RGB')
            cube = np.asarray(cube)
            cube = np.transpose(cube, (2, 0, 1))
        elif cube_format in ['dict', 'list']:
            img_paths = osp.join(data_path, "test_dict_{k}.jpg")
            cube = {}
            for k in ['F', 'R', 'B', 'L', 'U', 'D']:
                face = Image.open(
                    img_paths.format(cube_format=cube_format, k=k)
                )
                face = face.convert('RGB')
                face = np.asarray(face)
                face = np.transpose(face, (2, 0, 1))
                cube[k] = face
            if cube_format == 'list':
                cube = list(cube.values())
        else:
            raise ValueError

        batched_cube.append(cube)
    toc = time.perf_counter()
    print(f"Process Cube Image: {toc - tic:0.4f} seconds")

    batched_equi = run(batched_cube, cube_format=cube_format)

    tic = time.perf_counter()
    for i, equi in enumerate(batched_equi):
        equi = np.transpose(equi, (1, 2, 0))
        equi_img = Image.fromarray(equi)
        out_path = osp.join(
            result_path,
            f'cube2equi_numpy_batched_{cube_format}_{i}.jpg'
        )
        equi_img.save(out_path)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")
