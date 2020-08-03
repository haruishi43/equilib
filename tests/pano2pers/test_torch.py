#!/usr/bin/env python3

import os.path as osp

import time

import copy
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from panolib.pano2pers import TorchPano2Pers


def run(pano, rot):
    h_pano, w_pano = pano.shape[-2:]
    print('panorama size:')
    print(h_pano, w_pano)

    # Variables:
    h_pers = 480
    w_pers = 640
    fov_x = 90

    tic = time.perf_counter()
    pano2pers = TorchPano2Pers(
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
        sampling_method="torch",
        mode="bilinear",
    )
    toc = time.perf_counter()
    print(f"Sample: {toc - tic:0.4f} seconds")

    return sample


def test_torch_single():
    data_path = osp.join('.', 'tests', 'data')
    result_path = osp.join('.', 'tests', 'results')
    pano_path = osp.join(data_path, 'test.jpg')
    device = torch.device('cuda')

    # Transforms
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    to_PIL = transforms.Compose([
        transforms.ToPILImage(),
    ])

    tic = time.perf_counter()
    pano_img = Image.open(pano_path)
    # NOTE: Sometimes images are RGBA
    pano_img = pano_img.convert('RGB')
    pano = to_tensor(pano_img)
    pano = pano.to(device)
    toc = time.perf_counter()
    print(f"Process Pano: {toc - tic:0.4f} seconds")

    rot = {
        'roll': 0.,
        'pitch': 0.,
        'yaw': 0.,
    }

    sample = run(pano, rot)

    tic = time.perf_counter()
    pers = sample.to('cpu')
    pers_img = to_PIL(pers)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")

    pers_path = osp.join(result_path, 'output_torch_single.jpg')
    pers_img.save(pers_path)


def test_torch_batch():
    data_path = osp.join('.', 'tests', 'data')
    result_path = osp.join('.', 'tests', 'results')
    pano_path = osp.join(data_path, 'test.jpg')
    device = torch.device('cuda')
    batch_size = 16

    # Transforms
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    to_PIL = transforms.Compose([
        transforms.ToPILImage(),
    ])

    tic = time.perf_counter()
    pano_img = Image.open(pano_path)
    # NOTE: Sometimes images are RGBA
    pano_img = pano_img.convert('RGB')
    batched_pano = []
    for i in range(batch_size):
        pano = to_tensor(pano_img)
        batched_pano.append(copy.deepcopy(pano))
    batched_pano = torch.stack(batched_pano, dim=0)
    batched_pano = batched_pano.to(device)
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
        sample = copy.deepcopy(batched_sample[i])
        sample = sample.to('cpu')
        pers_img = to_PIL(sample)
        batched_pers.append(pers_img)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")

    for i, pers in enumerate(batched_pers):
        pers_path = osp.join(result_path, f'output_torch_batch_{i}.jpg')
        pers.save(pers_path)
