#!/usr/bin/env python3

import os.path as osp

import time
from copy import deepcopy

from PIL import Image
import math
import torch

from torchvision import transforms

from pano2pers_torch import (
    torch_sample,
    torch_original,
    utils,
)


if __name__ == "__main__":
    data_path = osp.join('.', 'data')
    pano_path = osp.join(data_path, '8081_earthmap4k.jpg')

    # Variables:
    h_pers = 480
    w_pers = 640
    fov_x = 90.
    batch_size = 16

    device = torch.device('cuda')

    # Transforms
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    to_PIL = transforms.Compose([
        transforms.ToPILImage(),
    ])

    def get_batch_img(imgs):
        return torch.stack(imgs, dim=0)

    tic = time.perf_counter()
    pano_img = Image.open(pano_path)
    # NOTE: Sometimes images are RGBA
    pano_img = pano_img.convert('RGB')
    imgs = []
    for i in range(batch_size):
        pano = to_tensor(pano_img)
        imgs.append(deepcopy(pano))
    panos = get_batch_img(imgs)
    panos = panos.to(device)
    toc = time.perf_counter()
    print(f"Process Pano: {toc - tic:0.4f} seconds")
    print("Pano: ", utils.sizeof(panos)/10e6, "mb")

    _, h_pano, w_pano = pano.shape
    print('panorama size:')
    print(h_pano, w_pano)

    def get_rots(batch_size):
        rots = []
        for i in range(batch_size):
            rot = {
                'roll': 0.,
                'pitch': i * (360/batch_size),
                'yaw': 0.,
            }
            rots.append(rot)
        return rots

    def get_batch_coord(coord, batch_size):
        coords = []
        for i in range(batch_size):
            coords.append(coord.clone())
        return torch.stack(coords, dim=0)

    tic = time.perf_counter()
    rot_coords = []
    for rot in get_rots(batch_size):
        rot_coord = utils.create_rot_coord(
            height=h_pers, width=w_pers,
            fov_x=fov_x, rot=rot,
            device=device)
        rot_coords.append(rot_coord)
    rot_coords = torch.stack(rot_coords, dim=0)
    toc = time.perf_counter()
    print(f"Create rot_coord: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    phi, theta = utils.pixel_wise_rot(rot_coords)
    toc = time.perf_counter()
    print(f"pixel_wise_rot: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    ui = (theta - math.pi) * w_pano / (2 * math.pi)
    uj = (phi - math.pi / 2) * h_pano / math.pi

    ui = torch.where(ui < 0, ui + w_pano, ui)
    ui = torch.where(ui >= w_pano, ui - w_pano, ui)
    uj = torch.where(uj < 0, uj + h_pano, uj)
    uj = torch.where(uj >= h_pano, uj - h_pano, uj)

    grid = torch.stack((uj, ui), axis=-3)  # 3rd to last
    print("grid:", grid.shape)
    toc = time.perf_counter()
    print(f"preprocess grid: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    pers = torch_sample(panos, grid, device=device, mode='bilinear')
    toc = time.perf_counter()
    print(f"torch mine: {toc - tic:0.4f} seconds")

    pers = pers[5]
    tic = time.perf_counter()
    pers = pers.to('cpu')
    pers_img = to_PIL(pers)
    pers_path = osp.join(data_path, 'output_torch_batch_1.jpg')
    pers_img.save(pers_path)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    pers = torch_original(panos, grid, device=device, mode='bilinear')
    toc = time.perf_counter()
    print(f"torch original: {toc - tic:0.4f} seconds")

    pers = pers[5]
    tic = time.perf_counter()
    pers = pers.to('cpu')
    pers_img = to_PIL(pers)
    pers_path = osp.join(data_path, 'output_torch_batch_2.jpg')
    pers_img.save(pers_path)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")
