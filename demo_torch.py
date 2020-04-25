#!/usr/bin/env python3

import os
import os.path as osp

import time

from PIL import Image
import math
import torch

from torchvision import transforms

from pano2pers_torch import (
    torch_sample,
    utils,
)


if __name__ == "__main__":
    data_path = osp.join('.', 'data')
    pano_path = osp.join(data_path, 'pano2.png')

    # Variables:
    h_pers = 480
    w_pers = 640
    rot = [45, 0, 0]
    fov_x = 90

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
    #NOTE: Sometimes images are RGBA
    pano_img = pano_img.convert('RGB')
    pano = to_tensor(pano_img)
    pano = pano.to(device)
    toc = time.perf_counter()
    print(f"Process Pano: {toc - tic:0.4f} seconds")

    _, h_pano, w_pano = pano.shape
    print('panorama size:')
    print(h_pano, w_pano)
    
    tic = time.perf_counter()
    coord = utils.create_coord(h_pers, w_pers, device)
    K = utils.create_K(h_pers, w_pers, fov_x, device)
    R = utils.create_rot_mat(rot, device)
    toc = time.perf_counter()
    print(f"Process coord, K, R: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    rot_coord = R.inverse() @ K.inverse() @ coord.unsqueeze(3)
    rot_coord = rot_coord.squeeze(3)
    toc = time.perf_counter()
    print(f"Take Inverse and rot_coord: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    a, b = utils.pixel_wise_rot(rot_coord)
    toc = time.perf_counter()
    print(f"pixel_wise_rot: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    ui = (a + math.pi) * w_pano / (2 * math.pi)
    uj = (b + math.pi / 2) * h_pano / math.pi

    ui = torch.where(ui < 0, ui + w_pano, ui)
    ui = torch.where(ui >= w_pano, ui - w_pano, ui)
    uj = torch.where(uj < 0, uj + h_pano, uj)
    uj = torch.where(uj >= h_pano, uj - h_pano, uj)

    grid = torch.stack((uj, ui), axis=0)
    toc = time.perf_counter()
    print(f"preprocess grid: {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    # pano = pano.unsqueeze(0)
    # grid = grid.unsqueeze(0)
    print(pano.shape)
    print(grid.shape)
    pers = torch_sample(pano, grid, device=device, mode='bilinear')
    toc = time.perf_counter()
    print(f"torch: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    pers = pers.to('cpu')
    pers_img = to_PIL(pers)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")

    pers_path = osp.join(data_path, 'output_torch.jpg')
    pers_img.save(pers_path)