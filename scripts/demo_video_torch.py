#!/usr/bin/env python3

import argparse

import time
import math
import matplotlib
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
from torchvision import transforms

from pano2pers_torch import (
    torch_original,
    torch_sample,
    utils,
)

matplotlib.use('Agg')


def rescale_frame(frame, percent=75):
    '''Rescale image for imshow'''
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--data', nargs='?', default=None, type=str)
    args = parser.parse_args()

    path = args.data
    if args.video:
        test_video(path)
    else:
        test_single_image(path)


def test_single_image(path=None):
    if path is not None:
        src_img = cv2.imread(path)
        src_img_show = rescale_frame(src_img, percent=10)
        cv2.imshow("input", src_img_show)
        cv2.waitKey()
    else:
        src_img = cv2.imread("./data/pano.jpg")

    pi = math.pi
    inc = pi / 36
    roll = 0  # -pi/2 < a < pi/2
    pitch = 0 * inc  # -pi < b < pi
    yaw = 0 * inc

    rot = {
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
    }
    h_pers = 480  # 480
    w_pers = 640  # 640
    fov_x = 90.

    device = torch.device('cuda')

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    pano = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    pano = to_tensor(pano)
    pano = pano.to(device)
    _, h_pano, w_pano = pano.shape

    s = time.time()

    # whole system
    m = utils.create_coord(h_pers, w_pers, device=device)
    K = utils.create_K(h_pers, w_pers, fov_x, device=device)
    R = utils.create_rot_mat(**rot, device=device)
    M = R.inverse() @ K.inverse() @ m.unsqueeze(3)
    M = M.squeeze(3)
    phi, theta = utils.pixel_wise_rot(M)
    ui = (theta - math.pi) * w_pano / (2 * math.pi)
    uj = (phi - math.pi / 2) * h_pano / math.pi
    ui = torch.where(ui < 0, ui + w_pano, ui)
    ui = torch.where(ui >= w_pano, ui - w_pano, ui)
    uj = torch.where(uj < 0, uj + h_pano, uj)
    uj = torch.where(uj >= h_pano, uj - h_pano, uj)
    grid = torch.stack((uj, ui), axis=-3)
    sampled = torch_sample(pano, grid, device=device, mode='bilinear')
    pers = np.asarray(sampled.cpu().numpy() * 255, dtype=np.uint8)
    pers = np.transpose(pers, (1, 2, 0))

    e = time.time()
    print(e - s)
    pers = cv2.cvtColor(pers, cv2.COLOR_RGB2BGR)
    # cv2.imshow("output", rescale_frame(pers, percent=100))
    # cv2.waitKey()
    cv2.imwrite("./data/output_numpy_single.jpg", pers)


def test_video(path=None):
    if path is not None:
        video_path = path
    else:
        video_path = "./data/R0010053_er_23.MP4"

    pi = math.pi
    inc = pi / 180
    roll = 0  # -pi/2 < a < pi/2
    pitch = 0  # -pi < b < pi
    yaw = 0

    device = torch.device('cuda')

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    h_pers = 480  # 480
    w_pers = 640  # 640
    fov_x = 80

    # initialize Pano2Perspective
    times = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()

        rot = {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
        }

        if not ret:
            break

        s = time.time()
        # whole system
        pano = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pano = to_tensor(pano)
        pano = pano.to(device)
        _, h_pano, w_pano = pano.shape
        m = utils.create_coord(h_pers, w_pers, device=device)
        K = utils.create_K(h_pers, w_pers, fov_x, device=device)
        R = utils.create_rot_mat(**rot, device=device)
        M = R.inverse() @ K.inverse() @ m.unsqueeze(3)
        M = M.squeeze(3)
        phi, theta = utils.pixel_wise_rot(M)
        ui = (theta - math.pi) * w_pano / (2 * math.pi)
        uj = (phi - math.pi / 2) * h_pano / math.pi
        ui = torch.where(ui < 0, ui + w_pano, ui)
        ui = torch.where(ui >= w_pano, ui - w_pano, ui)
        uj = torch.where(uj < 0, uj + h_pano, uj)
        uj = torch.where(uj >= h_pano, uj - h_pano, uj)
        grid = torch.stack((uj, ui), axis=-3)
        sampled = torch_sample(pano, grid, device=device, mode='bilinear')
        pers = np.asarray(sampled.cpu().numpy() * 255, dtype=np.uint8)
        pers = np.transpose(pers, (1, 2, 0))
        pers = cv2.cvtColor(pers, cv2.COLOR_RGB2BGR)
        e = time.time()

        # cv2.imshow("video", pers)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        if k == ord('w'):
            roll -= inc
        if k == ord('s'):
            roll += inc
        if k == ord('a'):
            pitch += inc
        if k == ord('d'):
            pitch -= inc

        times.append(e - s)

    cap.release()
    cv2.destroyAllWindows()

    print(sum(times)/len(times))
    x_axis = [i for i in range(len(times))]
    plt.plot(x_axis, times)
    plt.savefig('test_torch_video.png')


if __name__ == "__main__":
    main()
