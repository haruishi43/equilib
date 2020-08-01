#!/usr/bin/env python3

import argparse
import os.path as osp

import time
import math
import matplotlib
import matplotlib.pyplot as plt

import cv2
import numpy as np
from PIL import Image

from pano2pers import Pano2Pers

matplotlib.use('Agg')


def rescale_frame(frame, percent=75):
    r"""Rescale image for imshow"""
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
    yaw = 0 * inc  # -pi < b < pi
    pitch = 0 * inc  # -pi/2 < a < pi/2
    roll = 0
    rot = {
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
    }
    h = 480  # 480
    w = 640  # 640
    fov = 90

    p2p = Pano2Pers.from_crop_size(h, w, fov, device=0, debug=True)
    p2p.set_rotation([
        rot['roll'], rot['pitch'], rot['yaw'],
    ])

    s = time.time()
    dst_img = p2p.get_perspective(src_img)
    e = time.time()
    print(e - s)
    dst_img = p2p.convert_rgb(dst_img)
    cv2.imshow("output", rescale_frame(dst_img, percent=100))
    cv2.waitKey()
    cv2.imwrite("./data/output.jpg", dst_img)


def test_video(path=None):
    if path is not None:
        video_path = path
    else:
        video_path = "./data/R0010050_er_0.MP4"

    pi = math.pi
    inc = pi / 180
    yaw = 0  # -pi < b < pi
    pitch = 0  # -pi/2 < a < pi/2
    roll = 0
    h = 480  #480
    w = 640  #640
    fov = 80

    # initialize Pano2Perspective
    p2p = Pano2Pers.from_crop_size(h, w, fov, device=0, debug=True)

    times = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        s = time.time()
        arr = [frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]]
        p2p.set_rotation([yaw, pitch, roll])  # set rotation
        dst_img = p2p.get_perspective(frame)  # process the image
        e = time.time()

        cv2.imshow("video", dst_img)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        if k == ord('w'):
            pitch -= inc
        if k == ord('s'):
            pitch += inc
        if k == ord('a'):
            yaw -= inc
        if k == ord('d'):
            yaw += inc

        times.append(e - s)

    cap.release()
    cv2.destroyAllWindows()

    print(sum(times)/len(times))
    x_axis = [i for i in range(len(times))]
    plt.plot(x_axis, times)
    plt.savefig('test_video.png')



if __name__ == "__main__":
    data_path = osp.join('.', 'data')
    pano_path = osp.join(data_path, '8081_earthmap4k.jpg')

    tic = time.perf_counter()
    pano_img = Image.open(pano_path)

    # Sometimes images are RGBA
    pano_img = pano_img.convert('RGB')
    pano = np.asarray(pano_img)

    pano = np.transpose(pano, (2, 0, 1))

    toc = time.perf_counter()
    print(f"Process Pano: {toc - tic:0.4f} seconds")

    _, h_pano, w_pano = pano.shape
    print('panorama size:')
    print(h_pano, w_pano)

    # Variables:
    h_pers = 480
    w_pers = 640
    rot = {
        'roll': 0,
        'pitch': 0,
        'yaw': 0,
    }
    fov_x = 90

    tic = time.perf_counter()
    m = utils.create_coord(h_pers, w_pers)
    K = utils.create_K(h_pers, w_pers, fov_x)
    R = utils.create_rot_mat(**rot)
    toc = time.perf_counter()
    print(f"Process m, K, R: {toc - tic:0.4f} seconds")

    # m = P M
    # P = K [R | t] = K R (in this case...)
    # R^-1 K^-1 m = M

    tic = time.perf_counter()
    K_inv = np.linalg.inv(K)
    R_inv = np.linalg.inv(R)
    m = m[:, :, :, np.newaxis]
    toc = time.perf_counter()
    print(f"Take Inverse: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    M = R_inv @ K_inv @ m
    M = M.squeeze(3)
    toc = time.perf_counter()
    print(f"M = R^-1 K^-1 m: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    phi, theta = utils.pixel_wise_rot(M)
    toc = time.perf_counter()
    print(f"pixel_wise_rot: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    ui = (theta - np.pi) * w_pano / (2 * np.pi)
    uj = (phi - np.pi / 2) * h_pano / np.pi

    ui = np.where(ui < 0, ui + w_pano, ui)
    ui = np.where(ui >= w_pano, ui - w_pano, ui)
    uj = np.where(uj < 0, uj + h_pano, uj)
    uj = np.where(uj >= h_pano, uj - h_pano, uj)

    # pano = pano / 255.  # scaling 0.0 - 1.0

    grid = np.stack((uj, ui), axis=0)
    toc = time.perf_counter()
    print(f"preprocess grid: {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    sampled = naive_sample(pano, grid, mode='bilinear')
    toc = time.perf_counter()
    print(f"naive: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    sampled = faster_sample(pano, grid, mode='bilinear')
    toc = time.perf_counter()
    print(f"faster: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    # after sample
    pers = np.transpose(sampled, (1,2,0))
    # pers = (pers * 255).astype(np.uint8)  # unscaling
    pers_img = Image.fromarray(pers)
    toc = time.perf_counter()
    print(f"post process: {toc - tic:0.4f} seconds")

    pers_path = osp.join(data_path, 'output_numpy.jpg')
    pers_img.save(pers_path)