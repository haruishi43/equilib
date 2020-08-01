#!/usr/bin/env python3

import argparse
import os.path as osp

import time
import math
import matplotlib
import matplotlib.pyplot as plt

import cv2
import numpy as np

from pano2perspective import Pano2Perspective as Pano2Pers

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
        frame = cv2.imread("./data/pano.jpg")

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

    p2p = Pano2Pers(w, h, fov)
    p2p.set_rotation([
        rot['roll'], rot['pitch'], rot['yaw'],
    ])

    s = time.time()
    frame = [frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]]
    dst_img = p2p.process_image(frame)
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
        video_path = "./data/R0010053_er_23.MP4"

    pi = math.pi
    inc = pi / 180
    yaw = 0  # -pi < b < pi
    pitch = 0  # -pi/2 < a < pi/2
    roll = 0
    h = 480  # 480
    w = 640  # 640
    fov = 80

    # initialize Pano2Perspective
    p2p = Pano2Pers(w, h, fov)
    p2p.cuda(0)

    times = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        s = time.time()
        p2p.set_rotation([yaw, pitch, roll])  # set rotation
        frame = [frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]]
        dst_img = p2p.process_image(frame)  # process the image
        e = time.time()

        # cv2.imshow("video", dst_img)

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
    plt.savefig('test_cpp_video.png')


if __name__ == "__main__":
    main()
