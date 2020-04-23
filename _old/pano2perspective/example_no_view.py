#!/usr/bin/env python3
import sys
import argparse
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np

import pano2perspective as pano


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
    rot = [yaw, pitch, roll]

    arr = [src_img[:,:, 0], src_img[:, :, 1], src_img[:, :, 2]]
    s = time.time()
    dst_img = np.array(pano.process_single_image(arr, rot, True, 640, 480, 90.0), copy=False)
    e = time.time()
    print(e - s)


def test_video(path=None):
    if path is not None:
        video_path = path
    else:
        video_path = "./data/pano_4k_outdoor.MP4"

    # initialize Pano2Perspective
    p2p = pano.Pano2Perspective(640, 480, 90.0)
    p2p.cuda(0)  # use cuda

    K = p2p.get_intrinsics()
    print(K)

    pi = math.pi
    inc = pi / 180
    yaw = 0  # -pi < b < pi
    pitch = 0  # -pi/2 < a < pi/2
    roll = 0
    times = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        s = time.time()
        arr = [frame[:,:, 0], frame[:, :, 1], frame[:, :, 2]]
        p2p.set_rotation([yaw, pitch, roll])  # set rotation
        dst_img = np.array(p2p.process_image(arr), copy=False)  # process the image
        e = time.time()

        times.append(e - s)

    cap.release()

    print(sum(times)/len(times))
    x_axis = [ i for i in range(len(times)) ]
    plt.plot(x_axis, times)
    plt.savefig('test_video_no_view.png')


if __name__=="__main__":
    main()
