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

from pano2pers import Pano2Pers


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
        src_img = cv2.imread("./data/pano2.png")

    pi = math.pi
    inc = pi / 36
    yaw = 0 * inc  # -pi < b < pi
    pitch = 0 * inc  # -pi/2 < a < pi/2
    roll = 0
    rot = [yaw, pitch, roll]

    p2p = Pano2Pers.from_crop_size(640, 480, 90)
    p2p.set_rotation([0, 0, 0])

    s = time.time()
    dst_img = p2p.get_perspective(src_img)
    e = time.time()
    print(e - s)
    cv2.imshow("output", rescale_frame(dst_img,percent=100))
    cv2.waitKey()


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
    x_axis = [ i for i in range(len(times)) ]
    plt.plot(x_axis, times)
    plt.savefig('test_video.png')


if __name__=="__main__":
    main()
