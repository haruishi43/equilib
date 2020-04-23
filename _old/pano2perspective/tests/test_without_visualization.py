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
from PIL import Image

import pano2perspective as pano


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
        src_img = cv2.imread("./data/pano_4K.jpg")

    # Initialize with panorama image's size and fov (x, y)
    h, w, _ = src_img.shape
    p2p = pano.Pano2Perspective(w, h, 60, 45)
    p2p.cuda(0)  # use cuda

    p2p.set_center_point(3 * w//4, h//2)
    # p2p.set_rotation([math.pi/2, 0, 0])

    arr = [src_img[:,:, 0], src_img[:, :, 1], src_img[:, :, 2]]
    s = time.time()
    dst_img = np.array(p2p.process_image(arr), copy=False)
    e = time.time()
    print(e - s)


def test_video(path=None):
    if path is not None:
        video_path = path
    else:
        video_path = "./data/demo.mp4"


    # initialize Pano2Perspective
    p2p = pano.Pano2Perspective(640, 480, 90.0)
    p2p.cuda(0)  # use cuda
    #K = p2p.get_intrinsics()
    #print(K)


    pi = math.pi
    inc = pi / 180
    yaw = 90 * inc  # -pi < b < pi
    pitch = 0  # -pi/2 < a < pi/2
    roll = 0
    times = []

    iter_count = 0
    while True:

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(7))
        frame_count = 0

        while True:
            ret, frame = cap.read()

            assert ret is True, 'no frames'

            frame_count += 1
            if frame_count == 40: #total_frames - 1:
                break

            s = time.time()
            arr = [frame[:,:, 0], frame[:, :, 1], frame[:, :, 2]]
            p2p.set_rotation([yaw, pitch, roll])  # set rotation

            dst_img = np.array(p2p.process_image(arr), copy=False)  # process the image
            e = time.time()
            times.append(e - s)

            #time.sleep(0.5)
            #pil_img = Image.fromarray(dst_img)
            #pil_img.save('hello.png')
            #cv2.imwrite('hello.png', dst_img)
            #cv2.waitKey(30)

        cap.release()
        #cap.set(1, 0)

        iter_count += 1
        print(iter_count)
        print(sum(times)/len(times))
        times = []

        #del p2p
        if iter_count == 100:
            break


    cap.release()
    cv2.destroyAllWindows()

    print(sum(times)/len(times))
    x_axis = [ i for i in range(len(times)) ]
    plt.plot(x_axis, times)
    plt.savefig('test_video.png')


if __name__=="__main__":
    main()
