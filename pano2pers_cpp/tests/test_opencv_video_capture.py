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

# import pano2perspective as pano


def rescale_frame(frame, percent=75):
    '''Rescale image for imshow'''
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='?', default=None, type=str)
    args = parser.parse_args()

    path = args.data
    test_video(path)


def test_video(path=None):
    if path is not None:
        video_path = path
    else:
        video_path = "./data/outdoor_017.MP4"

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(7))
    frame_count = 0

    while True:
        ret, frame = cap.read()

        assert ret is True, 'no frames'

        frame_count += 1
        if frame_count == 20: #total_frames - 1:
            print('reset')
            cap.release()
            cap = cv2.VideoCapture(video_path)

            frame_count = 0

            # cap.set(1, 0)


    cap.release()


if __name__=="__main__":
    main()
