import os
import sys
import cv2
from scipy import ndimage
import numpy as np


class Pano2Pers:
    def __init__(self, w, h, fov):

        # setting up parameters:
        self.pers_w = w
        self.pers_h = h
        self.pers_shape = (w, h)
        self.pers_img = np.zeros(self.pers_shape)
        print(f'Perspective Image: w={self.pers_w}, h={self.pers_h}')

        self.radius = 1
        self.fov_x = fov
        self.rot_1 = 0
        self.rot_2 = 0

        self.cuda = False

    # Constructors:

    @classmethod
    def from_crop_size(cls, w, h, fov):
        """
        For now, FOV is in degrees since it is easier than passing using pi
        """
        return cls(w, h, fov)

    @classmethod
    def from_pano_size(cls, w, h, fov_x, fov_y):
        pers_w = int(w * fov_x / 360.0 + 0.5)
        pers_h = int(h * fov_y / 360.0 + 0.5)
        return cls(pers_w, pers_h, fov_x)
    
    # supporting function

    def set_rotation(self, angles):
        # [yaw, pitch, roll]
        theta = angles[0]
        phi = angles[0]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [rot_1, _] = cv2.Rodrigues(z_axis * np.radians(theta))
        [rot_2, _] = cv2.Rodrigues(np.dot(rot_1, y_axis) * np.radians(phi))

        self.rot_1 = rot_1
        self.rot_2 = rot_2

    def cuda(self):
        self.cuda = True

    def get_perspective(self, pano_img):
        """
        pano: numpy array
        """

        R = self.radius
        rot_1 = self.rot_1
        rot_2 = self.rot_2
        pers_h = self.pers_img.shape[1]
        pers_w = self.pers_img.shape[0]
        pers_cx = (pers_w - 1) / 2.0
        pers_cy = (pers_h - 1) / 2.0
        pano_h = pano_img.shape[0]
        pano_w = pano_img.shape[1]
        pano_cx = (pano_w - 1) / 2.0
        pano_cy = (pano_h - 1) / 2.0
        
        fov_x = self.fov_x
        fov_y = float(pers_h) / pers_w * self.fov_x

        wangle = (180 - fov_x) / 2.0
        w_len = 2 * R * np.sin(np.radians(fov_x / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (pers_w - 1)

        hangle = (180 - fov_y) / 2.0
        h_len = 2 * R * np.sin(np.radians(fov_y / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (pers_h - 1)

        x_map = np.zeros([pers_h, pers_w], np.float32) + R
        y_map = np.tile((np.arange(0, pers_w) - pers_cx) * w_interval, [pers_h, 1])
        z_map = -np.tile((np.arange(0, pers_h) - pers_cy) * h_interval, [pers_w, 1]).T
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)

        xyz = np.zeros([pers_h, pers_w, 3], np.float)
        xyz[:, :, 0] = (R / D * x_map)[:, :]
        xyz[:, :, 1] = (R / D * y_map)[:, :]
        xyz[:, :, 2] = (R / D * z_map)[:, :]
        
        xyz = xyz.reshape([pers_h * pers_w, 3]).T
        xyz = np.dot(rot_1, xyz)
        xyz = np.dot(rot_2, xyz).T

        lat = np.arcsin(xyz[:, 2] / R)
        lon = np.zeros([pers_h * pers_w], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)
        
        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([pers_h, pers_w]) / np.pi * 180
        lat = -lat.reshape([pers_h, pers_w]) / np.pi * 180
        lon = lon / 180 * pano_cx + pano_cx
        lat = lat / 90 * pano_cy + pano_cy

        _map = np.stack([lat, lon], -1)
        _map = np.rollaxis(_map, -1)
        print(_map.shape)
        print(pano_img.shape)

        persp = cv2.remap(pano_img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        # pers_img = np.zeros([pers_h, pers_w, 3])
        # pers_img[:,:,0] = ndimage.map_coordinates(pano_img[:,:,0], _map.astype(np.float32), order=1, mode='wrap')
        # pers_img[:,:,1] = ndimage.map_coordinates(pano_img[:,:,1], _map.astype(np.float32), order=1, mode='wrap')
        # pers_img[:,:,2] = ndimage.map_coordinates(pano_img[:,:,2], _map.astype(np.float32), order=1, mode='wrap')

        # print(pers_img.shape)

        return persp
