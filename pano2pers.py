import os
import sys
import cv2
import numpy as np


class Pano2Pers:
    def __init__(self, w, h, fov):

        # setting up parameters:
        self.pers_w = w
        self.pers_h = h
        self.pers_shape = (w, h)
        self.pers_img = np.zeros(self.pers_shape)
        print(f'Perspective Image: w={self.pers_w}, h={self.pers_h}')

        self.pano_w = 4*w
        self.pano_h = 4*h
        self.pano_shape = (self.pano_w, self.pano_h)
        self.pano_img = np.zeros(self.pano_shape)
        print(f'Panoramic Image: w={self.pano_w}, h={self.pano_h}')

        K = self.create_intrinsic_param(w, h, fov)
        self.K_inv = np.linalg.inv(K)
        self.im2ori = np.zeros((3,3))

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

    def create_intrinsic_param(self, w, h, fov):
        fov_rad = fov * np.pi / 180.0
        f = w/(2.0 * np.tan(fov_rad / 2.0))
        _K = [[f, 0., w/2], [0., f, h/2], [0., 0., 1.]]
        return np.array(_K)

    def set_rotation(self, angles):
        # default [yaw, pitch, roll]
        rot = self.angles2rot(np.array(angles))
        self.im2ori = np.dot(np.linalg.inv(rot), self.K_inv)

    def angles2rot(self, angles):
        """
        FIXME: the code works, but this calculation might be hacky
        """
        A = np.array([
            [1., 0., 0.],
            [0., np.cos(angles[1]), -np.sin(angles[1])],
            [0., np.sin(angles[1]), np.cos(angles[1])]
        ])
        B = np.array([
            [np.cos(angles[0]), 0., -np.sin(angles[0])],
            [0., 1., 0.],
            [np.sin(angles[0]), 0., np.cos(angles[0])]
        ])
        C = np.array([
            [np.cos(angles[2]), np.sin(angles[2]), 0.],
            [-np.sin(angles[2]), np.cos(angles[2]), 0.],
            [0., 0., 1.]
        ])
        return np.linalg.multi_dot([A, B, C])

    def cuda(self):
        self.cuda = True

    def get_perspective(self, pano_img, radius=128):
        """
        pano: numpy array
        """

        pers_img = self._with_numpy(pano_img, radius)




        return pers_img


    def _with_numpy(self, pano_img, radius=128):
        FOV = 90
        THETA = 0
        PHI = 0
        RADIUS = radius
        height = self.pers_img.shape[1]
        width = self.pers_img.shape[0]

        equ_h = self.pano_img.shape[1]
        equ_w = self.pano_img.shape[0]
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.zeros([height, width, 3], np.float)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]
        
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)
        
        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy
    
        persp = cv2.remap(pano_img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return persp


    def _with_gpu(self):

        pass