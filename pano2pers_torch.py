import os
import sys
import numpy as np

import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image


class Pano2Pers:
    def __init__(self, h, w, fov):

        # setting up parameters:
        self.pers_h = h
        self.pers_w = w
        self.pers_size = [h, w]
        print(f'Perspective Image: h={self.pers_h}, w={self.pers_w}')

        self.fov_x = fov

        self.cuda = False

    # Constructors:

    @classmethod
    def from_crop_size(cls, h, w, fov):
        """
        For now, FOV is in degrees since it is easier than passing using pi
        """
        return cls(h, w, fov)

    @classmethod
    def from_pano_size(cls, h, w, fov_x, fov_y):
        pers_w = int(w * fov_x / 360.0 + 0.5)
        pers_h = int(h * fov_y / 360.0 + 0.5)
        return cls(pers_h, pers_w, fov_x)
    
    # supporting function

    def set_rotation(self, rot):
        # [yaw, pitch, roll]
        rot_yaw, rot_pitch, rot_roll = rot

        R_yaw = [
            [np.cos(rot_yaw), 0, -np.sin(rot_yaw)],
            [0, 1, 0],
            [np.sin(rot_yaw), 0, np.cos(rot_yaw)]
        ]
        R_pitch = [
            [1, 0, 0],
            [0, np.cos(rot_pitch), -np.sin(rot_pitch)],
            [0, np.sin(rot_pitch), np.cos(rot_pitch)]
        ]
        R_roll = [
            [np.cos(rot_roll), -np.sin(rot_roll), 0],
            [np.sin(rot_roll), np.cos(rot_roll), 0],
            [0, 0, 1]
        ]

        self.R = torch.Tensor(R_roll) @ torch.Tensor(R_pitch) @ torch.Tensor(R_yaw)

    def get_perspective(self, pano_img):
        """
        :param pano_img: numpy array
        """

        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[[2, 1, 0]])
        ])

        

        fov_y = float(pano_img.shape[0]) / pano_img.shape[1] * self.fov_x

        deg_f_x = (180 - self.fov_x) / 2.0
        deg_f_y = (180 - fov_y) / 2.0

        f_x = np.radians(deg_f_x)
        f_y = np.radians(deg_f_y)

        v_map = torch.linspace(-np.pi, np.pi, self.pers_size[0])
        u_map = torch.linspace(-np.pi/2, np.pi/2, self.pers_size[1])

        v, u = torch.meshgrid([v_map, u_map])
        
        u = u / f_x
        v = v / f_y
        w = torch.ones_like(u)
        coord = torch.stack((u, v, w), dim=2)

        roted_coord = coord @ self.R

        x_sin = roted_coord[:, :, 0] / torch.norm(roted_coord[:, :, [0, 2]], dim=2)
        y_sin = roted_coord[:, :, 1] / torch.norm(roted_coord, dim=2)

        x_raw = torch.asin(x_sin)
        x_raw = torch.where(roted_coord[:, :, 2] < 0, np.pi - x_raw, x_raw)
        y_raw = torch.asin(y_sin)

        x = x_raw / np.pi
        y = 2 * y_raw / np.pi

        grid = torch.stack((x, y), 2).unsqueeze(0)
        pano_img = trans(pano_img)

        pano_img = pano_img.expand(grid.size(0), -1, -1, -1)

        pers = torch.nn.functional.grid_sample(pano_img, grid).squeeze(0)
        pers = pers.permute(1, 2, 0)
        return pers.numpy()[..., [2,1,0]]
