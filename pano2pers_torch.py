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

        self.K_inv = self.create_intrinsic_params(fov, self.pers_w, self.pers_h)

        self.cuda = False

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[[2, 1, 0]])
        ])

        _vi = torch.linspace(0, self.pers_w - 1, self.pers_w)
        _vj = torch.linspace(0, self.pers_h - 1, self.pers_h)

        vj, vi = torch.meshgrid([_vj, _vi])
        vk = torch.ones_like(vi)
        coord = torch.stack((vi, vj, vk), dim=2)
        self.coord = coord.unsqueeze(3)

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
    def create_intrinsic_params(self, fov_x, pers_w, pers_h):
        f = pers_w / (2 * np.tan(np.radians(fov_x) / 2))
        K = torch.tensor([[f, 0, pers_w / 2],
                        [0, f, pers_h / 2],
                        [0, 0, 1]])
        return K.inverse()

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
        R = torch.Tensor(R_roll) @ torch.Tensor(R_pitch) @ torch.Tensor(R_yaw)
        self.R_inv = R.inverse()

    def get_perspective(self, pano_img):
        """
        :param pano_img: numpy array
        """
        

        pano_h = pano_img.shape[0]
        pano_w = pano_img.shape[1]

        rot_coord = self.R_inv @ self.K_inv @ self.coord
        rot_coord = rot_coord.squeeze(3)

        a = torch.atan2(rot_coord[:, :, 0], rot_coord[:, :, 2])
        b = torch.asin(rot_coord[:, :, 1] / torch.norm(rot_coord, dim=2))

        ui = (a + np.pi) * pano_w / (2 * np.pi)
        uj = (b + np.pi / 2) * pano_h / np.pi

        norm_ui = 2 * (ui - pano_w / 2) / pano_w
        norm_uj = 2 * (uj - pano_h / 2) / pano_h

        grid = torch.stack((norm_ui, norm_uj), 2).unsqueeze(0)

        pano = self.trans(pano_img)
        pano = pano.expand(grid.size(0), -1, -1, -1)

        pers = torch.nn.functional.grid_sample(pano, grid).squeeze(0)
        pers = pers.permute(1, 2, 0)
        return pers.numpy()[..., [2,1,0]]
