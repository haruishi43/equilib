import os
import sys
import math

# don't really want to use this for gpu
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms


class Pano2Pers:
    def __init__(self, pers_h, pers_w, fov, device=-1, debug=False):
        """
        Initialize parameters
        :param pers_h, pers_w: height and width of perspective image (int)
        :param fov: field of view in degrees (int)
        """
        if debug:
            print(f'Perspective Image: h={pers_h}, w={pers_w}')

        self.K_inv = self.create_intrinsic_params(pers_h, pers_w, fov)

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[[2, 1, 0]])  # input assumes bgr
        ])

        self.invtrans = transforms.Compose([
            transforms.ToPILImage()
        ])

        _vi = torch.linspace(0, pers_w - 1, pers_w)
        _vj = torch.linspace(0, pers_h - 1, pers_h)

        vj, vi = torch.meshgrid([_vj, _vi])
        vk = torch.ones_like(vi)
        coord = torch.stack((vi, vj, vk), dim=2)
        self.coord = coord.unsqueeze(3)

        self.device = torch.device(device if device > -1 else "cpu")
        self.K_inv = self.K_inv.to(self.device)
        self.coord = self.coord.to(self.device)

    @classmethod
    def from_crop_size(cls, pers_h, pers_w, fov, device=-1, debug=False):
        """
        param: pers_h, pers_w: perspective image size
        param: fov: field of view in deg
        """
        return cls(pers_h, pers_w, fov, device, debug)

    @classmethod
    def from_pano_size(cls, pano_h, pano_w, fov_x, fov_y, device=-1, debug=False):
        """
        param: pano_h, pano_w: panorama image size
        param: fov_x, fov_y: field of view in deg
        """
        pers_w = int(pano_w * fov_x / 360.0 + 0.5)
        pers_h = int(pano_h * fov_y / 360.0 + 0.5)
        return cls(pers_h, pers_w, fov_x, device, debug)

    def create_intrinsic_params(self, pers_h, pers_w, fov_x):
        f = pers_w / (2 * np.tan(np.radians(fov_x) / 2))
        K = torch.tensor([
            [f, 0, pers_w / 2],
            [0, f, pers_h / 2],
            [0, 0, 1]])
        return K.inverse()

    def set_rotation(self, rot):
        """
        param: rot: [yaw, pitch, roll]
        """
        rot_yaw, rot_pitch, rot_roll = rot

        R_yaw = [
            [np.cos(rot_yaw), 0, -np.sin(rot_yaw)],
            [0, 1, 0],
            [np.sin(rot_yaw), 0, np.cos(rot_yaw)]]
        R_pitch = [
            [1, 0, 0],
            [0, np.cos(rot_pitch), -np.sin(rot_pitch)],
            [0, np.sin(rot_pitch), np.cos(rot_pitch)]]
        R_roll = [
            [np.cos(rot_roll), -np.sin(rot_roll), 0],
            [np.sin(rot_roll), np.cos(rot_roll), 0],
            [0, 0, 1]]
        R = torch.Tensor(R_roll) @ torch.Tensor(R_pitch) @ torch.Tensor(R_yaw)
        self.R_inv = R.inverse()
        self.R_inv = self.R_inv.to(self.device)

    def convert_rgb(self, img):
        return img[..., [2,1,0]]

    def transform_pano(self, pano):
        return self.trans(pano)

    def pixel_wise_rot(self, rot_coord):
        a = torch.atan2(rot_coord[:, :, 0], rot_coord[:, :, 2])
        b = torch.asin(rot_coord[:, :, 1] / torch.norm(rot_coord, dim=2))
        return a, b

    def get_perspective(self, pano_img):
        """
        :param pano_img: numpy array
        """
        pano_h = pano_img.shape[0]
        pano_w = pano_img.shape[1]

        rot_coord = self.R_inv @ self.K_inv @ self.coord
        rot_coord = rot_coord.squeeze(3)

        a, b = self.pixel_wise_rot(rot_coord)
        
        ui = (a + math.pi) * pano_w / (2 * math.pi)
        uj = (b + math.pi / 2) * pano_h / math.pi

        # i don't like how I normized my coordinates to fit  into a -1 ~ 1 grid here
        norm_ui = 2 * (ui - pano_w / 2) / pano_w
        norm_uj = 2 * (uj - pano_h / 2) / pano_h

        grid = torch.stack((norm_ui, norm_uj), 2).unsqueeze(0)

        pano = self.transform_pano(pano_img)
        pano = pano.expand(grid.size(0), -1, -1, -1)
        pano = pano.to(self.device)
        
        pers = F.grid_sample(pano, grid).squeeze(0)
        pers = pers.permute(1, 2, 0)
        pers = pers.to("cpu")  # output is always cpu (for now...)
        
        pers = pers.numpy()
        pers = pers * 255
        pers = pers.astype(np.uint8)

        return pers
