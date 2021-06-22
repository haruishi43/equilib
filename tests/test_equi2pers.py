#!/usr/bin/env python3

import copy
from typing import Dict, List, Union
import os.path as osp

import numpy as np

import torch

from torchvision import transforms

from PIL import Image

from equilib import Equi2Pers, equi2pers

from tests.common.timer import timer

# Variables
WIDTH, HEIGHT = (640, 480)  # Output pers shape
FOV = 90  # Output pers fov
SAMPLING_METHOD = "default"  # Sampling method
MODE = "bilinear"  # Sampling mode
Z_DOWN = False  # z-axis control
USE_CLASS = True  # Class or function

# Paths
DATA_PATH = osp.join(".", "tests", "data")
RESULT_PATH = osp.join(".", "tests", "results")

# Batch
BATCH_SIZE = 4

# PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@timer
def run_equi2pers(
    equi: Union[np.ndarray, torch.Tensor],
    rot: Union[Dict[str, float], List[Dict[str, float]]],
    w_out: int,
    h_out: int,
    fov: int,
    sampling_method: str,
    mode: str,
    z_down: bool,
    use_class: bool,
) -> Union[np.ndarray, torch.Tensor]:
    # h_equi, w_equi = equi.shape[-2:]
    # print(f"equirectangular image size: ({h_equi}, {w_equi}")
    if use_class:
        equi2pers_instance = Equi2Pers(
            w_pers=w_out,
            h_pers=h_out,
            fov_x=fov,
            sampling_method=sampling_method,
            mode=mode,
            z_down=z_down,
        )
        sample = equi2pers_instance(
            equi=equi,
            rot=rot,
        )
    else:
        sample = equi2pers(
            equi=equi,
            rot=rot,
            w_pers=w_out,
            h_pers=w_out,
            fov_x=fov,
            sampling_method=sampling_method,
            mode=mode,
            z_down=z_down,
        )
    return sample


def create_single_numpy_input(
    data_path: str,
) -> np.ndarray:
    equi_path = osp.join(data_path, "test.jpg")
    equi_img = Image.open(equi_path)
    # Sometimes images are RGBA
    equi_img = equi_img.convert("RGB")
    equi = np.asarray(equi_img)
    equi = np.transpose(equi, (2, 0, 1))
    return equi


def create_batch_numpy_input(
    data_path: str,
    batch_size: int,
) -> np.ndarray:
    equi_path = osp.join(data_path, "test.jpg")
    batch_equi = []
    for _ in range(batch_size):
        equi_img = Image.open(equi_path)
        # Sometimes images are RGBA
        equi_img = equi_img.convert("RGB")
        equi = np.asarray(equi_img)
        equi = np.transpose(equi, (2, 0, 1))
        batch_equi.append(equi)
    batch_equi = np.stack(batch_equi, axis=0)
    return batch_equi


def create_single_torch_input(
    data_path: str,
    device: torch.device,
) -> torch.Tensor:
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    equi_path = osp.join(data_path, "test.jpg")
    equi_img = Image.open(equi_path)
    # NOTE: Sometimes images are RGBA
    equi_img = equi_img.convert("RGB")
    equi = to_tensor(equi_img)
    equi = equi.to(device)
    return equi


def create_batch_torch_input(
    data_path: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    equi_path = osp.join(data_path, "test.jpg")
    equi_img = Image.open(equi_path)
    # NOTE: Sometimes images are RGBA
    equi_img = equi_img.convert("RGB")
    batch_equi = []
    for _ in range(batch_size):
        equi = to_tensor(equi_img)
        batch_equi.append(copy.deepcopy(equi))
    batch_equi = torch.stack(batch_equi, dim=0)
    batch_equi = batch_equi.to(device)
    return batch_equi


def create_single_rot() -> Dict[str, float]:
    rot = {
        "roll": 0,
        "pitch": 0,
        "yaw": 0,
    }
    return rot


def create_batch_rot(batch_size: int) -> List[Dict[str, float]]:
    batch_rot = []
    inc = np.pi / 8
    for i in range(batch_size):
        rot = {
            "roll": 0,
            "pitch": i * inc,
            "yaw": 0,
        }
        batch_rot.append(rot)
    return batch_rot


def process_single_numpy_output(
    pers: np.ndarray,
    result_path: str,
) -> None:
    pers = np.transpose(pers, (1, 2, 0))
    pers_img = Image.fromarray(pers)
    pers_path = osp.join(result_path, "equi2pers_numpy_single.jpg")
    pers_img.save(pers_path)


def process_batch_numpy_output(
    perss: np.ndarray,
    result_path: str,
    batch_size: int,
) -> None:
    batch_pers = []
    for i in range(batch_size):
        sample = perss[i]
        pers = np.transpose(sample, (1, 2, 0))
        pers_img = Image.fromarray(pers)
        batch_pers.append(pers_img)

    for i, pers in enumerate(batch_pers):
        pers_path = osp.join(
            result_path, "equi2pers_numpy_batch_{}.jpg".format(i)
        )
        pers.save(pers_path)


def process_single_torch_output(
    pers: torch.Tensor,
    result_path: str,
) -> None:
    to_PIL = transforms.Compose(
        [
            transforms.ToPILImage(),
        ]
    )
    pers = pers.to("cpu")
    pers_img = to_PIL(pers)
    pers_path = osp.join(result_path, "output_torch_single.jpg")
    pers_img.save(pers_path)


def process_batch_torch_output(
    perss: torch.Tensor,
    result_path: str,
    batch_size: int,
) -> None:
    to_PIL = transforms.Compose(
        [
            transforms.ToPILImage(),
        ]
    )
    batch_pers = []
    for i in range(batch_size):
        sample = copy.deepcopy(perss[i])
        sample = sample.to("cpu")
        pers_img = to_PIL(sample)
        batch_pers.append(pers_img)

    for i, pers in enumerate(batch_pers):
        pers_path = osp.join(result_path, "output_torch_batch_{}.jpg".format(i))
        pers.save(pers_path)


#########
# Tests #
#########


def test_numpy_single():
    print("test_numpy_single")
    equi = create_single_numpy_input(
        data_path=DATA_PATH,
    )
    rot = create_single_rot()
    pers = run_equi2pers(
        equi=equi,
        rot=rot,
        w_out=WIDTH,
        h_out=HEIGHT,
        fov=FOV,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        z_down=Z_DOWN,
        use_class=USE_CLASS,
    )
    process_single_numpy_output(
        pers=pers,
        result_path=RESULT_PATH,
    )


def test_numpy_batch():
    print("test_numpy_batch")
    equis = create_batch_numpy_input(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
    )
    rots = create_batch_rot(
        batch_size=BATCH_SIZE,
    )
    pers = run_equi2pers(
        equi=equis,
        rot=rots,
        w_out=WIDTH,
        h_out=HEIGHT,
        fov=FOV,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        z_down=Z_DOWN,
        use_class=USE_CLASS,
    )
    process_batch_numpy_output(
        perss=pers,
        result_path=RESULT_PATH,
        batch_size=BATCH_SIZE,
    )


def test_torch_single():
    print("test_torch_single")
    equi = create_single_torch_input(
        data_path=DATA_PATH,
        device=DEVICE,
    )
    rot = create_single_rot()
    pers = run_equi2pers(
        equi=equi,
        rot=rot,
        w_out=WIDTH,
        h_out=HEIGHT,
        fov=FOV,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        z_down=Z_DOWN,
        use_class=USE_CLASS,
    )
    process_single_torch_output(
        pers=pers,
        result_path=RESULT_PATH,
    )


def test_torch_batch():
    print("test_torch_batch")
    equis = create_batch_torch_input(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    rots = create_batch_rot(
        batch_size=BATCH_SIZE,
    )
    pers = run_equi2pers(
        equi=equis,
        rot=rots,
        w_out=WIDTH,
        h_out=HEIGHT,
        fov=FOV,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        z_down=Z_DOWN,
        use_class=USE_CLASS,
    )
    process_batch_torch_output(
        perss=pers,
        result_path=RESULT_PATH,
        batch_size=BATCH_SIZE,
    )
