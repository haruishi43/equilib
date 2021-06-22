#!/usr/bin/env python3

import copy
from typing import Dict, List, Union
import os.path as osp

import numpy as np

import torch

from torchvision import transforms

from PIL import Image

from equilib import Equi2Equi, equi2equi

from tests.common.timer import timer

# Variables
WIDTH, HEIGHT = (640, 320)  # Output panorama shape
SAMPLING_METHOD = "default"  # Sampling method
MODE = "bilinear"  # Sampling mode
Z_DOWN = True  # z-axis control
USE_CLASS = True  # Class or function

# Paths
DATA_PATH = osp.join(".", "tests", "data")
RESULT_PATH = osp.join(".", "tests", "results")

# Batch
BATCH_SIZE = 4

# PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@timer
def run_equi2equi(
    equi: Union[np.ndarray, torch.Tensor],
    rot: Union[Dict[str, float], List[Dict[str, float]]],
    w_out: int,
    h_out: int,
    sampling_method: str,
    mode: str,
    z_down: bool,
    use_class: bool,
) -> Union[np.ndarray, torch.Tensor]:
    # h_equi, w_equi = equi.shape[-2:]
    # print(f"equirectangular image size: ({h_equi}, {w_equi}")
    if use_class:
        equi2equi_instance = Equi2Equi(
            w_out=w_out,
            h_out=h_out,
            sampling_method=sampling_method,
            mode=mode,
            z_down=z_down,
        )
        sample = equi2equi_instance(
            src=equi,
            rot=rot,
        )
    else:
        sample = equi2equi(
            src=equi,
            rot=rot,
            sampling_method=sampling_method,
            mode=mode,
            z_down=z_down,
            w_out=w_out,
            h_out=h_out,
        )
    return sample


def create_single_numpy_input(
    data_path: str,
) -> np.ndarray:
    equi_path = osp.join(data_path, "test.jpg")
    equi_img = Image.open(equi_path)
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
    sample: np.ndarray,
    result_path: str,
) -> None:
    out = np.transpose(sample, (1, 2, 0))
    out_img = Image.fromarray(out)
    out_path = osp.join(result_path, "equi2equi_numpy_single.jpg")
    out_img.save(out_path)


def process_batch_numpy_output(
    samples: np.ndarray,
    result_path: str,
    batch_size: int,
) -> None:
    batch_out = []
    for i in range(batch_size):
        sample = samples[i]
        out = np.transpose(sample, (1, 2, 0))
        out_img = Image.fromarray(out)
        batch_out.append(out_img)

    for i, out in enumerate(batch_out):
        out_path = osp.join(
            result_path, "equi2equi_numpy_batch_{}.jpg".format(i)
        )
        out.save(out_path)


def process_single_torch_output(
    sample: torch.Tensor,
    result_path: str,
) -> None:
    to_PIL = transforms.Compose(
        [
            transforms.ToPILImage(),
        ]
    )
    out = sample.to("cpu")
    out_img = to_PIL(out)
    out_path = osp.join(result_path, "equi2equi_torch_single.jpg")
    out_img.save(out_path)


def process_batch_torch_output(
    samples: torch.Tensor,
    result_path: str,
    batch_size: int,
) -> None:
    to_PIL = transforms.Compose(
        [
            transforms.ToPILImage(),
        ]
    )
    batch_out = []
    for i in range(batch_size):
        sample = copy.deepcopy(samples[i])
        sample = sample.to("cpu")
        out_img = to_PIL(sample)
        batch_out.append(out_img)

    for i, out in enumerate(batch_out):
        out_path = osp.join(
            result_path, "equi2equi_torch_batch_{}.jpg".format(i)
        )
        out.save(out_path)


#########
# Tests #
#########


def test_numpy_single():
    print("test_numpy_single")
    equi = create_single_numpy_input(data_path=DATA_PATH)
    rot = create_single_rot()
    sample = run_equi2equi(
        equi=equi,
        rot=rot,
        w_out=WIDTH,
        h_out=HEIGHT,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        z_down=Z_DOWN,
        use_class=USE_CLASS,
    )
    process_single_numpy_output(
        sample=sample,
        result_path=RESULT_PATH,
    )


def test_numpy_batch():
    print("test_numpy_batch")
    equis = create_batch_numpy_input(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
    )
    rots = create_batch_rot(batch_size=BATCH_SIZE)
    samples = run_equi2equi(
        equi=equis,
        rot=rots,
        w_out=WIDTH,
        h_out=HEIGHT,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        z_down=Z_DOWN,
        use_class=USE_CLASS,
    )
    process_batch_numpy_output(
        samples=samples,
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
    sample = run_equi2equi(
        equi=equi,
        rot=rot,
        w_out=WIDTH,
        h_out=HEIGHT,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        z_down=Z_DOWN,
        use_class=USE_CLASS,
    )
    process_single_torch_output(
        sample=sample,
        result_path=RESULT_PATH,
    )


def test_torch_batch():
    print("test_torch_batch")
    equis = create_batch_torch_input(
        data_path=DATA_PATH, batch_size=BATCH_SIZE, device=DEVICE
    )
    rots = create_batch_rot(batch_size=BATCH_SIZE)
    samples = run_equi2equi(
        equi=equis,
        rot=rots,
        w_out=WIDTH,
        h_out=HEIGHT,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        z_down=Z_DOWN,
        use_class=USE_CLASS,
    )
    process_batch_torch_output(
        samples=samples,
        result_path=RESULT_PATH,
        batch_size=BATCH_SIZE,
    )
