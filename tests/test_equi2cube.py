#!/usr/bin/env python3

from typing import Dict, List, Union
import os.path as osp

import numpy as np

import torch

from torchvision import transforms

from PIL import Image

from equilib import Equi2Cube, equi2cube

from tests.common.timer import timer

# Variables
W_FACE = 256  # Output cubemap width
CUBE_FORMAT = "dict"  # Output cube format
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
def run_equi2cube(
    equi: Union[np.ndarray, torch.Tensor],
    rot: Union[Dict[str, float], List[Dict[str, float]]],
    w_face: int,
    cube_format: str,
    sampling_method: str,
    mode: str,
    z_down: bool,
    use_class: bool,
) -> Union[np.ndarray, torch.Tensor]:
    # h_equi, w_equi = equi.shape[-2:]
    # print(f"equirectangular image size: ({h_equi}, {w_equi}")
    if use_class:
        equi2cube_instance = Equi2Cube(
            w_face=w_face,
            cube_format=cube_format,
            sampling_method=sampling_method,
            mode=mode,
            z_down=z_down,
        )
        sample = equi2cube_instance(
            equi=equi,
            rot=rot,
        )
    else:
        sample = equi2cube(
            equi=equi,
            rot=rot,
            w_face=w_face,
            cube_format=cube_format,
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
    # Sometimes images are RGBA
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
    batch_equi = []
    for _ in range(batch_size):
        equi_img = Image.open(equi_path)
        # Sometimes images are RGBA
        equi = to_tensor(equi_img)
        batch_equi.append(equi)
    batch_equi = torch.stack(batch_equi, axis=0)
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
    cube: np.ndarray,
    cube_format: str,
    result_path: str,
) -> None:
    if cube_format == "dict":
        for k, c in cube.items():
            out = np.transpose(c, (1, 2, 0))
            out_img = Image.fromarray(out)
            out_path = osp.join(
                result_path, "equi2cube_numpy_single_dict_{}.jpg".format(k)
            )
            out_img.save(out_path)
    elif cube_format == "list":
        for k, c in zip(["F", "R", "B", "L", "U", "D"], cube):
            out = np.transpose(c, (1, 2, 0))
            out_img = Image.fromarray(out)
            out_path = osp.join(
                result_path, "equi2cube_numpy_single_list_{}.jpg".format(k)
            )
            out_img.save(out_path)
    elif cube_format in ["horizon", "dice"]:
        out = np.transpose(cube, (1, 2, 0))
        out_img = Image.fromarray(out)
        out_path = osp.join(
            result_path, "equi2cube_numpy_single_{}.jpg".format(cube_format)
        )
        out_img.save(out_path)


def process_batch_numpy_output(
    cubes: np.ndarray,
    cube_format: str,
    result_path: str,
    batch_size: int,
) -> None:
    if cube_format == "dict":
        for i in range(batch_size):
            for k, c in cubes[i].items():
                out = np.transpose(c, (1, 2, 0))
                out_img = Image.fromarray(out)
                out_path = osp.join(
                    result_path,
                    "equi2cube_numpy_batched_dict_{}_{}.jpg".format(i, k),
                )
                out_img.save(out_path)
    elif cube_format == "list":
        for i in range(batch_size):
            for k, c in zip(["F", "R", "B", "L", "U", "D"], cubes[i]):
                out = np.transpose(c, (1, 2, 0))
                out_img = Image.fromarray(out)
                out_path = osp.join(
                    result_path,
                    "equi2cube_numpy_batched_list_{}_{}.jpg".format(i, k),
                )
                out_img.save(out_path)
    elif cube_format in ["horizon", "dice"]:
        for i in range(batch_size):
            out = np.transpose(cubes[i], (1, 2, 0))
            out_img = Image.fromarray(out)
            out_path = osp.join(
                result_path,
                "equi2cube_numpy_batched_{}_{}.jpg".format(cube_format, i),
            )
            out_img.save(out_path)


def process_single_torch_output(
    cube: torch.Tensor,
    cube_format: str,
    result_path: str,
) -> None:
    to_PIL = transforms.Compose(
        [
            transforms.ToPILImage(),
        ]
    )
    if cube_format == "dict":
        for k, c in cube.items():
            out = c.to("cpu")
            out_img = to_PIL(out)
            out_path = osp.join(
                result_path, "equi2cube_torch_single_dict_{}.jpg".format(k)
            )
            out_img.save(out_path)
    elif cube_format == "list":
        for k, c in zip(["F", "R", "B", "L", "U", "D"], cube):
            out = c.to("cpu")
            out_img = to_PIL(out)
            out_path = osp.join(
                result_path, "equi2cube_torch_single_list_{}.jpg".format(k)
            )
            out_img.save(out_path)
    elif cube_format in ["horizon", "dice"]:
        out = cube.to("cpu")
        out_img = to_PIL(out)
        out_path = osp.join(
            result_path, "equi2cube_torch_single_{}.jpg".format(cube_format)
        )
        out_img.save(out_path)


def process_batch_torch_output(
    cubes: torch.Tensor,
    cube_format: str,
    result_path: str,
    batch_size: int,
) -> None:
    to_PIL = transforms.Compose(
        [
            transforms.ToPILImage(),
        ]
    )
    if cube_format == "dict":
        for i in range(batch_size):
            for k, c in cubes[i].items():
                out = c.to("cpu")
                out_img = to_PIL(out)
                out_path = osp.join(
                    result_path,
                    "equi2cube_torch_batched_dict_{}_{}.jpg".format(i, k),
                )
                out_img.save(out_path)
    elif cube_format == "list":
        for i in range(batch_size):
            for k, c in zip(["F", "R", "B", "L", "U", "D"], cubes[i]):
                out = c.to("cpu")
                out_img = to_PIL(out)
                out_path = osp.join(
                    result_path,
                    "equi2cube_torch_batched_list_{}_{}.jpg".format(i, k),
                )
                out_img.save(out_path)
    elif cube_format in ["horizon", "dice"]:
        print("output: {}".format(cube_format))
        for i in range(batch_size):
            out = cubes[i].to("cpu")
            out_img = to_PIL(out)
            out_path = osp.join(
                result_path,
                "equi2cube_torch_batched_{}_{}.jpg".format(cube_format, i),
            )
            out_img.save(out_path)


#########
# Tests #
#########


def test_numpy_single():
    print("test_numpy_single")
    equi = create_single_numpy_input(data_path=DATA_PATH)
    rot = create_single_rot()
    cube = run_equi2cube(
        equi=equi,
        rot=rot,
        w_face=W_FACE,
        cube_format=CUBE_FORMAT,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        z_down=Z_DOWN,
        use_class=USE_CLASS,
    )
    process_single_numpy_output(
        cube=cube,
        cube_format=CUBE_FORMAT,
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
    cubes = run_equi2cube(
        equi=equis,
        rot=rots,
        w_face=W_FACE,
        cube_format=CUBE_FORMAT,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        z_down=Z_DOWN,
        use_class=USE_CLASS,
    )
    process_batch_numpy_output(
        cubes=cubes,
        cube_format=CUBE_FORMAT,
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
    cube = run_equi2cube(
        equi=equi,
        rot=rot,
        w_face=W_FACE,
        cube_format=CUBE_FORMAT,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        z_down=Z_DOWN,
        use_class=USE_CLASS,
    )
    process_single_torch_output(
        cube=cube,
        cube_format=CUBE_FORMAT,
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
    cubes = run_equi2cube(
        equi=equis,
        rot=rots,
        w_face=W_FACE,
        cube_format=CUBE_FORMAT,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        z_down=Z_DOWN,
        use_class=USE_CLASS,
    )
    process_batch_torch_output(
        cubes=cubes,
        cube_format=CUBE_FORMAT,
        result_path=RESULT_PATH,
        batch_size=BATCH_SIZE,
    )
