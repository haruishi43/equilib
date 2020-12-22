#!/usr/bin/env python3

from typing import Dict, List, Union
import os.path as osp

import numpy as np

import torch

from torchvision import transforms

from PIL import Image

from equilib import Cube2Equi, cube2equi

from tests.common.timer import timer

# Variables
W_OUT, H_OUT = (480, 240)  # Output panorama shape
CUBE_FORMAT = "dict"  # Input cube format
SAMPLING_METHOD = "default"  # Sampling method
MODE = "bilinear"  # Sampling mode
USE_CLASS = True  # Class or function

# Paths
DATA_PATH = osp.join(".", "tests", "data")
RESULT_PATH = osp.join(".", "tests", "results")

# Batch
BATCH_SIZE = 4

# PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_cubemap_shape(
    cube: Union[
        np.ndarray,
        torch.Tensor,
        Dict[str, Union[np.ndarray, torch.Tensor]],
        List[Union[np.ndarray, torch.Tensor]],
    ],
    cube_format: str,
) -> None:
    if isinstance(cube, list):
        c = cube[0]  # get the first of the batch
        if isinstance(c, list):
            assert cube_format == "list"
            print("one: {}".format(c[0].shape))
        elif isinstance(c, dict):
            assert cube_format == "dict"
            print("one: {}".format(c["F"].shape))
        elif isinstance(c, np.ndarray):
            assert cube_format in ["horizon", "dice", "list"]
            # can be single list
            if cube_format == "list":
                print("one: {}".format(c.shape))
            else:
                print("one: {}".format(c[0].shape))
    elif isinstance(cube, dict):
        assert cube_format == "dict"
        print("one: {}".format(cube["F"].shape))
    else:
        assert cube_format in ["horizon", "dice"]
        print("one: {}".format(cube.shape))


@timer
def run_cube2equi(
    cube: Union[
        np.ndarray,
        torch.Tensor,
        Dict[str, Union[np.ndarray, torch.Tensor]],
        List[Union[np.ndarray, torch.Tensor]],
    ],
    cube_format: str,
    w_out: int,
    h_out: int,
    sampling_method: str,
    mode: str,
    use_class: bool,
) -> Union[np.ndarray, torch.Tensor]:
    # check_cubemap_shape(cube, cube_format)
    if use_class:
        cube2equi_instance = Cube2Equi(
            cube_format=cube_format,
            w_out=w_out,
            h_out=h_out,
            sampling_method=sampling_method,
            mode=mode,
        )
        samples = cube2equi_instance(cube)
    else:
        samples = cube2equi(
            cubemap=cube,
            cube_format=cube_format,
            w_out=w_out,
            h_out=h_out,
            sampling_method=sampling_method,
            mode=mode,
        )
    return samples


def create_single_numpy_input(
    data_path: str,
    cube_format: str,
) -> Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]]:
    if cube_format in ("horizon", "dice"):
        img_path = osp.join(data_path, "test_{}.jpg".format(cube_format))
        cube = Image.open(img_path)
        cube = cube.convert("RGB")
        cube = np.asarray(cube)
        cube = np.transpose(cube, (2, 0, 1))
    elif cube_format in ("dict", "list"):
        img_paths = osp.join(data_path, "test_dict_{k}.jpg")
        cube = {}
        for k in ("F", "R", "B", "L", "U", "D"):
            face = Image.open(img_paths.format(cube_format=cube_format, k=k))
            face = face.convert("RGB")
            face = np.asarray(face)
            face = np.transpose(face, (2, 0, 1))
            cube[k] = face
        if cube_format == "list":
            cube = list(cube.values())
    else:
        raise ValueError
    return cube


def create_batch_numpy_input(
    data_path: str,
    cube_format: str,
    batch_size: int,
) -> List[Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]]]:
    batch_cube = []
    for _ in range(batch_size):
        if cube_format in ("horizon", "dice"):
            img_path = osp.join(data_path, "test_{}.jpg".format(cube_format))
            cube = Image.open(img_path)
            cube = cube.convert("RGB")
            cube = np.asarray(cube)
            cube = np.transpose(cube, (2, 0, 1))
        elif cube_format in ("dict", "list"):
            img_paths = osp.join(data_path, "test_dict_{k}.jpg")
            cube = {}
            for k in ("F", "R", "B", "L", "U", "D"):
                face = Image.open(
                    img_paths.format(cube_format=cube_format, k=k)
                )
                face = face.convert("RGB")
                face = np.asarray(face)
                face = np.transpose(face, (2, 0, 1))
                cube[k] = face
            if cube_format == "list":
                cube = list(cube.values())
        else:
            raise ValueError
        batch_cube.append(cube)
    return batch_cube


def create_single_torch_input(
    data_path: str,
    cube_format: str,
    device: torch.device,
) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    if cube_format in ("horizon", "dice"):
        img_path = osp.join(data_path, "test_{}.jpg".format(cube_format))
        cube = Image.open(img_path)
        cube = cube.convert("RGB")
        cube = to_tensor(cube).to(device)
    elif cube_format in ("dict", "list"):
        img_paths = osp.join(data_path, "test_dict_{k}.jpg")
        cube = {}
        for k in ("F", "R", "B", "L", "U", "D"):
            face = Image.open(img_paths.format(cube_format=cube_format, k=k))
            face = face.convert("RGB")
            face = to_tensor(face).to(device)
            cube[k] = face
        if cube_format == "list":
            cube = list(cube.values())
    else:
        raise ValueError
    return cube


def create_batch_torch_input(
    data_path: str,
    cube_format: str,
    batch_size: int,
    device: torch.device,
) -> List[Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]]:
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    batch_cube = []
    for _ in range(batch_size):
        if cube_format in ("horizon", "dice"):
            img_path = osp.join(data_path, "test_{}.jpg".format(cube_format))
            cube = Image.open(img_path)
            cube = cube.convert("RGB")
            cube = to_tensor(cube).to(device)
        elif cube_format in ("dict", "list"):
            img_paths = osp.join(data_path, "test_dict_{k}.jpg")
            cube = {}
            for k in ("F", "R", "B", "L", "U", "D"):
                face = Image.open(
                    img_paths.format(cube_format=cube_format, k=k)
                )
                face = face.convert("RGB")
                face = to_tensor(face).to(device)
                cube[k] = face
            if cube_format == "list":
                cube = list(cube.values())
        else:
            raise ValueError
        batch_cube.append(cube)
    return batch_cube


def process_single_numpy_output(
    equi: np.ndarray,
    cube_format: str,
    result_path: str,
) -> None:
    equi = np.transpose(equi, (1, 2, 0))
    equi_img = Image.fromarray(equi)
    out_path = osp.join(
        result_path, "cube2equi_numpy_single_{}.jpg".format(cube_format)
    )
    equi_img.save(out_path)


def process_batch_numpy_output(
    equis: np.ndarray,
    cube_format: str,
    result_path: str,
) -> None:
    for i, equi in enumerate(equis):
        equi = np.transpose(equi, (1, 2, 0))
        equi_img = Image.fromarray(equi)
        out_path = osp.join(
            result_path,
            "cube2equi_numpy_batched_{}_{}.jpg".format(cube_format, i),
        )
        equi_img.save(out_path)


def process_single_torch_output(
    equi: torch.Tensor,
    cube_format: str,
    result_path: str,
) -> None:
    to_PIL = transforms.Compose(
        [
            transforms.ToPILImage(),
        ]
    )
    equi_img = to_PIL(equi.to("cpu"))
    out_path = osp.join(
        result_path, "cube2equi_torch_single_{}.jpg".format(cube_format)
    )
    equi_img.save(out_path)


def process_batch_torch_output(
    equis: torch.Tensor,
    cube_format: str,
    result_path: str,
) -> None:
    to_PIL = transforms.Compose(
        [
            transforms.ToPILImage(),
        ]
    )
    for i, equi in enumerate(equis):
        equi_img = to_PIL(equi.to("cpu"))
        out_path = osp.join(
            result_path,
            "cube2equi_torch_batched_{}_{}.jpg".format(cube_format, i),
        )
        equi_img.save(out_path)


#########
# Tests #
#########


def test_numpy_single():
    print("test_numpy_single")
    cube = create_single_numpy_input(
        data_path=DATA_PATH,
        cube_format=CUBE_FORMAT,
    )
    equi = run_cube2equi(
        cube,
        cube_format=CUBE_FORMAT,
        w_out=W_OUT,
        h_out=H_OUT,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        use_class=USE_CLASS,
    )
    process_single_numpy_output(
        equi,
        cube_format=CUBE_FORMAT,
        result_path=RESULT_PATH,
    )


def test_numpy_batch():
    print("test_numpy_batch")
    cubes = create_batch_numpy_input(
        data_path=DATA_PATH,
        cube_format=CUBE_FORMAT,
        batch_size=BATCH_SIZE,
    )
    equis = run_cube2equi(
        cubes,
        cube_format=CUBE_FORMAT,
        w_out=W_OUT,
        h_out=H_OUT,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        use_class=USE_CLASS,
    )
    process_batch_numpy_output(
        equis,
        cube_format=CUBE_FORMAT,
        result_path=RESULT_PATH,
    )


def test_torch_single():
    print("test_torch_single")
    cube = create_single_torch_input(
        data_path=DATA_PATH,
        cube_format=CUBE_FORMAT,
        device=DEVICE,
    )
    equi = run_cube2equi(
        cube,
        cube_format=CUBE_FORMAT,
        w_out=W_OUT,
        h_out=H_OUT,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        use_class=USE_CLASS,
    )
    process_single_torch_output(
        equi,
        cube_format=CUBE_FORMAT,
        result_path=RESULT_PATH,
    )


def test_torch_batch():
    print("test_torch_batch")
    cubes = create_batch_torch_input(
        data_path=DATA_PATH,
        cube_format=CUBE_FORMAT,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    equis = run_cube2equi(
        cubes,
        cube_format=CUBE_FORMAT,
        w_out=W_OUT,
        h_out=H_OUT,
        sampling_method=SAMPLING_METHOD,
        mode=MODE,
        use_class=USE_CLASS,
    )
    process_batch_torch_output(
        equis,
        cube_format=CUBE_FORMAT,
        result_path=RESULT_PATH,
    )
