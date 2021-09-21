
from functools import lru_cache
from typing import Dict, List, Union, Tuple
from numpy import float32

import torch

import torch

from equilib.grid_sample import torch_grid_sample
from equilib.torch_utils import (
    create_global2camera_rotation_matrix,
    create_grid,
    create_intrinsic_matrix,
    create_rotation_matrices,
    get_device,
    calculate_tangent_rots
)


def ico2dict(icos: torch.Tensor) -> Dict[str, torch.Tensor]:
    ico_dict = {}
    for i, ico in enumerate(icos):
        ico_dict[i] = ico
    return ico_dict


@lru_cache(maxsize=128)
def create_cam2global_matrix(
    height: int,
    width: int,
    fov_x: float,
    skew: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:

    K = create_intrinsic_matrix(
        height=height,
        width=width,
        fov_x=fov_x,
        skew=skew,
        dtype=dtype,
        device=device,
    )
    g2c_rot = create_global2camera_rotation_matrix(
        dtype=dtype,
        device=device,
    )

    return g2c_rot @ K.inverse()


def prep_matrices(
    height: int,
    width: int,
    batch: int,
    fov_x: float,
    skew: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:

    m = create_grid(
        height=height,
        width=width,
        batch=batch,
        dtype=dtype,
        device=device,
    )
    m = m.unsqueeze(-1)
    G = create_cam2global_matrix(
        height=height,
        width=width,
        fov_x=fov_x,
        skew=skew,
        dtype=dtype,
        device=device,
    )

    return m, G


def matmul(
    m: torch.Tensor,
    G: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:

    M = torch.matmul(torch.matmul(R, G)[:, None, None, ...], m)
    M = M.squeeze(-1)

    return M


def convert_grid(
    M: torch.Tensor,
    h_equi: int,
    w_equi: int,
    method: str = "robust",
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:

    # convert to rotation
    phi = torch.asin(M[..., 2] / torch.norm(M, dim=-1))
    theta = torch.atan2(M[..., 1], M[..., 0])
    pi = torch.Tensor([3.14159265358979323846]).to(device=device)

    if method == "robust":
        ui = (theta - pi) * w_equi / (2 * pi)
        uj = (phi - pi / 2) * h_equi / pi
        ui += 0.5
        uj += 0.5
        ui %= w_equi
        uj %= h_equi
    elif method == "faster":
        ui = (theta - pi) * w_equi / (2 * pi)
        uj = (phi - pi / 2) * h_equi / pi
        ui += 0.5
        uj += 0.5
        ui = torch.where(ui < 0, ui + w_equi, ui)
        ui = torch.where(ui >= w_equi, ui - w_equi, ui)
        uj = torch.where(uj < 0, uj + h_equi, uj)
        uj = torch.where(uj >= h_equi, uj - h_equi, uj)
    else:
        raise ValueError(f"ERR: {method} is not supported")

    # stack the pixel maps into a grid
    grid = torch.stack((uj, ui), dim=-3)

    return grid


def run(
    equi: torch.Tensor,
    sub_level: List[int],
    w_face: int,
    fov_x: float,
    ico_format: str,
    mode: str,
    backend: str = "native",
) -> Union[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Run Equi2Pers

    params:
    - equi (np.ndarray): 4 dims (b, c, h, w)
    - rot (List[dict]): dict of ('yaw', 'pitch', 'roll')
    - w_face (int): icosahedron face width
    - fov_x (float): fov of horizontal axis in degrees
    - mode (str): sampling mode for grid_sample
    - override_func (Callable): function for overriding `grid_sample`

    return:
    - out (np.ndarray)

    NOTE: acceptable dtypes for `equi` are currently uint8, float32, and float64.
    Floats are prefered since numpy calculations are optimized for floats.

    NOTE: output array has the same dtype as `equi`

    NOTE: you can override `equilib`'s grid_sample with over grid sampling methods
    using `override_func`. The input to this function have to match `grid_sample`.

    """

    # NOTE: Assume that the inputs `equi` and `rots` are already batched up
    assert (
        len(equi.shape) == 4
    ), f"ERR: input `equi` should be 4-dim (b, c, h, w), but got {len(equi.shape)}"
    assert len(equi) == len(
        sub_level
    ), f"ERR: batch size of equi and rot differs: {len(equi)} vs {len(sub_level)}"

    equi_dtype = equi.dtype
    assert equi_dtype in (
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
    ), (
        f"ERR: input equirectangular image has dtype of {equi_dtype}which is\n"
        f"incompatible: try {(torch.uint8, torch.float16, torch.float32, torch.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as equi
    if equi.device.type == "cuda":
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype in (torch.float16, torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float16, torch.float32, torch.float64)}"
        )
    else:
        # NOTE: for cpu, it can't use half-precision
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float32, torch.float64)}"
        )
    if backend == "native" and equi_dtype == torch.uint8:
        # FIXME: hacky way of dealing with images that are uint8 when using
        # torch.grid_sample
        equi = equi.type(torch.float32)

    bs, c, h_equi, w_equi = equi.shape
    img_device = get_device(equi)
    cpu_device = torch.device("cpu")

    # FIXME: for now, calculate the grid in cpu
    # I need to benchmark performance of it when grid is created on cuda
    if equi.device.type == "cuda" and dtype == torch.float16:
        tmp_dtype = torch.float32
    else:
        tmp_dtype = dtype

    rots = calculate_tangent_rots(
        subdivision_level=sub_level,
        device=img_device
    )
    skew = 0.0
    z_down = False

    out_batch = [None for _ in range(bs)]

    for bn, (rot, img) in enumerate(zip(rots, equi)):
        # number of icosahedron faces
        fn = len(rot)

        # switch device in case if whole batch can't be allocated in gpu
        # added to avoid out of memory error on high subdivision levels
        device = img_device
        mem_dtype_size = (
            32 if tmp_dtype == torch.float32
            else 64
        )
        mem_reserved = torch.cuda.memory_reserved(img_device)
        mem_allocated = torch.cuda.memory_allocated(img_device)
        mem_available = mem_reserved - mem_allocated
        batch_mem_alloc_size = fn*c*w_face*w_face*mem_dtype_size
        if mem_available <= batch_mem_alloc_size:
            device = cpu_device

        out = torch.empty(
            (fn, c, w_face, w_face), dtype=dtype, device=device
        )


        # create grid and transfrom matrix
        m, G = prep_matrices(
            height=w_face,
            width=w_face,
            batch=fn,
            fov_x=fov_x,
            skew=skew,
            dtype=tmp_dtype,
            device=device,
        )

        # create batched rotation matrices
        R = create_rotation_matrices(
            rots=rot,
            z_down=z_down,
            dtype=tmp_dtype,
            device=device,
        )
        
        # In case of using gpu "matmul" fells down with CUDA out of memory error
        #
        # RuntimeError: CUDA out of memory. 
        # Tried to allocate 720.00 MiB 
        # (GPU 0; 2.00 GiB total capacity; 
        #       486.01 MiB already allocated; 
        #       632.93 MiB free; 
        #       502.00 MiB reserved 
        # in total by PyTorch)

        # m = m.to(cpu_device)
        # G = G.to(cpu_device)
        # R = R.to(cpu_device)

        # rotate and transform the grid
        M = matmul(m, G, R)#.to(device)

        # create a pixel map grid
        grid = convert_grid(
            M=M,
            h_equi=h_equi,
            w_equi=w_equi,
            method="robust",
            device=device
        )

        # if backend == "native":
        #     grid = grid.to(img_device)
        # FIXME: putting `grid` to device since `pure`'s bilinear interpolation requires it
        # FIXME: better way of forcing `grid` to be the same dtype?
        if equi.dtype != grid.dtype:
            grid = grid.type(equi.dtype)
        if equi.device != grid.device:
            grid = grid.to(equi.device)

        # iterate image transformation over all grids
        for i, grid in enumerate(grid):
            img_b = img[None, ...]
            grid = grid[None, ...]
            out[i] = torch.squeeze(
                torch_grid_sample(  # type: ignore
                        img=img_b,
                        grid=grid,
                        out=out[None, i, ...],
                        mode=mode,
                        backend=backend,
                )
            )

        out = (
            out.type(equi_dtype)
            if equi_dtype == torch.uint8
            else torch.clip(out, 0.0, 1.0)
        )

        # reformat the output
        if ico_format == 'dict':
            out = ico2dict(out)

        out_batch[bn] = out.to(cpu_device)
        #torch.cuda.empty_cache()

    return out_batch