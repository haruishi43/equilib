import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from equilib import equi2cube


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("z_down", [False, True])
@pytest.mark.parametrize("cube_format", ["horizon", "list", "dict", "dice"])
def test_cube_faces_follow_the_rotation_convention(
    backend, z_down, cube_format
):
    height, width, face_width = 96, 192, 5
    longitude = 2 * np.pi * ((np.arange(width) + 0.5) / width - 0.5)
    latitude = np.pi * ((np.arange(height) + 0.5) / height - 0.5)
    lon, lat = np.meshgrid(longitude, latitude)
    # Encode the unit direction at each panorama pixel in its RGB values.
    directions = np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )
    angles = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.5],
            [-0.3, 0.4, 0.5],
        ]
    )
    rots = [
        dict(zip(("roll", "pitch", "yaw"), map(float, row))) for row in angles
    ]
    panorama = np.repeat((directions[None] + 1) / 2, len(rots), axis=0)
    if backend == "torch":
        panorama = torch.from_numpy(panorama)
    cube = equi2cube(panorama, rots, face_width, cube_format, z_down=z_down)

    # Each row gives a face's center, right direction, and down direction.
    frames = np.array(
        [
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)],  # F
            [(0, 1, 0), (-1, 0, 0), (0, 0, 1)],  # R
            [(-1, 0, 0), (0, -1, 0), (0, 0, 1)],  # B
            [(0, -1, 0), (1, 0, 0), (0, 0, 1)],  # L
            [(0, 0, -1), (0, 1, 0), (1, 0, 0)],  # U
            [(0, 0, 1), (0, 1, 0), (-1, 0, 0)],  # D
        ]
    )
    offsets = 2 * (np.arange(face_width) + 0.5) / face_width - 1
    u, v = np.meshgrid(offsets, offsets)
    center, right, down = (frames[:, i, None, None, :] for i in range(3))
    rays = center + u[..., None] * right + v[..., None] * down
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    signed_angles = angles if z_down else angles * (1, -1, -1)
    rotation = Rotation.from_euler("xyz", signed_angles).as_matrix()
    expected = (np.einsum("bij,fhwj->bfihw", rotation, rays) + 1) / 2

    if cube_format == "horizon":
        faces = np.stack(np.split(np.asarray(cube), 6, axis=-1), axis=1)
    elif cube_format == "dice":
        positions = [(1, 1), (1, 2), (1, 3), (1, 0), (0, 1), (2, 1)]
        faces = np.stack(
            [
                np.asarray(cube)[
                    ...,
                    row * face_width : (row + 1) * face_width,
                    col * face_width : (col + 1) * face_width,
                ]
                for row, col in positions
            ],
            axis=1,
        )
    else:
        face_order = "FRBLUD" if cube_format == "dict" else range(6)
        faces = np.stack(
            [
                np.stack([np.asarray(item[face]) for face in face_order])
                for item in cube
            ]
        )

    # Allow one source pixel of color error at the native sampler's seam.
    np.testing.assert_allclose(faces, expected, atol=np.pi / width, rtol=0)
