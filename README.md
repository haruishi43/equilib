# equilib

<img src=".img/equilib.png" alt="equilib" width="720"/>

- A library for processing equirectangular image that runs on Python.
- Developed using `numpy`, `pytorch`, and `c++`.
- Able to use GPU for faster processing.
- No need for other dependencies except for `numpy` and `pytorch`.
- Added functionality like rotation matrix and batched processing.

## Installation:

```Bash
git clone --recursive https://github.com/Toraudonn/equilib.git
cd equilib

pip install -r requirements.txt

python setup.py develop
```


## What is a equirectangular image

<img src="data/equi.jpg" alt="equi" width="480"/>

Any image size with `2:1` ratio that captures 360 degree field of view.

Common image sizes:
- `2160s`: `3840x1920`
- `2880s`: `5760x2880`

## How to use:

Initialize `equi2pers`:

```Python
import numpy as np
from PIL import Image
from equilib.equi2pers import NumpyEqui2Pers

# Intialize equi2pers
equi2pers = NumpyEqui2Pers(w_pers=640, h_pers=480, fov_x=90)

equi_img = Image.open("./some_image.jpg")
equi_img = np.asarray(equi_img)
equi_img = np.transpose(equi_img, (2, 0, 1))

# rotations
rot = {
    'roll': 0.,
    'pitch': np.pi/4,  # rotate vertical
    'yaw': np.pi/4,  # rotate horizontal
}

# obtain perspective image
pers_img = equi2pers(equi_img, rot, sampling_method="faster")
```

### Coordinate System:

Right-handed rule XYZ global coordinate system. `x-axis` faces forward and `z-axis` faces up.
- `roll`: counter-clockwise rotation about the `x-axis`
- `pitch`: counter-clockwise rotation about the `y-axis`
- `yaw`: counter-clockwise rotation about the `z-axis`

See demo scripts under `scripts`.

## Grid Sampling

In order to process equirectangular images fast, whether to crop perspective images from the equirectangular image, the library takes advantage from grid sampling techniques.
In this library we implement varieties of methods using `c++`, `numpy`, and `pytorch`.
This part of the code needs `cuda` acceleration because grid sampling is parallizable.
For `c++` and `pytorch`, I tried to take advantage of `cuda`.
For `numpy`, I implemented `naive` and `faster` approaches for learning purposes.
Developing _faster_ `c++` and `pytorch` approaches are __WIP__.

See [here](equilib/grid_sample/README.md) for more info on implementations.

## equi2pers

equirectangular to perspective transformation

I try to keep a common api that can be used in both `c++`, `numpy`, and `pytorch`.

```Python
class SomeEqui2Pers(BaseEqui2Pers):
    def __init__(self, w_pers, h_pers, fov_x):
        ...
    def run(self, equi, rot, **kwargs):
        ...
```

### Numpy

```Python
from equilib.equi2pers import NumpyEqui2Pers
```

### PyTorch

```Python
from equilib.equi2pers import TorchEqui2Pers
```

### C++

__WIP__

### TODO:

- [ ] Crop is slightly different `numpy` and `torch`
- [ ] equi2Pers for `c++`/`cuda` is still messy, WIP
- [ ] equi2Pers outputs for `numpy` and `torch` differs a little bit. Need to figure out why this happens. The outputs are the same regardless of the sampling method, so it must be the preprocessing (where the rotation matrix is set).

## equi2equi

equirectangular to equirectangular transformation

I try to keep a common api that can be used in both `c++`, `numpy`, and `pytorch`.

```Python
class SomeEqui2Equi(BaseEqui2Equi):
    def __init__(self, h_out: int, w_out: int):
        ...
    def run(self, src, rot, **kwargs):
        ...
```

### Numpy

```Python
from equilib.equi2equi import NumpyEqui2Equi
```

### PyTorch

```Python
from equilib.equi2equi import TorchEqui2Equi
```

### TODO:

- [x] Implement `numpy`
- [x] Implement `torch`
- [x] Implement `torch` with batches
- [x] Fix rotation axis
- [ ] Implement `c++` with `cuda`

## equi2cube

equirectangular to cubemap transformation

```Python
class SomeEqui2Cube(BaseEqui2Cube):
    def __init__(self, w_face: int):
        ...
    def run(self, equi, rot, cube_format, **kwargs):
        ...
```

### Numpy

```Python
from equilib.equi2cube import NumpyEqui2Cube
```

### PyTorch

```Python
from equilib.equi2cube import TorchEqui2Cube
```

### TODO:

- [x] Implement `numpy`
- [x] Implement `torch`
- [x] Implement `torch` with batches
- [x] Fix rotation axis
- [ ] Implement `c++` with `cuda`

## cube2equi

cubemap to equirectangular transformation

```Python
class SomeCube2Equi(BaseCube2Equi):
    def __init__(self, w_face: int):
        ...
    def run(self, cubemap, cube_format, **kwargs):
        ...
```

## TODO:

- [x] Implement `numpy`
- [ ] Implement `torch`

## Develop:

Test files for `equilib` is included under `tests`.

Running tests:
```Bash
pytest tests
```

### TODO:

- [x] Keeping `.vscode` for my development
- [ ] Document better @ `README`


## Acknowledgements:

- [py360convert](https://github.com/sunset1995/py360convert)
- [Perspective-and-Equirectangular](https://github.com/timy90022/Perspective-and-Equirectangular/tree/master/lib)
