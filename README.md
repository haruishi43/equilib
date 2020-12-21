<h1 align="center">
  equilib
</h1>

<h4 align="center">
  Processing Equirectangular Images with Python
</h4>

<div align="center">
  <a href="https://pypi.org/project/pyequilib/0.1.0/"><img src="https://badge.fury.io/py/pyequilib.svg"></a>
  <a href="https://pypi.org/project/equilib"><img src="https://img.shields.io/pypi/pyversions/pyequilib"></a>
  <a href="https://github.com/haruishi43/equilib/actions"><img src="https://github.com/haruishi43/equilib/workflows/ci/badge.svg"></a>
  <a href="https://github.com/haruishi43/equilib/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/haruishi43/equilib"></a>
</div>

<img src=".img/equilib.png" alt="equilib" width="720"/>

- A library for processing equirectangular image that runs on Python.
- Developed using `numpy`, `pytorch`, and `c++`.
- Able to use GPU for faster processing.
- No need for other dependencies except for `numpy` and `pytorch`.
- Added functionality like rotation matrix and batched processing.
- Highly modular

## Installation:

Prerequisites:
- Python (>=3.5)
- Pytorch

```Bash
pip install pyequilib
```

For developing, use:

```Bash
git clone --recursive https://github.com/Toraudonn/equilib.git
cd equilib

pip install -r requirements.txt

python setup.py develop
```

## Basic Usage:

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

The API for each module is pretty similar with other conversions.
First, you initialize the module (the naming of the module is `Numpy`, `Torch` or `CPP` + function name `[Equi2Pers, Equi2Equi, Equi2Cube, Cube2Equi]`).
Lastly, input the image(s) (and rotations if needed) to the function to obtain the transformed output.
For more information about how each functions work, take a look in [.readme](.readme/) or go through example codes in the `tests` or `scripts`.


### Coordinate System:

Right-handed rule XYZ global coordinate system. `x-axis` faces forward and `z-axis` faces up.
- `roll`: counter-clockwise rotation about the `x-axis`
- `pitch`: counter-clockwise rotation about the `y-axis`
- `yaw`: counter-clockwise rotation about the `z-axis`

See demo scripts under `scripts`.

## Equirectangular image

<img src="data/equi.jpg" alt="equi" width="480"/>

Any image size with `2:1` ratio that captures 360 degree field of view.

Common image sizes:

- `2160s`: `3840x1920`
- `2880s`: `5760x2880`

## Grid Sampling

In order to process equirectangular images fast, whether to crop perspective images from the equirectangular image, the library takes advantage from grid sampling techniques.
There are some sampling techniques already implemented such as `scipy.ndimage.map_coordiantes` and `cv2.remap`.
The goal of this project was reduce these dependencies and use `cuda` processing with `pytorch` and `c++` for a faster processing of equirectangular images.
There was not many projects online for these purposes.
In this library we implement varieties of methods using `c++`, `numpy`, and `pytorch`.
This part of the code needs `cuda` acceleration because grid sampling is parallizable.
For `c++` and `pytorch`, I tried to take advantage of `cuda`.
For `numpy`, I implemented `naive` and `faster` approaches for learning purposes.
Developing _faster_ `c++` and `pytorch` approaches are __WIP__.

See [here](equilib/grid_sample/README.md) for more info on implementations.

## Develop:

Test files for `equilib` is included under `tests`.

Running tests:
```Bash
pytest tests
```

### TODO:

- [ ] Add graphs and statistics for speed improvements


## Acknowledgements:

- [py360convert](https://github.com/sunset1995/py360convert)
- [Perspective-and-Equirectangular](https://github.com/timy90022/Perspective-and-Equirectangular/tree/master/lib)
