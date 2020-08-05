# equilib

A library for processing equirectangular image

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

## Grid Sampling

In order to process equirectangular images fast, whether to crop perspective images from the equirectangular image, the library takes advantage from grid sampling techniques.
In this library we implement varieties of methods using `c++`, `numpy`, and `pytorch`.
This part of the code needs `cuda` acceleration because grid sampling is parallizable.
For `c++` and `pytorch`, I tried to take advantage of `cuda`.
For `numpy`, I implemented `naive` and `faster` approaches for learning purposes.
Developing _faster_ `c++` and `pytorch` approaches are __WIP__.

See [here](equilib/grid_sample/README.md) for more info on implementations.

## equi2Pers

equirectangular to perspective transformation

I try to keep a common api that can be used in both `c++`, `numpy`, and `pytorch`.

```Python
class SomeEqui2Pers(BaseEqui2Pers):
    def __init__(self, w_pers, h_pers, fov_x):
        ...
    def __call__(self, equi, rot):
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

- [ ] equi2Pers for `c++`/`cuda` is still messy, WIP
- [ ] equi2Pers outputs for `numpy` and `torch` differs a little bit. Need to figure out why this happens. The outputs are the same regardless of the sampling method, so it must be the preprocessing (where the rotation matrix is set).


## equi2equi

equirectangular to equirectangular transformation

I try to keep a common api that can be used in both `c++`, `numpy`, and `pytorch`.

```Python
class SomeEqui2Equi(BaseEqui2Equi):
    def __init__(self, h_out: int, w_out: int):
        ...
    def __call__(self, src, rot):
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
- [ ] Fix rotation axis
- [ ] Implement `c++` with `cuda`


## Develop:

Test files for `equilib` is included under `tests`.

Running tests:
```Bash
pytest tests
```