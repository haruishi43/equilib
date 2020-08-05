# panolib

A library for processing panorama (equirectangular) image

## Installation:

```Bash
git clone --recursive https://github.com/Toraudonn/panolib.git
cd panolib

pip install -r requirements.txt

python setup.py develop
```


## What is a panorama image

<img src="data/pano.jpg" alt="pano" width="480"/>

Any image size with `2:1` ratio that captures 360 degree field of view.

Common image sizes:
- `2160s`: `3840x1920`
- `2880s`: `5760x2880`

## Grid Sampling

In order to process panorama images fast, whether to crop perspective images from the panorama image, the library takes advantage from grid sampling techniques.
In this library we implement varieties of methods using `c++`, `numpy`, and `pytorch`.
This part of the code needs `cuda` acceleration because grid sampling is parallizable.
For `c++` and `pytorch`, I tried to take advantage of `cuda`.
For `numpy`, I implemented `naive` and `faster` approaches for learning purposes.
Developing _faster_ `c++` and `pytorch` approaches are __WIP__.

See [here](panolib/grid_sample/README.md) for more info on implementations.

## Pano2Pers

Panorama to perspective transformation

I try to keep a common api that can be used in both `c++`, `numpy`, and `pytorch`.

```Python
class SomePano2Pers(BasePano2Pers):
    def __init__(self, ...):
        ...
    def __call__(self, ...):
        ...
```

### C++

__WIP__

### Numpy

```Python
from panolib.pano2pers import NumpyPano2Pers
```

### PyTorch

```Python
from panolib.pano2pers import TorchPano2Pers
```

### TODO:

- [ ] Pano2Pers for `c++`/`cuda` is still messy, WIP
- [ ] Pano2Pers outputs for `numpy` and `torch` differs a little bit. Need to figure out why this happens. The outputs are the same regardless of the sampling method, so it must be the preprocessing (where the rotation matrix is set).


## Pano2Pano


### TODO:

- [ ] Implement `numpy`
- [ ] Implement `torch`
- [ ] Implement `torch` with batches
- [ ] Implement `c++` with `cuda`


## Develop:

Test files for `panolib` is included under `tests`.

Running tests:
```Bash
pytest tests
```