# panolib

A library for processing panorama (equirectangular) image

## What is a panorama image

![](data/pano.jpg =320x)

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
Developing a _faster_ `c++` and `pytorch` approach is __WIP__.

See [here](panolib/grid_sample/README.md) for more info on implementations.

## Pano2Pers

Panorama to perspective transformation
I try to keep a common api that can be used in both `c++`, `numpy`, and `pytorch`.

### C++

### Numpy

### PyTorch