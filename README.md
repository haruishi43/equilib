# panolib

A library for processing panorama (equirectangular) image

## Grid Sampling

In order to process panorama images fast, whether to crop perspective images from the panorama image, the library takes advantage from grid sampling techniques.
In this library we implement varieties of methods using `c++`, `numpy`, and `pytorch`.

See [here](panolib/grid_sample/README.md) for more info on implementations.

## Pano2Pers Numpy

## Pano2Pers Torch

## Panorama Image

Any image size with `2:1` ratio

- `2160s`: `3840x1920`
- `2880s`: `5760x2880`

