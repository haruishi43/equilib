# equilib

**Processing Equirectangular Images with Python**

![equilib](https://raw.githubusercontent.com/haruishi43/equilib/master/.img/equilib.png)

`equilib` is a library for processing equirectangular (360°) images in Python.

- Pure Python, with `numpy` and `torch` as the only runtime dependencies.
- Runs on CPU and CUDA tensors, with batched and mixed-precision processing.
- Automatic input-type detection (`numpy.ndarray` or `torch.Tensor`).
- Extras such as rotation-matrix creation and a customizable grid sampler.
- Highly modular and extensible.

## Quick start

```bash
pip install pyequilib
```

See [Installation](installation.md) for development setup, then [Usage](usage.md)
for examples.

## Transforms

`equilib` provides transforms between equirectangular, cubemap, and perspective
images. Each transform ships both a `class` API and a `func` API.

| Transform | Description |
| --- | --- |
| [`Cube2Equi`][equilib.Cube2Equi] / `cube2equi` | cubemap → equirectangular |
| [`Equi2Cube`][equilib.Equi2Cube] / `equi2cube` | equirectangular → cubemap |
| [`Equi2Equi`][equilib.Equi2Equi] / `equi2equi` | equirectangular → equirectangular |
| [`Equi2Pers`][equilib.Equi2Pers] / `equi2pers` | equirectangular → perspective |
| [`Pers2Equi`][equilib.Pers2Equi] / `pers2equi` | perspective → equirectangular |

See the [API Reference](api.md) for the full signatures.

## Citation

If you found this module helpful to your project, please cite this repository:

```bibtex
@software{pyequilib2021github,
  author = {Haruya Ishikawa},
  title = {PyEquilib: Processing Equirectangular Images with Python},
  url = {https://github.com/haruishi43/equilib},
  version = {0.6.0},
  year = {2021},
}
```

## Acknowledgements

- [py360convert](https://github.com/sunset1995/py360convert)
- [Perspective-and-Equirectangular](https://github.com/timy90022/Perspective-and-Equirectangular)
