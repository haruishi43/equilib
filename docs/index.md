# equilib

**Processing Equirectangular Images with Python**

`equilib` is a library for processing equirectangular (360°) images that runs on
pure Python.

- Developed for Python `>=3.9`.
- Compatible with `cuda` tensors for faster processing.
- No dependencies beyond `numpy` and `torch`.
- Extra functionality: rotation-matrix creation, batched processing, and
  automatic input-type detection.
- Works with various input modalities.
- Highly modular.

## Transforms

`equilib` provides several transforms between equirectangular and other image
representations. Each transform ships both a `class` API and a `func` API.

| Transform | Description |
| --- | --- |
| [`Cube2Equi`][equilib.Cube2Equi] / `cube2equi` | cubemap → equirectangular |
| [`Equi2Cube`][equilib.Equi2Cube] / `equi2cube` | equirectangular → cubemap |
| [`Equi2Equi`][equilib.Equi2Equi] / `equi2equi` | equirectangular → equirectangular |
| [`Equi2Pers`][equilib.Equi2Pers] / `equi2pers` | equirectangular → perspective |
| [`Pers2Equi`][equilib.Pers2Equi] / `pers2equi` | perspective → equirectangular |

See [Usage](usage.md) for examples and the [API Reference](api.md) for the full
signatures.

## Citation

If you found this module helpful to your project, please cite this repository:

```bibtex
@software{pyequilib2021github,
  author = {Haruya Ishikawa},
  title = {PyEquilib: Processing Equirectangular Images with Python},
  url = {http://github.com/haruishi43/equilib},
  version = {0.6.0},
  year = {2021},
}
```

## Acknowledgements

- [py360convert](https://github.com/sunset1995/py360convert)
- [Perspective-and-Equirectangular](https://github.com/timy90022/Perspective-and-Equirectangular)
