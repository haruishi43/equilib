# Grid Sampling

To process equirectangular images quickly, `equilib` relies on grid sampling
techniques. The goal of the project is to minimize external dependencies and
take advantage of `cuda` and batch processing with `torch` (and, eventually,
`c++`) for fast processing.

The library implements a variety of methods in `numpy` and `torch`:

- For `torch`, the built-in `torch.nn.functional.grid_sample` is fast and
  reliable. A pure-`torch` implementation is also provided and is highly
  customizable (though not necessarily as fast as the native function).
- For `numpy`, the implementations are faster than `scipy` and more robust than
  `cv2.remap`, and are just as customizable as the `torch` version.

It is also possible to pass `scipy.ndimage.map_coordinates` or `cv2.remap` as
the sampling function via the `override_func` argument of `grid_sample`.

## Notes

- By default, the `numpy` backend uses the pure-`numpy` implementation. Override
  it with `scipy` or `cv2` via `override_func`.
- By default, the `torch` backend uses the official `grid_sample`.
- Benchmarking scripts live under
  [`benchmarks/`](https://github.com/haruishi43/equilib/tree/master/benchmarks).
  For example, `benchmarks/equi2pers/numpy_run_baselines.py` benchmarks the
  `numpy` `equi2pers` path against `scipy` and `cv2`.

A `c++` with `cuda` implementation is **work in progress**.
