<h1 align="center">
  equilib
</h1>

<h4 align="center">
  Processing Equirectangular Images with Python
</h4>

<div align="center">
  <a href="https://badge.fury.io/py/pyequilib"><img src="https://badge.fury.io/py/pyequilib.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/pyequilib"><img src="https://img.shields.io/pypi/pyversions/pyequilib" alt="Python versions"></a>
  <a href="https://github.com/haruishi43/equilib/actions"><img src="https://github.com/haruishi43/equilib/workflows/ci/badge.svg" alt="CI"></a>
  <a href="https://github.com/haruishi43/equilib/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/haruishi43/equilib"></a>
  <a href="https://haruishi43.github.io/equilib/"><img alt="Documentation" src="https://img.shields.io/badge/docs-mkdocs-blue"></a>
</div>

<img src="https://raw.githubusercontent.com/haruishi43/equilib/master/docs/img/equilib.png" alt="equilib" width="720"/>

`equilib` is a library for processing equirectangular (360°) images in Python.

- Pure Python, with `numpy` and `torch` as the only runtime dependencies.
- Runs on CPU and CUDA tensors, with batched and mixed-precision processing.
- Automatic input-type detection (`numpy.ndarray` or `torch.Tensor`).
- Extras such as rotation-matrix creation and a customizable grid sampler.
- Highly modular and extensible.

📖 **Full documentation: <https://haruishi43.github.io/equilib/>**

## Installation

Prerequisites:

- Python `>=3.9`
- PyTorch `>=2.8`

```bash
pip install pyequilib
```

## Transforms

`equilib` provides transforms between equirectangular, cubemap, and perspective
images. Each transform ships both a `class` API and a `func` API.

| Transform | Description |
| --- | --- |
| `Cube2Equi` / `cube2equi` | cubemap → equirectangular |
| `Equi2Cube` / `equi2cube` | equirectangular → cubemap |
| `Equi2Equi` / `equi2equi` | equirectangular → equirectangular |
| `Equi2Pers` / `equi2pers` | equirectangular → perspective |
| `Pers2Equi` / `pers2equi` | perspective → equirectangular |

The `class` API instantiates a reusable object configured once; the `func` API
takes the configuration on every call. The `class` API calls the `func` API
internally, so there is no behavioral difference — both are extensible.

Inputs are channel-first (`BxCxHxW` or `CxHxW`); the output type matches the
input. Common arguments shared across transforms:

- `rots`: rotation as three angles [pitch, yaw, roll](https://simple.wikipedia.org/wiki/Pitch,_yaw,_and_roll) in radians.
- `z_down` (`bool`): use a z-axis-down coordinate system. Default `False`.
- `mode` (`str`): interpolation mode. Default `"bilinear"`.
- `clip_output` (`bool`): clip values to the input range. Default `True`.

## Basic usage

Example with `Equi2Pers` / `equi2pers`.

**`class` API**

```python
import numpy as np
from PIL import Image
from equilib import Equi2Pers

# Input equirectangular image (channel-first: HWC -> CHW)
equi_img = np.asarray(Image.open("./some_image.jpg"))
equi_img = np.transpose(equi_img, (2, 0, 1))

rots = {
    "roll": 0.0,
    "pitch": np.pi / 4,  # rotate vertical
    "yaw": np.pi / 4,    # rotate horizontal
}

equi2pers = Equi2Pers(height=480, width=640, fov_x=90.0, mode="bilinear")
pers_img = equi2pers(equi=equi_img, rots=rots)
```

**`func` API**

```python
import numpy as np
from PIL import Image
from equilib import equi2pers

equi_img = np.asarray(Image.open("./some_image.jpg"))
equi_img = np.transpose(equi_img, (2, 0, 1))

rots = {"roll": 0.0, "pitch": np.pi / 4, "yaw": np.pi / 4}

pers_img = equi2pers(
    equi=equi_img,
    rots=rots,
    height=480,
    width=640,
    fov_x=90.0,
    mode="bilinear",
)
```

See the [documentation](https://haruishi43.github.io/equilib/) for every
transform's arguments, or browse the examples under `tests`, `benchmarks`, and
`scripts`.

## Coordinate system

A **right-handed XYZ global coordinate system**: `x-axis` faces forward and
`z-axis` faces up.

- `roll`: counter-clockwise rotation about the `x-axis`
- `pitch`: counter-clockwise rotation about the `y-axis`
- `yaw`: counter-clockwise rotation about the `z-axis`

Pass `z_down=True` to flip the system so the `z-axis` faces down. See more in the
[coordinate system docs](https://haruishi43.github.io/equilib/coordinate-system/).

## Grid sampling

To process equirectangular images quickly, `equilib` relies on grid sampling and
implements its own `numpy` and `torch` backends to minimize dependencies and
exploit `cuda` and batching:

- The `torch` backend uses the built-in `torch.nn.functional.grid_sample` by
  default, with a customizable pure-`torch` implementation also available.
- The `numpy` backend uses a pure-`numpy` implementation that is faster than
  `scipy` and more robust than `cv2.remap`. You can override it with `scipy` or
  `cv2` via the `override_func` argument.

A `c++`/`cuda` implementation is **WIP**. See the
[grid sampling docs](https://haruishi43.github.io/equilib/grid-sampling/) and the
benchmark scripts in
[`benchmarks/`](https://github.com/haruishi43/equilib/tree/master/benchmarks).

## Development

This project uses [uv](https://docs.astral.sh/uv/) and
[Ruff](https://docs.astral.sh/ruff/). Image/video assets are stored with
[Git LFS](https://git-lfs.com/) (`git lfs install` once before cloning).

```bash
git clone https://github.com/haruishi43/equilib.git
cd equilib
uv sync --group dev      # create the venv and install package + dev tools

uv run pytest tests      # run tests
uv run ruff check .      # lint
uv run ruff format .     # format
```

Pull requests and issues are welcome. See
[CONTRIBUTING.md](https://github.com/haruishi43/equilib/blob/master/CONTRIBUTING.md)
for the full workflow, including how releases are published.

## Roadmap

- [ ] `c++`/`cuda` grid sampling
- [ ] More accurate intrinsic matrix using vertical FOV for `equi2pers`
- [ ] Multiprocessing support (slow on `torch.distributed`)

## Citation

If this project was helpful to your work, please cite it:

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
