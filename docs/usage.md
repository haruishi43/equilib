# Usage

Every transform exposes two equivalent APIs:

- A **`class` API** (e.g. `Equi2Pers`) that you instantiate once with a fixed
  configuration and call many times.
- A **`func` API** (e.g. `equi2pers`) that takes the configuration on every call.

The `class` API simply calls the `func` API internally, so there is no
behavioral difference between them.

Each API automatically detects the input data type (`numpy.ndarray` or
`torch.Tensor`) and returns the same type. Input images are **channel-first**
with dimensions `BxCxHxW` or `CxHxW`.

## Common arguments

- `rots`: rotation, specified as three angles
  [pitch, yaw, roll](https://simple.wikipedia.org/wiki/Pitch,_yaw,_and_roll) in
  radians.
- `z_down` (`bool`): use a coordinate system with the z-axis pointing down.
  Defaults to `False`.
- `mode` (`str`): interpolation mode. Defaults to `"bilinear"`.
- `clip_output` (`bool`): clip values to the range of the input. Defaults to
  `True`.

## Example: `Equi2Pers` / `equi2pers`

=== "class API"

    ```python
    import numpy as np
    from PIL import Image
    from equilib import Equi2Pers

    # Input equirectangular image
    equi_img = Image.open("./some_image.jpg")
    equi_img = np.asarray(equi_img)
    equi_img = np.transpose(equi_img, (2, 0, 1))  # HWC -> CHW

    rots = {
        "roll": 0.0,
        "pitch": np.pi / 4,  # rotate vertical
        "yaw": np.pi / 4,    # rotate horizontal
    }

    # Initialize equi2pers
    equi2pers = Equi2Pers(
        height=480,
        width=640,
        fov_x=90.0,
        mode="bilinear",
    )

    # obtain perspective image
    pers_img = equi2pers(equi=equi_img, rots=rots)
    ```

=== "func API"

    ```python
    import numpy as np
    from PIL import Image
    from equilib import equi2pers

    # Input equirectangular image
    equi_img = Image.open("./some_image.jpg")
    equi_img = np.asarray(equi_img)
    equi_img = np.transpose(equi_img, (2, 0, 1))  # HWC -> CHW

    rots = {
        "roll": 0.0,
        "pitch": np.pi / 4,  # rotate vertical
        "yaw": np.pi / 4,    # rotate horizontal
    }

    # Run equi2pers
    pers_img = equi2pers(
        equi=equi_img,
        rots=rots,
        height=480,
        width=640,
        fov_x=90.0,
        mode="bilinear",
    )
    ```

## Other transforms

The same pattern applies to every transform. Swap the import and the
configuration arguments:

- `Cube2Equi` / `cube2equi`: cubemap → equirectangular. Output `height` and
  `width` must be divisible by 8.
- `Equi2Cube` / `equi2cube`: equirectangular → cubemap.
- `Equi2Equi` / `equi2equi`: equirectangular → equirectangular (re-projection by
  rotation).
- `Pers2Equi` / `pers2equi`: perspective → equirectangular.

See the [API Reference](api.md) for the exact arguments of each transform.
