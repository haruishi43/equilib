# Coordinate System

`equilib` uses a **right-handed XYZ global coordinate system**. The `x-axis`
faces forward and the `z-axis` faces up.

- `roll`: counter-clockwise rotation about the `x-axis`
- `pitch`: counter-clockwise rotation about the `y-axis`
- `yaw`: counter-clockwise rotation about the `z-axis`

You can flip the coordinate system so that the `z-axis` faces **down** by passing
`z_down=True` to any transform.

## Equirectangular image

An equirectangular image is any image with a `2:1` aspect ratio that captures a
360° field of view. Common sizes:

- `2160s`: `3840x1920`
- `2880s`: `5760x2880`

See the demo scripts under [`scripts/`](https://github.com/haruishi43/equilib/tree/master/scripts).
