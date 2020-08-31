## equi2cube

equirectangular to cubemap transformation

```Python
class SomeEqui2Cube(BaseEqui2Cube):
    def __init__(self, w_face: int):
        ...
    def run(self, equi, rot, cube_format, **kwargs):
        ...
```

### Numpy

```Python
from equilib.equi2cube import NumpyEqui2Cube
```

### PyTorch

```Python
from equilib.equi2cube import TorchEqui2Cube
```

### TODO:

- [x] Implement `numpy`
- [x] Implement `torch`
- [x] Implement `torch` with batches
- [x] Fix rotation axis
- [ ] Implement `c++` with `cuda`