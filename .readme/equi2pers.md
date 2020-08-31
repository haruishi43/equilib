## equi2pers

equirectangular to perspective transformation

I try to keep a common api that can be used in both `c++`, `numpy`, and `pytorch`.

```Python
class SomeEqui2Pers(BaseEqui2Pers):
    def __init__(self, w_pers, h_pers, fov_x):
        ...
    def run(self, equi, rot, **kwargs):
        ...
```

### Numpy

```Python
from equilib.equi2pers import NumpyEqui2Pers
```

### PyTorch

```Python
from equilib.equi2pers import TorchEqui2Pers
```

### C++

__WIP__

### TODO:

- [ ] Crop is slightly different `numpy` and `torch`
- [ ] equi2Pers for `c++`/`cuda` is still messy, WIP
- [ ] equi2Pers outputs for `numpy` and `torch` differs a little bit. Need to figure out why this happens. The outputs are the same regardless of the sampling method, so it must be the preprocessing (where the rotation matrix is set).
