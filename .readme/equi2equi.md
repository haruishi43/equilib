## equi2equi

equirectangular to equirectangular transformation

I try to keep a common api that can be used in both `c++`, `numpy`, and `pytorch`.

```Python
class SomeEqui2Equi(BaseEqui2Equi):
    def __init__(self, h_out: int, w_out: int):
        ...
    def run(self, src, rot, **kwargs):
        ...
```

### Numpy

```Python
from equilib.equi2equi import NumpyEqui2Equi
```

### PyTorch

```Python
from equilib.equi2equi import TorchEqui2Equi
```

### TODO:

- [x] Implement `numpy`
- [x] Implement `torch`
- [x] Implement `torch` with batches
- [x] Fix rotation axis
- [ ] Implement `c++` with `cuda`
