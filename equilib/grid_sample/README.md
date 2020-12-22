# Grid Sampling Functions

Fast grid sampling methods are needed for gathering pixels from equirectangular images to convert into perspective (or transform to another equirectangular) image.

Implemented for two different libraries:
- `numpy_func`
- `torch_func`

You can import them using:

```Python
from equilib.grid_sample import (
    numpy_func,
    torch_fund,
)
```

## Numpy

- `default`: faster implementation by removing iteration
- `naive`: slow implementation (serves no purpose really)

## PyTorch

- `default`: default `torch.grid_sample` function with wrapper
- `custom`: custom implementation
