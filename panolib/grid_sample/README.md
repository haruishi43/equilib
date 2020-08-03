# Grid Sampling Functions

Fast grid sampling methods are needed for gathering pixels from panorama images to convert into perspective (or transform to another panorama) image.

Implemented for two different libraries:
- `numpy_func`
- `torch_func`

You can import them using:

```Python
from panolib.grid_sample import (
    numpy_func,
    torch_fund,
)
```

## Numpy

- `faster`: faster implementation by removing iteration
- `naive`: slow implementation (serves no purpose really)

## PyTorch

- `torch`: default `torch.grid_sample` function with wrapper
- `custom`: custom implementation