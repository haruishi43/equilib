## cube2equi

cubemap to equirectangular transformation

```Python
class SomeCube2Equi(BaseCube2Equi):
    def __init__(self, w_face: int):
        ...
    def run(self, cubemap, cube_format, **kwargs):
        ...
```

## TODO:

- [x] Implement `numpy`
- [x] Implement `torch`