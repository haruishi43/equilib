## equi2pers

equirectangular to perspective transformation


### TODO:

- [ ] Crop is slightly different `numpy` and `torch`
- [ ] equi2Pers for `c++`/`cuda` is still messy, WIP
- [ ] equi2Pers outputs for `numpy` and `torch` differs a little bit. Need to figure out why this happens. The outputs are the same regardless of the sampling method, so it must be the preprocessing (where the rotation matrix is set).
