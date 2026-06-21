#!/usr/bin/env python3

"""Bounded cache for the rotation-invariant grid prep used by the `class`
transform APIs (`Equi2Pers`, `Equi2Cube`, `Equi2Equi`, `Pers2Equi`).

The cache itself is instance-scoped (each transform object owns its `_cache`
dict), so it is freed when the instance is. This helper just bounds it so a
long-lived instance fed many distinct input shapes/dtypes cannot grow without
limit.
"""

from typing import Any, Callable, Dict, Hashable, Optional

__all__ = ["cached_grid", "GRID_CACHE_MAXSIZE"]

# Max distinct (shape, dtype, ...) entries kept per instance. The intended
# "fixed config, same shape" usage needs 1; this only caps pathological growth.
GRID_CACHE_MAXSIZE = 16


def cached_grid(
    cache: Optional[Dict[Hashable, Any]],
    key: Hashable,
    build: Callable[[], Any],
) -> Any:
    """Return `cache[key]`, computing it with `build()` on a miss.

    `cache is None` disables caching (the func API path always recomputes).
    On insert, the oldest entry is evicted once `GRID_CACHE_MAXSIZE` would be
    exceeded (FIFO -- dicts preserve insertion order).
    """
    if cache is None:
        return build()
    if key in cache:
        return cache[key]
    value = build()
    if len(cache) >= GRID_CACHE_MAXSIZE:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value
