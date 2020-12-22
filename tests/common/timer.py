#!/usr/bin/env python3

import time


def timer(func):
    r"""Decorator that reports the execution time."""

    def wrap(*args, **kwargs):
        tic = time.perf_counter()
        result = func(*args, **kwargs)
        toc = time.perf_counter()
        print("Func:{}:\t{:0.4f}".format(func.__name__, toc - tic))
        return result

    return wrap
