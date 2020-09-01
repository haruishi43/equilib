#!/usr/bin/env python3

import torch


def get_device(a: torch.Tensor) -> torch.device:
    r"""Get device of a Tensor"""
    return torch.device(a.get_device() if a.get_device() > 0 else "cpu")
