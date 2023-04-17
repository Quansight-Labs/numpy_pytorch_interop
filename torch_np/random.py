"""Wrapper to mimic (parts of) np.random API surface.

NumPy has strict guarantees on reproducibility etc; here we don't give any.

Q: default dtype is float64 in numpy

"""
from __future__ import annotations

from math import sqrt
from typing import Optional

import torch

from . import _helpers
from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, array_or_scalar, normalizer

_default_dtype = _dtypes_impl.default_float_dtype

__all__ = [
    "seed",
    "random_sample",
    "sample",
    "random",
    "rand",
    "randn",
    "normal",
    "choice",
    "randint",
    "shuffle",
    "uniform",
]


def seed(seed=None):
    if seed is not None:
        torch.random.manual_seed(seed)


def random_sample(size=None):
    if size is None:
        size = ()
    values = torch.empty(size, dtype=_default_dtype).uniform_()
    return array_or_scalar(values, return_scalar=size is None)


def rand(*size):
    return random_sample(size)


sample = random_sample
random = random_sample


def uniform(low=0.0, high=1.0, size=None):
    if size is None:
        size = ()
    values = torch.empty(size, dtype=_default_dtype).uniform_(low, high)
    return array_or_scalar(values, return_scalar=size is None)


def randn(*size):
    values = torch.randn(size, dtype=_default_dtype)
    return array_or_scalar(values, return_scalar=size is None)


def normal(loc=0.0, scale=1.0, size=None):
    if size is None:
        size = ()
    values = torch.empty(size, dtype=_default_dtype).normal_(loc, scale)
    return array_or_scalar(values, return_scalar=size is None)


@normalizer
def shuffle(x: ArrayLike):
    perm = torch.randperm(x.shape[0])
    xp = x[perm]
    x.copy_(xp)


def randint(low, high=None, size=None):
    if size is None:
        size = ()
    if not isinstance(size, (tuple, list)):
        size = (size,)
    if high is None:
        low, high = 0, low
    values = torch.randint(low, high, size=size)
    return array_or_scalar(values, int, return_scalar=size is None)


@normalizer
def choice(a: ArrayLike, size=None, replace=True, p: Optional[ArrayLike] = None):

    # https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
    if a.numel() == 1:
        a = torch.arange(a)

    # TODO: check a.dtype is integer -- cf np.random.choice(3.4) which raises

    # number of draws
    if size is None:
        num_el = 1
    elif _util.is_sequence(size):
        num_el = 1
        for el in size:
            num_el *= el
    else:
        num_el = size

    # prepare the probabilities
    if p is None:
        p = torch.ones_like(a) / a.shape[0]

    # cf https://github.com/numpy/numpy/blob/main/numpy/random/mtrand.pyx#L973
    atol = sqrt(torch.finfo(torch.float64).eps)
    if abs(p.sum() - 1.0) > atol:
        raise ValueError("probabilities do not sum to 1.")

    # actually sample
    indices = torch.multinomial(p, num_el, replacement=replace)

    if _util.is_sequence(size):
        indices = indices.reshape(size)

    samples = a[indices]

    return samples
