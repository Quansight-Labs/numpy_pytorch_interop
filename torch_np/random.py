"""Wrapper to mimic (parts of) np.random API surface.

NumPy has strict guarantees on reproducibility etc; here we don't give any.

Q: default dtype is float64 in numpy

"""
from math import sqrt

import torch

from . import asarray
from ._detail._scalar_types import default_float_type as _default_float_type
from ._detail import _util

_default_dtype = _default_float_type.torch_dtype

__all__ = ["seed", "random_sample", "sample", "random", "rand", "randn", "normal"]


def seed(seed=None):
    if seed is not None:
        torch.random.manual_seed()


def random_sample(size=None):
    if size is None:
        values = torch.rand(())
        return float(values)
    else:
        values = torch.rand(size).to(_default_dtype)
        return asarray(values)


def rand(*size):
    return random_sample(size)


sample = random_sample
random = random_sample


def uniform(low=0.0, high=1.0, size=None):
    if size is None:
        values = torch.rand(())
        return float(low + (high - low) * values)
    else:
        values = torch.rand(size).to(_default_dtype)
        return asarray(low + (high - low) * values)


def randn(*size):
    if size == ():
        return float(torch.randn(size))
    else:
        values = torch.randn(*size).to(_default_dtypes)
        return asarray(values)


def normal(loc=0.0, scale=1.0, size=None):
    if size is None:
        size = ()
    return loc + scale * randn(*size).to(_default_dtype)


def shuffle(x):
    x_tensor = asarray(x).get()
    perm = torch.randperm(x_tensor.shape[0])
    xp = x_tensor[perm]
    x_tensor.copy_(xp)


def randint(low, high=None, size=None):
    if size is None:
        size = ()
    if not isinstance(size, (tuple, list)):
        size = (size,)
    if high is None:
        low, high = 0, high
    values = torch.randint(low, high, size=size)
    return asarray(values)


def choice(a, size=None, replace=True, p=None):
    # https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
    if isinstance(a, int):
        a_tensor = torch.arange(a)
    else:
        a_tensor = asarray(a).get()

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
        p_tensor = torch.ones_like(a_tensor) / a_tensor.shape[0]
    else:
        p_tensor = asarray(p, dtype=float).get()

    # cf https://github.com/numpy/numpy/blob/main/numpy/random/mtrand.pyx#L973
    atol = sqrt(torch.finfo(torch.float64).eps)
    if abs(p_tensor.sum() - 1.) > atol:
        raise ValueError("probabilities do not sum to 1.")

    # actually sample
    indices = torch.multinomial(p_tensor, num_el, replacement=replace)

    if _util.is_sequence(size):
        indices = indices.reshape(size)

    samples = a_tensor[indices]

    return asarray(samples)
