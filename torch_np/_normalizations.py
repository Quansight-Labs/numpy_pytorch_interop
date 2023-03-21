""" "Normalize" arguments: convert array_likes to tensors, dtypes to torch dtypes and so on.
"""
import operator
import typing
from typing import Optional, Sequence

import torch

from . import _helpers

ArrayLike = typing.TypeVar("ArrayLike")
DTypeLike = typing.TypeVar("DTypeLike")
SubokLike = typing.TypeVar("SubokLike")
AxisLike = typing.TypeVar("AxisLike")
NDArray = typing.TypeVar("NDarray")


import inspect

from . import _dtypes


def normalize_array_like(x, name=None):
    (tensor,) = _helpers.to_tensors(x)
    return tensor


def normalize_optional_array_like(x, name=None):
    # This explicit normalizer is needed because otherwise normalize_array_like
    # does not run for a parameter annotated as Optional[ArrayLike]
    return None if x is None else normalize_array_like(x, name)


def normalize_seq_array_like(x, name=None):
    tensors = _helpers.to_tensors(*x)
    return tensors


def normalize_dtype(dtype, name=None):
    # cf _decorators.dtype_to_torch
    torch_dtype = None
    if dtype is not None:
        dtype = _dtypes.dtype(dtype)
        torch_dtype = dtype.torch_dtype
    return torch_dtype


def normalize_subok_like(arg, name):
    if arg:
        raise ValueError(f"'{name}' parameter is not supported.")


def normalize_axis_like(arg, name=None):
    from ._ndarray import ndarray

    if isinstance(arg, ndarray):
        arg = operator.index(arg)
    return arg


def normalize_ndarray(arg, name=None):
    if arg is None:
        return arg

    from ._ndarray import ndarray

    if not isinstance(arg, ndarray):
        raise TypeError("'out' must be an array")
    return arg


normalizers = {
    ArrayLike: normalize_array_like,
    Optional[ArrayLike]: normalize_optional_array_like,
    Sequence[ArrayLike]: normalize_seq_array_like,
    Optional[NDArray]: normalize_ndarray,
    DTypeLike: normalize_dtype,
    SubokLike: normalize_subok_like,
    AxisLike: normalize_axis_like,
}

import functools


def normalize_this(arg, parm):
    """Normalize arg if a normalizer is registred."""
    normalizer = normalizers.get(parm.annotation, None)
    if normalizer:
        return normalizer(arg)
    else:
        # untyped arguments pass through
        return arg


def normalizer(func):
    @functools.wraps(func)
    def wrapped(*args, **kwds):
        sig = inspect.signature(func)
        sp = dict(sig.parameters)

        # check for *args. If detected, duplicate the correspoding parameter
        # to have len(args) annotations for each element of *args.
        for j, param in enumerate(sp.values()):
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                sp.pop(param.name)
                variadic = {param.name + str(i): param for i in range(len(args))}
                variadic.update(sp)
                sp = variadic
                break

        # normalize positional and keyword arguments
        # NB: extra unknown arguments: pass through, will raise in func(*lst) below
        lst = [normalize_this(arg, parm) for arg, parm in zip(args, sp.values())]
        lst += args[len(lst) :]

        dct = {
            name: normalize_this(arg, sp[name]) if name in sp else arg
            for name, arg in kwds.items()
        }

        result = func(*lst, **dct)

        return result

    return wrapped
