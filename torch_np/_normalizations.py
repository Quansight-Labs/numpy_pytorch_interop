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


def normalize_subok_like(arg, name="subok"):
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

_sentinel = object()


def maybe_normalize(arg, parm, return_on_failure=_sentinel):
    """Normalize arg if a normalizer is registred."""
    normalizer = normalizers.get(parm.annotation, None)
    try:
        return normalizer(arg, parm.name) if normalizer else arg
    except Exception as exc:
        if return_on_failure is not _sentinel:
            return return_on_failure
        else:
            raise exc from None


def normalizer(_func=None, *, return_on_failure=_sentinel):
    def normalizer_inner(func):
        @functools.wraps(func)
        def wrapped(*args, **kwds):
            params = inspect.signature(func).parameters
            first_param = next(iter(params.values()))
            # NumPy's API does not have positional args before variadic positional args
            if first_param.kind == inspect.Parameter.VAR_POSITIONAL:
                args = [
                    maybe_normalize(arg, first_param, return_on_failure) for arg in args
                ]
            else:
                # NB: extra unknown arguments: pass through, will raise in func(*args) below
                args = (
                    tuple(
                        maybe_normalize(arg, parm, return_on_failure)
                        for arg, parm in zip(args, params.values())
                    )
                    + args[len(params.values()) :]
                )

            kwds = {
                name: maybe_normalize(arg, params[name]) if name in params else arg
                for name, arg in kwds.items()
            }
            return func(*args, **kwds)

        return wrapped

    if _func is None:
        return normalizer_inner
    else:
        return normalizer_inner(_func)
