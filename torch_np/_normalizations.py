""" "Normalize" arguments: convert array_likes to tensors, dtypes to torch dtypes and so on.
"""
from __future__ import annotations

import functools
import operator
import typing

import torch

from . import _helpers

ArrayLike = typing.TypeVar("ArrayLike")
DTypeLike = typing.TypeVar("DTypeLike")
SubokLike = typing.TypeVar("SubokLike")
AxisLike = typing.TypeVar("AxisLike")
NDArray = typing.TypeVar("NDarray")

# OutArray is to annotate the out= array argument.
#
# This one is special is several respects:
# First, It needs to be an NDArray, and we need to preserve the `result is out`
# semantics. Therefore, we cannot just extract the Tensor from the out array.
# So we never pass the out array to implementer functions and handle it in the
# `normalizer` below.
# Second, the out= argument can be either keyword or positional argument, and
# as a positional arg, it can be anywhere in the signature.
# To handle all this, we define a special `OutArray` annotation and dispatch on it.
#
OutArray = typing.TypeVar("OutArray")


import inspect

from . import _dtypes


def normalize_array_like(x, name=None):
    from ._ndarray import asarray

    return asarray(x).tensor


def normalize_optional_array_like(x, name=None):
    # This explicit normalizer is needed because otherwise normalize_array_like
    # does not run for a parameter annotated as Optional[ArrayLike]
    return None if x is None else normalize_array_like(x, name)


def normalize_seq_array_like(x, name=None):
    return tuple(normalize_array_like(value) for value in x)


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
    # check the arg is an ndarray, extract its tensor attribute
    if arg is None:
        return arg

    from ._ndarray import ndarray

    if not isinstance(arg, ndarray):
        raise TypeError(f"'{name}' must be an array")
    return arg.tensor


def normalize_outarray(arg, name=None):
    # almost normalize_ndarray, only return the array, not its tensor
    if arg is None:
        return arg

    from ._ndarray import ndarray

    if not isinstance(arg, ndarray):
        raise TypeError("'out' must be an array")
    return arg


normalizers = {
    "ArrayLike": normalize_array_like,
    "Optional[ArrayLike]": normalize_optional_array_like,
    "Sequence[ArrayLike]": normalize_seq_array_like,
    "Optional[NDArray]": normalize_ndarray,
    "Optional[OutArray]": normalize_outarray,
    "NDArray": normalize_ndarray,
    "DTypeLike": normalize_dtype,
    "SubokLike": normalize_subok_like,
    "AxisLike": normalize_axis_like,
}


def maybe_normalize(arg, parm):
    """Normalize arg if a normalizer is registred."""
    normalizer = normalizers.get(parm.annotation, None)
    return normalizer(arg, parm.name) if normalizer else arg


# ### Return value helpers ###


def maybe_copy_to(out, result, promote_scalar_result=False):
    # NB: here out is either an ndarray or None
    if out is None:
        return result
    elif isinstance(result, torch.Tensor):
        if result.shape != out.shape:
            can_fit = result.numel() == 1 and out.ndim == 0
            if promote_scalar_result and can_fit:
                result = result.squeeze()
            else:
                raise ValueError(
                    f"Bad size of the out array: out.shape = {out.shape}"
                    f" while result.shape = {result.shape}."
                )
        out.tensor.copy_(result)
        return out
    elif isinstance(result, (tuple, list)):
        return type(result)(
            maybe_copy_to(o, r, promote_scalar_result) for o, r in zip(out, result)
        )
    else:
        assert False  # We should never hit this path


def wrap_tensors(result):
    from ._ndarray import ndarray

    if isinstance(result, torch.Tensor):
        return ndarray(result)
    elif isinstance(result, (tuple, list)):
        result = type(result)(
            ndarray(x) if isinstance(x, torch.Tensor) else x for x in result
        )
    return result


def array_or_scalar(values, py_type=float, return_scalar=False):
    if return_scalar:
        return py_type(values.item())
    else:
        from ._ndarray import ndarray

        return ndarray(values)


# ### The main decorator to normalize arguments / postprocess the output ###


def normalizer(_func=None, *, promote_scalar_result=False):
    def normalizer_inner(func):
        @functools.wraps(func)
        def wrapped(*args, **kwds):
            sig = inspect.signature(func)
            params = sig.parameters
            first_param = next(iter(params.values()))
            # NumPy's API does not have positional args before variadic positional args
            if first_param.kind == inspect.Parameter.VAR_POSITIONAL:
                args = [maybe_normalize(arg, first_param) for arg in args]
            else:
                # NB: extra unknown arguments: pass through, will raise in func(*args) below
                args = (
                    tuple(
                        maybe_normalize(arg, parm)
                        for arg, parm in zip(args, params.values())
                    )
                    + args[len(params.values()) :]
                )

            kwds = {
                name: maybe_normalize(arg, params[name]) if name in params else arg
                for name, arg in kwds.items()
            }
            result = func(*args, **kwds)

            if "out" in params:
                out = sig.bind(*args, **kwds).arguments.get("out")
                result = maybe_copy_to(out, result, promote_scalar_result)
            result = wrap_tensors(result)

            return result

        return wrapped

    if _func is None:
        return normalizer_inner
    else:
        return normalizer_inner(_func)
