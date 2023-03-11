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

# annotate e.g. atleast_1d(*arys)
UnpackedSeqArrayLike = typing.TypeVar("UnpackedSeqArrayLike")


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
    UnpackedSeqArrayLike: normalize_seq_array_like,  # cf handling in normalize
    Optional[NDArray]: normalize_ndarray,
    DTypeLike: normalize_dtype,
    SubokLike: normalize_subok_like,
    AxisLike: normalize_axis_like,
}

import functools

_sentinel = object()


def normalize_this(arg, parm, return_on_failure=_sentinel):
    """Normalize arg if a normalizer is registred."""
    normalizer = normalizers.get(parm.annotation, None)
    if normalizer:
        try:
            return normalizer(arg)
        except Exception as exc:
            if return_on_failure is not _sentinel:
                return return_on_failure
            else:
                raise exc from None
    else:
        # untyped arguments pass through
        return arg


def normalizer(_func=None, *, return_on_failure=_sentinel):
    def normalizer_inner(func):
        @functools.wraps(func)
        def wrapped(*args, **kwds):
            sig = inspect.signature(func)

            # first, check for *args in positional parameters. Case in point:
            # atleast_1d(*arys: UnpackedSequenceArrayLike)
            # if found,  consume all args into a tuple to normalize as a whole
            for j, param in enumerate(sig.parameters.values()):
                if param.annotation == UnpackedSeqArrayLike:
                    if j == 0:
                        args = (args,)
                    else:
                        # args = args[:j] + (args[j:],) would likely work
                        # not present in numpy codebase, so do not bother just yet.
                        # NB: branching on j ==0 is to avoid the empty tuple, args[:j]
                        raise NotImplementedError
                    break

            # loop over positional parameters and actual arguments
            lst, dct = [], {}
            for arg, (name, parm) in zip(args, sig.parameters.items()):
                print(arg, name, parm.annotation)
                lst.append(normalize_this(arg, parm, return_on_failure))

            # normalize keyword arguments
            for name, arg in kwds.items():
                if not name in sig.parameters:
                    # unknown kwarg, bail out
                    raise TypeError(
                        f"{func.__name__}() got an unexpected keyword argument '{name}'."
                    )

                print("kw: ", name, sig.parameters[name].annotation)
                parm = sig.parameters[name]
                dct[name] = normalize_this(arg, parm, return_on_failure)

            ba = sig.bind(*lst, **dct)
            ba.apply_defaults()

            # Now that all parameters have been consumed, check:
            # Anything that has not been bound is unexpected positional arg => raise.
            # If there are too few actual arguments, this fill fail in func(*ba.args) below
            if len(args) > len(ba.args):
                raise TypeError(
                    f"{func.__name__}() takes {len(ba.args)} positional argument but {len(args)} were given."
                )

            # TODO:
            # 1. [LOOKS OK] kw-only parameters : see vstack
            # 2. [LOOKS OK] extra unknown args -- error out : nonzero([2, 0, 3], oops=42)
            # 3. [LOOKS OK] optional (tensor_or_none) : untyped => pass through
            # 4. [LOOKS OK] DTypeLike : positional or kw
            # 5. axes : live in _impl or in types? several ways of handling them
            # 6. [OK, NOT HERE] keepdims : peel off, postprocess
            # 7. OutLike : normal & keyword-only, peel off, postprocess
            # 8. [LOOKS OK] *args
            # 9. [LOOKS OK] consolidate normalizations (_funcs, _wrapper)
            # 10. [LOOKS OK] consolidate decorators (_{unary,binary}_ufuncs)
            # 11. out= arg : validate it's an ndarray

            # finally, pass normalized arguments through
            result = func(*ba.args, **ba.kwargs)
            return result

        return wrapped

    if _func is None:
        return normalizer_inner
    else:
        return normalizer_inner(_func)
