import operator
import typing
from typing import Optional, Sequence

import torch

from . import _decorators, _helpers
from ._detail import _dtypes_impl, _flips, _util, _reductions
from ._detail import implementations as _impl

################################## normalizations

ArrayLike = typing.TypeVar("ArrayLike")
DTypeLike = typing.TypeVar("DTypeLike")
SubokLike = typing.TypeVar("SubokLike")
AxisLike = typing.TypeVar("AxisLike")

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


normalizers = {
    ArrayLike: normalize_array_like,
    Optional[ArrayLike]: normalize_optional_array_like,
    Sequence[ArrayLike]: normalize_seq_array_like,
    UnpackedSeqArrayLike: normalize_seq_array_like,  # cf handling in normalize
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
            lst.append(normalize_this(arg, parm))

        # normalize keyword arguments
        for name, arg in kwds.items():
            if not name in sig.parameters:
                # unknown kwarg, bail out
                raise TypeError(
                    f"{func.__name__}() got an unexpected keyword argument '{name}'."
                )

            print("kw: ", name, sig.parameters[name].annotation)
            parm = sig.parameters[name]
            dct[name] = normalize_this(arg, parm)

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
        # 6. keepdims : peel off, postprocess
        # 7. OutLike : normal & keyword-only, peel off, postprocess
        # 8. [LOOKS OK] *args

        # finally, pass normalized arguments through
        result = func(*ba.args, **ba.kwargs)
        return result

    return wrapped


##################################


@normalizer
def nonzero(a: ArrayLike):
    result = a.nonzero(as_tuple=True)
    return _helpers.tuple_arrays_from(result)


def argwhere(a):
    (tensor,) = _helpers.to_tensors(a)
    result = torch.argwhere(tensor)
    return _helpers.array_from(result)


@normalizer
def clip(
    a: ArrayLike,
    min: Optional[ArrayLike] = None,
    max: Optional[ArrayLike] = None,
    out=None,
):
    # np.clip requires both a_min and a_max not None, while ndarray.clip allows
    # one of them to be None. Follow the more lax version.
    result = _impl.clip(a, min, max)
    return _helpers.result_or_out(result, out)


@normalizer
def repeat(a: ArrayLike, repeats: ArrayLike, axis=None):
    # XXX: scalar repeats; ArrayLikeOrScalar ?
    result = torch.repeat_interleave(a, repeats, axis)
    return _helpers.array_from(result)


# ### diag et al ###


@normalizer
def diagonal(a: ArrayLike, offset=0, axis1=0, axis2=1):
    result = _impl.diagonal(a, offset, axis1, axis2)
    return _helpers.array_from(result)


@normalizer
def trace(a: ArrayLike, offset=0, axis1=0, axis2=1, dtype: DTypeLike = None, out=None):
    result = _impl.trace(a, offset, axis1, axis2, dtype)
    return _helpers.result_or_out(result, out)


@normalizer
def eye(N, M=None, k=0, dtype: DTypeLike = float, order="C", *, like: SubokLike = None):
    if order != "C":
        raise NotImplementedError
    result = _impl.eye(N, M, k, dtype)
    return _helpers.array_from(result)


@normalizer
def identity(n, dtype: DTypeLike = None, *, like: SubokLike = None):
    result = torch.eye(n, dtype=dtype)
    return _helpers.array_from(result)


@normalizer
def diag(v: ArrayLike, k=0):
    result = torch.diag(v, k)
    return _helpers.array_from(result)


@normalizer
def diagflat(v: ArrayLike, k=0):
    result = torch.diagflat(v, k)
    return _helpers.array_from(result)


def diag_indices(n, ndim=2):
    result = _impl.diag_indices(n, ndim)
    return _helpers.tuple_arrays_from(result)


@normalizer
def diag_indices_from(arr: ArrayLike):
    result = _impl.diag_indices_from(arr)
    return _helpers.tuple_arrays_from(result)


@normalizer
def fill_diagonal(a: ArrayLike, val: ArrayLike, wrap=False):
    result = _impl.fill_diagonal(a, val, wrap)
    return _helpers.array_from(result)


@normalizer
def vdot(a: ArrayLike, b: ArrayLike, /):
    result = _impl.vdot(a, b)
    return result.item()


@normalizer
def dot(a: ArrayLike, b: ArrayLike, out=None):
    result = _impl.dot(a, b)
    return _helpers.result_or_out(result, out)


# ### sort and partition ###


@normalizer
def sort(a: ArrayLike, axis=-1, kind=None, order=None):
    result = _impl.sort(a, axis, kind, order)
    return _helpers.array_from(result)


@normalizer
def argsort(a: ArrayLike, axis=-1, kind=None, order=None):
    result = _impl.argsort(a, axis, kind, order)
    return _helpers.array_from(result)


@normalizer
def searchsorted(
    a: ArrayLike, v: ArrayLike, side="left", sorter: Optional[ArrayLike] = None
):
    result = torch.searchsorted(a, v, side=side, sorter=sorter)
    return _helpers.array_from(result)


# ### swap/move/roll axis ###


@normalizer
def moveaxis(a: ArrayLike, source, destination):
    result = _impl.moveaxis(a, source, destination)
    return _helpers.array_from(result)


@normalizer
def swapaxes(a: ArrayLike, axis1, axis2):
    result = _flips.swapaxes(a, axis1, axis2)
    return _helpers.array_from(result)


@normalizer
def rollaxis(a: ArrayLike, axis, start=0):
    result = _flips.rollaxis(a, axis, start)
    return _helpers.array_from(result)


# ### shape manipulations ###


@normalizer
def squeeze(a: ArrayLike, axis=None):
    result = _impl.squeeze(a, axis)
    return _helpers.array_from(result, a)


@normalizer
def reshape(a: ArrayLike, newshape, order="C"):
    result = _impl.reshape(a, newshape, order=order)
    return _helpers.array_from(result, a)


@normalizer
def transpose(a: ArrayLike, axes=None):
    result = _impl.transpose(a, axes)
    return _helpers.array_from(result, a)


@normalizer
def ravel(a: ArrayLike, order="C"):
    result = _impl.ravel(a)
    return _helpers.array_from(result, a)


# leading underscore since arr.flatten exists but np.flatten does not
@normalizer
def _flatten(a: ArrayLike, order="C"):
    result = _impl._flatten(a)
    return _helpers.array_from(result, a)


# ### Type/shape etc queries ###


@normalizer
def real(a: ArrayLike):
    result = torch.real(a)
    return _helpers.array_from(result)


@normalizer
def imag(a: ArrayLike):
    result = _impl.imag(a)
    return _helpers.array_from(result)


@normalizer
def round_(a: ArrayLike, decimals=0, out=None):
    result = _impl.round(a, decimals)
    return _helpers.result_or_out(result, out)


around = round_
round = round_


# ### reductions ###


NoValue = None   # FIXME

@normalizer
def sum(a : ArrayLike, axis: AxisLike=None, dtype : DTypeLike=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    result = _reductions.sum(a, axis=axis, dtype=dtype, initial=initial, where=where, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def prod(a : ArrayLike, axis: AxisLike=None, dtype : DTypeLike=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    result = _reductions.prod(a, axis=axis, dtype=dtype, initial=initial, where=where, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


product = prod


@normalizer
def mean(a : ArrayLike, axis: AxisLike=None, dtype : DTypeLike=None, out=None, keepdims=NoValue,  *, where=NoValue):
    result = _reductions.mean(a, axis=axis, dtype=dtype, where=NoValue, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def var(a: ArrayLike, axis: AxisLike=None, dtype : DTypeLike=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue):
    result = _reductions.var(a, axis=axis, dtype=dtype, ddof=ddof, where=where, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def std(a: ArrayLike, axis: AxisLike=None, dtype : DTypeLike=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue):
    result = _reductions.std(a, axis=axis, dtype=dtype, ddof=ddof, where=where, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def argmin(a: ArrayLike, axis: AxisLike=None, out=None, *, keepdims=NoValue):
    result = _reductions.argmin(a, axis=axis, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def argmax(a: ArrayLike, axis: AxisLike=None, out=None, *, keepdims=NoValue):
    result = _reductions.argmax(a, axis=axis, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def amax(a : ArrayLike, axis : AxisLike=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    result = _reductions.max(a, axis=axis, initial=initial, where=where, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


max = amax


@normalizer
def amin(a: ArrayLike, axis: AxisLike=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    result = _reductions.min(a, axis=axis, initial=initial, where=where, keepdims=keepdims)
    return _helpers.result_or_out(result, out)

min = amin


@normalizer
def ptp(a: ArrayLike, axis: AxisLike=None, out=None, keepdims=NoValue):
    result = _reductions.ptp(a, axis=axis, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def all(a: ArrayLike, axis: AxisLike=None, out=None, keepdims=NoValue, *, where=NoValue):
    result = _reductions.all(a, axis=axis, where=where, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def any(a: ArrayLike, axis: AxisLike=None, out=None, keepdims=NoValue, *, where=NoValue):
    result = _reductions.any(a, axis=axis, where=where, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def count_nonzero(a: ArrayLike, axis: AxisLike=None, *, keepdims=False):
    result = _reductions.count_nonzero(a, axis=axis, keepdims=keepdims)
    return _helpers.array_from(result)


@normalizer
def cumsum(a: ArrayLike, axis: AxisLike = None, dtype: DTypeLike = None, out=None):
    result = _reductions.cumsum(a, axis=axis, dtype=dtype)
    return _helpers.result_or_out(result, out)


@normalizer
def cumprod(a: ArrayLike, axis: AxisLike = None, dtype: DTypeLike = None, out=None):
    result = _reductions.cumprod(a, axis=axis, dtype=dtype)
    return _helpers.result_or_out(result, out)


@normalizer
def quantile(
    a : ArrayLike,
    q : ArrayLike,
    axis: AxisLike=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None,
):
    if interpolation is not None:
        raise ValueError("'interpolation' argument is deprecated; use 'method' instead")

    result = _reductions.quantile(a, q, axis, method=method, keepdims=keepdims)
    return _helpers.result_or_out(result, out, promote_scalar=True)
