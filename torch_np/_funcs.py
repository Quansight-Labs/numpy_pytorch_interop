from typing import Optional

import torch

from . import _detail as _impl
from . import _helpers
from ._detail import _util
from ._normalizations import (
    ArrayLike,
    AxisLike,
    DTypeLike,
    NDArray,
    SubokLike,
    normalizer,
)

NoValue = _util.NoValue


@normalizer
def nonzero(a: ArrayLike):
    result = a.nonzero(as_tuple=True)
    return _helpers.tuple_arrays_from(result)


@normalizer
def argwhere(a: ArrayLike):
    result = torch.argwhere(a)
    return _helpers.array_from(result)


@normalizer
def flatnonzero(a: ArrayLike):
    result = a.ravel().nonzero(as_tuple=True)[0]
    return _helpers.array_from(result)


@normalizer
def clip(
    a: ArrayLike,
    min: Optional[ArrayLike] = None,
    max: Optional[ArrayLike] = None,
    out: Optional[NDArray] = None,
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


@normalizer
def tile(A: ArrayLike, reps):
    result = _impl.tile(A, reps)
    return _helpers.array_from(result)


# ### diag et al ###


@normalizer
def diagonal(a: ArrayLike, offset=0, axis1=0, axis2=1):
    result = _impl.diagonal(a, offset, axis1, axis2)
    return _helpers.array_from(result)


@normalizer
def trace(
    a: ArrayLike,
    offset=0,
    axis1=0,
    axis2=1,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
):
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
def dot(a: ArrayLike, b: ArrayLike, out: Optional[NDArray] = None):
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
    result = _impl.swapaxes(a, axis1, axis2)
    return _helpers.array_from(result)


@normalizer
def rollaxis(a: ArrayLike, axis, start=0):
    result = _impl.rollaxis(a, axis, start)
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
def round_(a: ArrayLike, decimals=0, out: Optional[NDArray] = None):
    result = _impl.round(a, decimals)
    return _helpers.result_or_out(result, out)


around = round_
round = round_


# ### reductions ###


@normalizer
def sum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    initial=NoValue,
    where=NoValue,
):
    result = _impl.sum(
        a, axis=axis, dtype=dtype, initial=initial, where=where, keepdims=keepdims
    )
    return _helpers.result_or_out(result, out)


@normalizer
def prod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    initial=NoValue,
    where=NoValue,
):
    result = _impl.prod(
        a, axis=axis, dtype=dtype, initial=initial, where=where, keepdims=keepdims
    )
    return _helpers.result_or_out(result, out)


product = prod


@normalizer
def mean(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    *,
    where=NoValue,
):
    result = _impl.mean(a, axis=axis, dtype=dtype, where=NoValue, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def var(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
    ddof=0,
    keepdims=NoValue,
    *,
    where=NoValue,
):
    result = _impl.var(
        a, axis=axis, dtype=dtype, ddof=ddof, where=where, keepdims=keepdims
    )
    return _helpers.result_or_out(result, out)


@normalizer
def std(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
    ddof=0,
    keepdims=NoValue,
    *,
    where=NoValue,
):
    result = _impl.std(
        a, axis=axis, dtype=dtype, ddof=ddof, where=where, keepdims=keepdims
    )
    return _helpers.result_or_out(result, out)


@normalizer
def argmin(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    *,
    keepdims=NoValue,
):
    result = _impl.argmin(a, axis=axis, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def argmax(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    *,
    keepdims=NoValue,
):
    result = _impl.argmax(a, axis=axis, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def amax(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    initial=NoValue,
    where=NoValue,
):
    result = _impl.max(a, axis=axis, initial=initial, where=where, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


max = amax


@normalizer
def amin(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    initial=NoValue,
    where=NoValue,
):
    result = _impl.min(a, axis=axis, initial=initial, where=where, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


min = amin


@normalizer
def ptp(
    a: ArrayLike, axis: AxisLike = None, out: Optional[NDArray] = None, keepdims=NoValue
):
    result = _impl.ptp(a, axis=axis, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def all(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    *,
    where=NoValue,
):
    result = _impl.all(a, axis=axis, where=where, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def any(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    *,
    where=NoValue,
):
    result = _impl.any(a, axis=axis, where=where, keepdims=keepdims)
    return _helpers.result_or_out(result, out)


@normalizer
def count_nonzero(a: ArrayLike, axis: AxisLike = None, *, keepdims=False):
    result = _impl.count_nonzero(a, axis=axis, keepdims=keepdims)
    return _helpers.array_from(result)


@normalizer
def cumsum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
):
    result = _impl.cumsum(a, axis=axis, dtype=dtype)
    return _helpers.result_or_out(result, out)


@normalizer
def cumprod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
):
    result = _impl.cumprod(a, axis=axis, dtype=dtype)
    return _helpers.result_or_out(result, out)


cumproduct = cumprod


@normalizer
def quantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None,
):
    result = _impl.quantile(
        a,
        q,
        axis,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        interpolation=interpolation,
    )
    return _helpers.result_or_out(result, out, promote_scalar=True)


@normalizer
def percentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None,
):
    result = _impl.percentile(
        a,
        q,
        axis,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        interpolation=interpolation,
    )
    return _helpers.result_or_out(result, out, promote_scalar=True)


def median(
    a, axis=None, out: Optional[NDArray] = None, overwrite_input=False, keepdims=False
):
    return quantile(
        a, 0.5, axis=axis, overwrite_input=overwrite_input, out=out, keepdims=keepdims
    )
