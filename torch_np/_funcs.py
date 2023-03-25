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

_dtypes_impl = _impl._dtypes_impl

NoValue = _util.NoValue


@normalizer
def nonzero(a: ArrayLike):
    result = a.nonzero(as_tuple=True)
    return result


@normalizer
def argwhere(a: ArrayLike):
    result = torch.argwhere(a)
    return result


@normalizer
def flatnonzero(a: ArrayLike):
    result = a.ravel().nonzero(as_tuple=True)[0]
    return result


@normalizer
def clip(
    a: ArrayLike,
    min: Optional[ArrayLike] = None,
    max: Optional[ArrayLike] = None,
    out: Optional[NDArray] = None,
):
    # np.clip requires both a_min and a_max not None, while ndarray.clip allows
    # one of them to be None. Follow the more lax version.
    if min is None and max is None:
        raise ValueError("One of max or min must be given")
    result = a.clamp(min, max)
    return result


@normalizer
def repeat(a: ArrayLike, repeats: ArrayLike, axis=None):
    # XXX: scalar repeats; ArrayLikeOrScalar ?
    result = torch.repeat_interleave(a, repeats, axis)
    return result


@normalizer
def tile(A: ArrayLike, reps):
    if isinstance(reps, int):
        reps = (reps,)
    result = torch.tile(A, reps)
    return result


# ### diag et al ###


@normalizer
def diagonal(a: ArrayLike, offset=0, axis1=0, axis2=1):
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    result = torch.diagonal(a, offset, axis1, axis2)
    return result


@normalizer
def trace(
    a: ArrayLike,
    offset=0,
    axis1=0,
    axis2=1,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
):
    result = torch.diagonal(a, offset, dim1=axis1, dim2=axis2).sum(-1, dtype=dtype)
    return result


@normalizer
def eye(N, M=None, k=0, dtype: DTypeLike = float, order="C", *, like: SubokLike = None):
    if order != "C":
        raise NotImplementedError
    if M is None:
        M = N
    z = torch.zeros(N, M, dtype=dtype)
    z.diagonal(k).fill_(1)
    return z


@normalizer
def identity(n, dtype: DTypeLike = None, *, like: SubokLike = None):
    result = torch.eye(n, dtype=dtype)
    return result


@normalizer
def diag(v: ArrayLike, k=0):
    result = torch.diag(v, k)
    return result


@normalizer
def diagflat(v: ArrayLike, k=0):
    result = torch.diagflat(v, k)
    return result


def diag_indices(n, ndim=2):
    idx = torch.arange(n)
    return (idx,) * ndim


@normalizer
def diag_indices_from(arr: ArrayLike):
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    s = arr.shape
    if s[1:] != s[:-1]:
        raise ValueError("All dimensions of input must be of equal length")
    return diag_indices(s[0], arr.ndim)


@normalizer
def fill_diagonal(a: ArrayLike, val: ArrayLike, wrap=False):
    # torch.Tensor.fill_diagonal_ only accepts scalars. Thus vendor the numpy source,
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/index_tricks.py#L786-L917

    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    end = None
    if a.ndim == 2:
        # Explicit, fast formula for the common case.  For 2-d arrays, we
        # accept rectangular ones.
        step = a.shape[1] + 1
        # This is needed to don't have tall matrix have the diagonal wrap.
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        # For more than d=2, the strided formula is only valid for arrays with
        # all dimensions equal, so we check first.
        s = a.shape
        if s[1:] != s[:-1]:
            raise ValueError("All dimensions of input must be of equal length")
        sz = torch.as_tensor(a.shape[:-1])
        step = 1 + (torch.cumprod(sz, 0)).sum()

    # Write the value out into the diagonal.
    a.ravel()[:end:step] = val
    return a


@normalizer
def vdot(a: ArrayLike, b: ArrayLike, /):
    # 1. torch only accepts 1D arrays, numpy ravels
    # 2. torch requires matching dtype, while numpy casts (?)
    t_a, t_b = torch.atleast_1d(a, b)
    if t_a.ndim > 1:
        t_a = t_a.ravel()
    if t_b.ndim > 1:
        t_b = t_b.ravel()

    dtype = _dtypes_impl.result_type_impl((t_a.dtype, t_b.dtype))
    is_half = dtype == torch.float16
    is_bool = dtype == torch.bool

    # work around torch's "dot" not implemented for 'Half', 'Bool'
    if is_half:
        dtype = torch.float32
    if is_bool:
        dtype = torch.uint8

    t_a = _util.cast_if_needed(t_a, dtype)
    t_b = _util.cast_if_needed(t_b, dtype)

    result = torch.vdot(t_a, t_b)

    if is_half:
        result = result.to(torch.float16)
    if is_bool:
        result = result.to(torch.bool)

    return result.item()


@normalizer
def dot(a: ArrayLike, b: ArrayLike, out: Optional[NDArray] = None):
    dtype = _dtypes_impl.result_type_impl((a.dtype, b.dtype))
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    if a.ndim == 0 or b.ndim == 0:
        result = a * b
    elif a.ndim == 1 and b.ndim == 1:
        result = torch.dot(a, b)
    elif a.ndim == 1:
        result = torch.mv(b.T, a).T
    elif b.ndim == 1:
        result = torch.mv(a, b)
    else:
        result = torch.matmul(a, b)
    return result

# ### sort and partition ###


def _sort_helper(tensor, axis, kind, order):
    if order is not None:
        # only relevant for structured dtypes; not supported
        raise NotImplementedError(
            "'order' keyword is only relevant for structured dtypes"
        )

    (tensor,), axis = _util.axis_none_ravel(tensor, axis=axis)
    axis = _util.normalize_axis_index(axis, tensor.ndim)

    stable = kind == "stable"

    return tensor, axis, stable


def _sort(tensor, axis, kind, order):
    # pure torch implementation, used below and in ndarray.sort
    tensor, axis, stable = _sort_helper(tensor, axis, kind, order)
    result = torch.sort(tensor, dim=axis, stable=stable)
    return result.values


@normalizer
def sort(a: ArrayLike, axis=-1, kind=None, order=None):
    result = _sort(a, axis, kind, order)
    return result


@normalizer
def argsort(a: ArrayLike, axis=-1, kind=None, order=None):
    a, axis, stable = _sort_helper(a, axis, kind, order)
    result = torch.argsort(a, dim=axis, stable=stable)
    return result


@normalizer
def searchsorted(
    a: ArrayLike, v: ArrayLike, side="left", sorter: Optional[ArrayLike] = None
):
    result = torch.searchsorted(a, v, side=side, sorter=sorter)
    return result


# ### swap/move/roll axis ###


@normalizer
def moveaxis(a: ArrayLike, source, destination):
    source = _util.normalize_axis_tuple(source, a.ndim, "source")
    destination = _util.normalize_axis_tuple(destination, a.ndim, "destination")
    result = torch.moveaxis(a, source, destination)
    return result


@normalizer
def swapaxes(a: ArrayLike, axis1, axis2):
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    result = torch.swapaxes(a, axis1, axis2)
    return result


@normalizer
def rollaxis(a: ArrayLike, axis, start=0):
    # Straight vendor from:
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1259
    #
    # Also note this function in NumPy is mostly retained for backwards compat
    # (https://stackoverflow.com/questions/29891583/reason-why-numpy-rollaxis-is-so-confusing)
    # so let's not touch it unless hard pressed.
    n = a.ndim
    axis = _util.normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise _util.AxisError(msg % ("start", -n, "start", n + 1, start))
    if axis < start:
        # it's been removed
        start -= 1
    if axis == start:
        # numpy returns a view, here we try returning the tensor itself
        # return tensor[...]
        return a
    axes = list(range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.view(axes)


@normalizer
def roll(a: ArrayLike, shift, axis=None):
    if axis is not None:
        axis = _util.normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        if not isinstance(shift, tuple):
            shift = (shift,) * len(axis)
    result = a.roll(shift, axis)
    return result


# ### shape manipulations ###


@normalizer
def squeeze(a: ArrayLike, axis=None):
    if axis == ():
        result = a
    elif axis is None:
        result = a.squeeze()
    else:
        if isinstance(axis, tuple):
            result = a
            for ax in axis:
                result = a.squeeze(ax)
        else:
            result = a.squeeze(axis)
    return result


@normalizer
def reshape(a: ArrayLike, newshape, order="C"):
    if order != "C":
        raise NotImplementedError
    # if sh = (1, 2, 3), numpy allows both .reshape(sh) and .reshape(*sh)
    newshape = newshape[0] if len(newshape) == 1 else newshape
    result = a.reshape(newshape)
    return result


@normalizer
def transpose(a: ArrayLike, axes=None):
    # numpy allows both .tranpose(sh) and .transpose(*sh)
    if axes in [(), None, (None,)]:
        axes = tuple(range(a.ndim))[::-1]
    try:
        result = a.permute(axes)
    except RuntimeError:
        raise ValueError("axes don't match array")
    return result


@normalizer
def ravel(a: ArrayLike, order="C"):
    if order != "C":
        raise NotImplementedError
    result = a.ravel()
    return result


# leading underscore since arr.flatten exists but np.flatten does not
@normalizer
def _flatten(a: ArrayLike, order="C"):
    if order != "C":
        raise NotImplementedError
    # return a copy
    result = a.flatten()
    return result


# ### Type/shape etc queries ###


@normalizer
def real(a: ArrayLike):
    result = torch.real(a)
    return result


@normalizer
def imag(a: ArrayLike):
    if a.is_complex():
        result = a.imag
    else:
        result = torch.zeros_like(a)
    return result


@normalizer
def round_(a: ArrayLike, decimals=0, out: Optional[NDArray] = None) -> OutArray:
    if a.is_floating_point():
        result = torch.round(a, decimals=decimals)
    elif a.is_complex():
        # RuntimeError: "round_cpu" not implemented for 'ComplexFloat'
        result = (
            torch.round(a.real, decimals=decimals)
            + torch.round(a.imag, decimals=decimals) * 1j
        )
    else:
        # RuntimeError: "round_cpu" not implemented for 'int'
        result = a
    return result


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
    return result


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
    return result


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
    return result


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
    return result


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
    return result


@normalizer
def argmin(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    *,
    keepdims=NoValue,
):
    result = _impl.argmin(a, axis=axis, keepdims=keepdims)
    return result


@normalizer
def argmax(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    *,
    keepdims=NoValue,
):
    result = _impl.argmax(a, axis=axis, keepdims=keepdims)
    return result


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
    return result


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
    return result


min = amin


@normalizer
def ptp(
    a: ArrayLike, axis: AxisLike = None, out: Optional[NDArray] = None, keepdims=NoValue
):
    result = _impl.ptp(a, axis=axis, keepdims=keepdims)
    return result


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
    return result


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
    return result


@normalizer
def count_nonzero(a: ArrayLike, axis: AxisLike = None, *, keepdims=False):
    result = _impl.count_nonzero(a, axis=axis, keepdims=keepdims)
    return result


@normalizer
def cumsum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
):
    result = _impl.cumsum(a, axis=axis, dtype=dtype)
    return result


@normalizer
def cumprod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
):
    result = _impl.cumprod(a, axis=axis, dtype=dtype)
    return result


cumproduct = cumprod


@normalizer(promote_scalar_result=True)
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
    return result


@normalizer(promote_scalar_result=True)
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
    return result


def median(
    a, axis=None, out: Optional[NDArray] = None, overwrite_input=False, keepdims=False
):
    return quantile(
        a, 0.5, axis=axis, overwrite_input=overwrite_input, out=out, keepdims=keepdims
    )
