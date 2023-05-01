""" Implementation of reduction operations, to be wrapped into arrays, dtypes etc
in the 'public' layer.

Anything here only deals with torch objects, e.g. "dtype" is a torch.dtype instance etc
"""
from __future__ import annotations

import functools
from typing import Optional

import torch

from . import _dtypes_impl, _util
from ._normalizations import (
    ArrayLike,
    AxisLike,
    DTypeLike,
    KeepDims,
    NotImplementedType,
    OutArray,
)


def _deco_axis_expand(func):
    """
    Generically handle axis arguments in reductions.
    axis is *always* the 2nd arg in the funciton so no need to have a look at its signature
    """

    @functools.wraps(func)
    def wrapped(a, axis=None, *args, **kwds):

        if axis is not None:
            axis = _util.normalize_axis_tuple(axis, a.ndim)

        if axis == ():
            # So we insert a length-one axis and run the reduction along it.
            # We cannot return a.clone() as this would sidestep the checks inside the function
            newshape = _util.expand_shape(a.shape, axis=0)
            a = a.reshape(newshape)
            axis = (0,)

        return func(a, axis, *args, **kwds)

    return wrapped


def _atleast_float(dtype, other_dtype):
    """Return a dtype that is real or complex floating-point.

    For inputs that are boolean or integer dtypes, this returns the default
    float dtype; inputs that are complex get converted to the default complex
    dtype; real floating-point dtypes (`float*`) get passed through unchanged
    """
    if dtype is None:
        dtype = other_dtype
    if not (dtype.is_floating_point or dtype.is_complex):
        return _dtypes_impl.default_dtypes.float_dtype
    return dtype


@_deco_axis_expand
def count_nonzero(a: ArrayLike, axis: AxisLike = None, *, keepdims: KeepDims = False):
    return a.count_nonzero(axis)


@_deco_axis_expand
def argmax(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    *,
    keepdims: KeepDims = False,
):
    axis = _util.allow_only_single_axis(axis)

    if a.dtype == torch.bool:
        # RuntimeError: "argmax_cpu" not implemented for 'Bool'
        a = a.to(torch.uint8)

    return torch.argmax(a, axis)


@_deco_axis_expand
def argmin(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    *,
    keepdims: KeepDims = False,
):
    axis = _util.allow_only_single_axis(axis)

    if a.dtype == torch.bool:
        # RuntimeError: "argmin_cpu" not implemented for 'Bool'
        a = a.to(torch.uint8)

    return torch.argmin(a, axis)


@_deco_axis_expand
def any(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
):
    axis = _util.allow_only_single_axis(axis)
    axis_kw = {} if axis is None else {"dim": axis}
    return torch.any(a, **axis_kw)


@_deco_axis_expand
def all(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
):
    axis = _util.allow_only_single_axis(axis)
    axis_kw = {} if axis is None else {"dim": axis}
    return torch.all(a, **axis_kw)


@_deco_axis_expand
def amax(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    keepdims: KeepDims = False,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
):
    return a.amax(axis)


max = amax


@_deco_axis_expand
def amin(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    keepdims: KeepDims = False,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
):
    return a.amin(axis)


min = amin


@_deco_axis_expand
def ptp(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    keepdims: KeepDims = False,
):
    return a.amax(axis) - a.amin(axis)


@_deco_axis_expand
def sum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    out: Optional[OutArray] = None,
    keepdims: KeepDims = False,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
):
    assert dtype is None or isinstance(dtype, torch.dtype)

    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes.int_dtype

    axis_kw = {} if axis is None else {"dim": axis}
    return a.sum(dtype=dtype, **axis_kw)


@_deco_axis_expand
def prod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    out: Optional[OutArray] = None,
    keepdims: KeepDims = False,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
):
    axis = _util.allow_only_single_axis(axis)

    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes.int_dtype

    axis_kw = {} if axis is None else {"dim": axis}
    return a.prod(dtype=dtype, **axis_kw)


product = prod


@_deco_axis_expand
def mean(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    out: Optional[OutArray] = None,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
):
    dtype = _atleast_float(dtype, a.dtype)

    is_half = dtype == torch.float16
    if is_half:
        # XXX revisit when the pytorch version has pytorch/pytorch#95166
        dtype = torch.float32

    axis_kw = {} if axis is None else {"dim": axis}
    result = a.mean(dtype=dtype, **axis_kw)

    if is_half:
        result = result.to(torch.float16)

    return result


@_deco_axis_expand
def std(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    out: Optional[OutArray] = None,
    ddof=0,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
):
    dtype = _atleast_float(dtype, a.dtype)
    tensor = _util.cast_if_needed(a, dtype)
    return tensor.std(dim=axis, correction=ddof)


@_deco_axis_expand
def var(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    out: Optional[OutArray] = None,
    ddof=0,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
):
    dtype = _atleast_float(dtype, a.dtype)
    tensor = _util.cast_if_needed(a, dtype)
    return tensor.var(dim=axis, correction=ddof)


# cumsum / cumprod are almost reductions:
#   1. no keepdims
#   2. axis=None flattens


def cumsum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    out: Optional[OutArray] = None,
):
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes.int_dtype
    if dtype is None:
        dtype = a.dtype

    (a,), axis = _util.axis_none_flatten(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)

    return a.cumsum(axis=axis, dtype=dtype)


def cumprod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: Optional[DTypeLike] = None,
    out: Optional[OutArray] = None,
):
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes.int_dtype
    if dtype is None:
        dtype = a.dtype

    (a,), axis = _util.axis_none_flatten(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)

    return a.cumprod(axis=axis, dtype=dtype)


cumproduct = cumprod


@_deco_axis_expand
def average(
    a: ArrayLike,
    axis=None,
    weights: ArrayLike = None,
    returned=False,
    *,
    keepdims=False,
):
    if weights is None:
        result = mean(a, axis=axis)
        wsum = torch.as_tensor(a.numel() / result.numel(), dtype=result.dtype)
    else:
        result, wsum = _average_weights(a, axis, weights)

    # We process keepdims manually because the decorator does not deal with variadic returns
    if keepdims:
        result = _util.apply_keepdims(result, axis, a.ndim)

    if returned:
        if wsum.shape != result.shape:
            wsum = torch.broadcast_to(wsum, result.shape).clone()
        return result, wsum
    else:
        return result


def _average_weights(a, axis, w):
    if not a.dtype.is_floating_point:
        a = a.double()

    result_dtype = _dtypes_impl.result_type_impl([a.dtype, w.dtype])

    # axis & weights
    if a.shape != w.shape:
        if axis is None:
            raise TypeError(
                "Axis must be specified when shapes of a and weights " "differ."
            )
        if w.ndim != 1:
            raise TypeError("1D weights expected when shapes of a and weights differ.")
        if w.shape[0] != a.shape[axis]:
            raise ValueError("Length of weights not compatible with specified axis.")

        # setup weight to broadcast along axis
        w = torch.broadcast_to(w, (a.ndim - 1) * (1,) + w.shape)
        w = w.swapaxes(-1, axis)

    # do the work
    numerator = torch.mul(a, w).sum(axis, dtype=result_dtype)
    denominator = w.sum(axis, dtype=result_dtype)
    result = numerator / denominator

    return result, denominator


# Not using deco_axis_expand as it assumes that axis is the second arg
def quantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    overwrite_input=False,
    method="linear",
    keepdims: KeepDims = False,
    *,
    interpolation: NotImplementedType = None,
):
    if overwrite_input:
        # raise NotImplementedError("overwrite_input in quantile not implemented.")
        # NumPy documents that `overwrite_input` MAY modify inputs:
        # https://numpy.org/doc/stable/reference/generated/numpy.percentile.html#numpy-percentile
        # Here we choose to work out-of-place because why not.
        pass

    if not a.dtype.is_floating_point:
        dtype = _dtypes_impl.default_dtypes.float_dtype
        a = a.to(dtype)

    # edge case: torch.quantile only supports float32 and float64
    if a.dtype == torch.float16:
        a = a.to(torch.float32)

    if axis is None:
        a = a.flatten()
        q = q.flatten()
        axis = (0,)
    else:
        axis = _util.normalize_axis_tuple(axis, a.ndim)

    # FIXME(Mario) Doesn't np.quantile accept a tuple?
    # torch.quantile does accept a number. If we don't want to implement the tuple behaviour
    # (it's deffo low prio) change `normalize_axis_tuple` into a normalize_axis index above.
    axis = _util.allow_only_single_axis(axis)

    q = _util.cast_if_needed(q, a.dtype)

    return torch.quantile(a, q, axis=axis, interpolation=method)


def percentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = None,
    out: Optional[OutArray] = None,
    overwrite_input=False,
    method="linear",
    keepdims: KeepDims = False,
    *,
    interpolation: NotImplementedType = None,
):
    return quantile(
        a,
        q / 100.0,
        axis=axis,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        interpolation=interpolation,
    )


def median(
    a: ArrayLike,
    axis=None,
    out: Optional[OutArray] = None,
    overwrite_input=False,
    keepdims: KeepDims = False,
):
    return quantile(
        a,
        torch.as_tensor(0.5),
        axis=axis,
        overwrite_input=overwrite_input,
        out=out,
        keepdims=keepdims,
    )
