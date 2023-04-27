""" Implementation of reduction operations, to be wrapped into arrays, dtypes etc
in the 'public' layer.

Anything here only deals with torch objects, e.g. "dtype" is a torch.dtype instance etc
"""

import functools

import torch

from . import _dtypes_impl, _util


def deco_axis_expand(func):
    """Generically handle axis arguments in reductions."""

    @functools.wraps(func)
    def wrapped(tensor, axis=None, *args, **kwds):

        if axis is not None:
            axis = _util.normalize_axis_tuple(axis, tensor.ndim)

        if axis == ():
            # NumPy does essentially an identity operation:
            # >>> np.sum(np.ones(2), axis=())
            # array([1., 1.])
            # So we insert a length-one axis and run the reduction along it.
            newshape = _util.expand_shape(tensor.shape, axis=0)
            tensor = tensor.reshape(newshape)
            axis = (0,)

        result = func(tensor, axis, *args, **kwds)
        return result

    return wrapped


def emulate_keepdims(func):
    @functools.wraps(func)
    def wrapped(tensor, axis=None, keepdims=None, *args, **kwds):
        result = func(tensor, axis, *args, **kwds)
        if keepdims:
            result = _util.apply_keepdims(result, axis, tensor.ndim)
        return result

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


@emulate_keepdims
@deco_axis_expand
def count_nonzero(a, axis=None):
    return a.count_nonzero(axis)


@emulate_keepdims
@deco_axis_expand
def argmax(tensor, axis=None):
    axis = _util.allow_only_single_axis(axis)

    if tensor.dtype == torch.bool:
        # RuntimeError: "argmax_cpu" not implemented for 'Bool'
        tensor = tensor.to(torch.uint8)

    return torch.argmax(tensor, axis)


@emulate_keepdims
@deco_axis_expand
def argmin(tensor, axis=None):
    axis = _util.allow_only_single_axis(axis)

    if tensor.dtype == torch.bool:
        # RuntimeError: "argmin_cpu" not implemented for 'Bool'
        tensor = tensor.to(torch.uint8)

    return torch.argmin(tensor, axis)


@emulate_keepdims
@deco_axis_expand
def any(tensor, axis=None, *, where=None):
    axis = _util.allow_only_single_axis(axis)
    axis_kw = {} if axis is None else {"dim": axis}
    return torch.any(tensor, **axis_kw)


@emulate_keepdims
@deco_axis_expand
def all(tensor, axis=None, *, where=None):
    axis = _util.allow_only_single_axis(axis)
    axis_kw = {} if axis is None else {"dim": axis}
    return torch.all(tensor, **axis_kw)


@emulate_keepdims
@deco_axis_expand
def max(tensor, axis=None, initial=None, where=None):
    return tensor.amax(axis)


@emulate_keepdims
@deco_axis_expand
def min(tensor, axis=None, initial=None, where=None):
    return tensor.amin(axis)


@emulate_keepdims
@deco_axis_expand
def ptp(tensor, axis=None):
    return tensor.amax(axis) - tensor.amin(axis)


@emulate_keepdims
@deco_axis_expand
def sum(tensor, axis=None, dtype=None, initial=None, where=None):
    assert dtype is None or isinstance(dtype, torch.dtype)

    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes.int_dtype

    if axis is None:
        result = tensor.sum(dtype=dtype)
    else:
        result = tensor.sum(dtype=dtype, dim=axis)

    return result


@emulate_keepdims
@deco_axis_expand
def prod(tensor, axis=None, dtype=None, initial=None, where=None):
    axis = _util.allow_only_single_axis(axis)

    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes.int_dtype

    if axis is None:
        result = tensor.prod(dtype=dtype)
    else:
        result = tensor.prod(dtype=dtype, dim=axis)

    return result


@emulate_keepdims
@deco_axis_expand
def mean(tensor, axis=None, dtype=None, *, where=None):
    dtype = _atleast_float(dtype, tensor.dtype)

    is_half = dtype == torch.float16
    if is_half:
        # XXX revisit when the pytorch version has pytorch/pytorch#95166
        dtype = torch.float32

    if axis is None:
        result = tensor.mean(dtype=dtype)
    else:
        result = tensor.mean(dtype=dtype, dim=axis)

    if is_half:
        result = result.to(torch.float16)

    return result


@emulate_keepdims
@deco_axis_expand
def std(tensor, axis=None, dtype=None, ddof=0, *, where=None):
    dtype = _atleast_float(dtype, tensor.dtype)
    tensor = _util.cast_if_needed(tensor, dtype)
    return tensor.std(dim=axis, correction=ddof)


@emulate_keepdims
@deco_axis_expand
def var(tensor, axis=None, dtype=None, ddof=0, *, where=None):
    dtype = _atleast_float(dtype, tensor.dtype)
    tensor = _util.cast_if_needed(tensor, dtype)
    return tensor.var(dim=axis, correction=ddof)


# cumsum / cumprod are almost reductions:
#   1. no keepdims
#   2. axis=None flattens


def cumprod(tensor, axis, dtype=None):
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes.int_dtype
    if dtype is None:
        dtype = tensor.dtype

    (tensor,), axis = _util.axis_none_flatten(tensor, axis=axis)
    axis = _util.normalize_axis_index(axis, tensor.ndim)

    return tensor.cumprod(axis=axis, dtype=dtype)


def cumsum(tensor, axis, dtype=None):
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes.int_dtype
    if dtype is None:
        dtype = tensor.dtype

    (tensor,), axis = _util.axis_none_flatten(tensor, axis=axis)
    axis = _util.normalize_axis_index(axis, tensor.ndim)

    return tensor.cumsum(axis=axis, dtype=dtype)


def average(a, axis, weights, returned=False, keepdims=False):
    if weights is None:
        result, wsum = average_noweights(a, axis, keepdims=keepdims)
    else:
        result, wsum = average_weights(a, axis, weights, keepdims=keepdims)

    if returned and wsum.shape != result.shape:
        wsum = torch.broadcast_to(wsum, result.shape).clone()
    return result, wsum


def average_noweights(a, axis, keepdims=False):
    result = mean(a, axis=axis, keepdims=keepdims)
    scl = torch.as_tensor(a.numel() / result.numel(), dtype=result.dtype)
    return result, scl


def average_weights(a, axis, w, keepdims=False):
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

    # keepdims
    if keepdims:
        result = _util.apply_keepdims(result, axis, a.ndim)

    return result, denominator


def quantile(
    a,
    q,
    axis,
    overwrite_input,
    method,
    keepdims=False,
    interpolation=None,
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

    # axis=None flattens, so store the originals to reuse with keepdims=True below
    ax, ndim = axis, a.ndim
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

    result = torch.quantile(a, q, axis=axis, interpolation=method)

    # NB: not using @emulate_keepdims here because the signature is (a, q, axis, ...)
    # while the decorator expects (a, axis, ...)
    # this can be fixed, of course, but the cure seems worse then the desease
    if keepdims:
        result = _util.apply_keepdims(result, ax, ndim)
    return result


def percentile(
    a,
    q,
    axis,
    overwrite_input,
    method,
    keepdims=False,
    interpolation=None,
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
