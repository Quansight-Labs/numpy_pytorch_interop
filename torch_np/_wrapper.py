"""A thin pytorch / numpy compat layer.

Things imported from here have numpy-compatible signatures but operate on
pytorch tensors.
"""
# import numpy as np

import torch

from ._detail import _dtypes_impl, _flips, _reductions, _util
from ._detail import implementations as _impl
from ._ndarray import (
    array,
    asarray,
    asarray_replacer,
    can_cast,
    maybe_set_base,
    ndarray,
    newaxis,
    result_type,
)

from . import _dtypes, _helpers, _decorators  # isort: skip  # XXX


# Things to decide on (punt for now)
#
# 1. Q: What are the return types of wrapper functions: plain torch.Tensors or
#       wrapper ndarrays.
#    A: Wrapper ndarrays.
#
# 2. Q: Default dtypes: numpy defaults to float64, pytorch defaults to float32
#    A: Stick to pytorch defaults?
#    NB: numpy recommends `dtype=float`?
#
# 3. Q: Masked arrays. Record, structured arrays.
#    A: Ignore for now
#
# 4. Q: What are the defaults for pytorch-specific args not present in numpy signatures?
#       device=..., requires_grad=... etc
#    A: ignore, keep whatever they are from inputs; test w/various combinations
#
# 5. Q: What is the useful action for numpy-specific arguments? e.g. like=...
#    A: like=... and subok=True both raise ValueErrors.
#       initial=... can be useful though, punt on
#       where=...   punt on for now


# TODO
# 1. Mapping of the numpy layout ('C', 'K' etc) to torch layout / memory_format.
# 2. np.dtype <-> torch.dtype
# 3. numpy type casting rules (to be cleaned up in numpy: follow old or new)
#
# 4. wrap/unwrap/wrap patterns:
#   - inputs are scalars, output is an array
#   - two-arg functions (second may be None)
#   - first arg is a sequence/tuple (_stack familty, concatenate, atleast_Nd etc)
#   - optional out arg


NoValue = None


###### array creation routines


def copy(a, order="K", subok=False):
    a = asarray(a)
    _util.subok_not_ok(subok=subok)
    if order != "K":
        raise NotImplementedError
    # XXX: ndarray.copy only accepts order='C'
    return a.copy(order="C")


def atleast_1d(*arys):
    tensors = _helpers.to_tensors(*arys)
    res = torch.atleast_1d(tensors)
    if len(res) == 1:
        return asarray(res[0])
    else:
        return list(asarray(_) for _ in res)


def atleast_2d(*arys):
    tensors = _helpers.to_tensors(*arys)
    res = torch.atleast_2d(tensors)
    if len(res) == 1:
        return asarray(res[0])
    else:
        return list(asarray(_) for _ in res)


def atleast_3d(*arys):
    tensors = _helpers.to_tensors(*arys)
    res = torch.atleast_3d(tensors)
    if len(res) == 1:
        return asarray(res[0])
    else:
        return list(asarray(_) for _ in res)


def _concat_check(tup, dtype, out):
    """Check inputs in concatenate et al."""
    if tup == ():
        # XXX: RuntimeError in torch, ValueError in numpy
        raise ValueError("need at least one array to concatenate")

    if out is not None:
        if not isinstance(out, ndarray):
            raise ValueError("'out' must be an array")

        if dtype is not None:
            # mimic numpy
            raise TypeError(
                "concatenate() only takes `out` or `dtype` as an "
                "argument, but both were provided."
            )


@_decorators.dtype_to_torch
def concatenate(ar_tuple, axis=0, out=None, dtype=None, casting="same_kind"):
    _concat_check(ar_tuple, dtype, out=out)
    tensors = _helpers.to_tensors(*ar_tuple)
    result = _impl.concatenate(tensors, axis, out, dtype, casting)
    return _helpers.result_or_out(result, out)


@_decorators.dtype_to_torch
def vstack(tup, *, dtype=None, casting="same_kind"):
    tensors = _helpers.to_tensors(*tup)
    _concat_check(tensors, dtype, out=None)
    result = _impl.vstack(tensors, dtype=dtype, casting=casting)
    return asarray(result)


row_stack = vstack


@_decorators.dtype_to_torch
def hstack(tup, *, dtype=None, casting="same_kind"):
    tensors = _helpers.to_tensors(*tup)
    _concat_check(tensors, dtype, out=None)
    result = _impl.hstack(tensors, dtype=dtype, casting=casting)
    return asarray(result)


@_decorators.dtype_to_torch
def dstack(tup, *, dtype=None, casting="same_kind"):
    # XXX: in numpy 1.24 dstack does not have dtype and casting keywords
    # but {h,v}stack do.  Hence add them here for consistency.
    tensors = _helpers.to_tensors(*tup)
    result = _impl.dstack(tensors, dtype=dtype, casting=casting)
    return asarray(result)


@_decorators.dtype_to_torch
def column_stack(tup, *, dtype=None, casting="same_kind"):
    # XXX: in numpy 1.24 column_stack does not have dtype and casting keywords
    # but row_stack does. (because row_stack is an alias for vstack, really).
    # Hence add these keywords here for consistency.
    tensors = _helpers.to_tensors(*tup)
    _concat_check(tensors, dtype, out=None)
    result = _impl.column_stack(tensors, dtype=dtype, casting=casting)
    return asarray(result)


@_decorators.dtype_to_torch
def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    tensors = _helpers.to_tensors(*arrays)
    _concat_check(tensors, dtype, out=out)
    result = _impl.stack(tensors, axis=axis, out=out, dtype=dtype, casting=casting)
    return _helpers.result_or_out(result, out)


def array_split(ary, indices_or_sections, axis=0):
    tensor = asarray(ary).get()
    base = ary if isinstance(ary, ndarray) else None
    result = _impl.split_helper(tensor, indices_or_sections, axis)
    return tuple(maybe_set_base(x, base) for x in result)


def split(ary, indices_or_sections, axis=0):
    tensor = asarray(ary).get()
    base = ary if isinstance(ary, ndarray) else None
    result = _impl.split_helper(tensor, indices_or_sections, axis, strict=True)
    return tuple(maybe_set_base(x, base) for x in result)


def hsplit(ary, indices_or_sections):
    tensor = asarray(ary).get()
    base = ary if isinstance(ary, ndarray) else None
    result = _impl.hsplit(tensor, indices_or_sections)
    return tuple(maybe_set_base(x, base) for x in result)


def vsplit(ary, indices_or_sections):
    tensor = asarray(ary).get()
    base = ary if isinstance(ary, ndarray) else None
    result = _impl.vsplit(tensor, indices_or_sections)
    return tuple(maybe_set_base(x, base) for x in result)


def dsplit(ary, indices_or_sections):
    tensor = asarray(ary).get()
    base = ary if isinstance(ary, ndarray) else None
    result = _impl.dsplit(tensor, indices_or_sections)
    return tuple(maybe_set_base(x, base) for x in result)


def kron(a, b):
    a_tensor, b_tensor = _helpers.to_tensors(a, b)
    result = torch.kron(a_tensor, b_tensor)
    return asarray(result)


def tile(A, reps):
    a_tensor = asarray(A).get()
    if isinstance(reps, int):
        reps = (reps,)

    result = torch.tile(a_tensor, reps)
    return asarray(result)


def vander(x, N=None, increasing=False):
    x_tensor = asarray(x).get()
    result = torch.vander(x_tensor, N, increasing)
    return asarray(result)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    if axis != 0 or retstep or not endpoint:
        raise NotImplementedError
    # XXX: raises TypeError if start or stop are not scalars
    return asarray(torch.linspace(start, stop, num, dtype=dtype))


@_decorators.dtype_to_torch
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    if axis != 0 or not endpoint:
        raise NotImplementedError
    start, stop = _helpers.to_tensors(start, stop)
    result = _impl.geomspace(start, stop, num, endpoint, dtype, axis)
    return asarray(result)


@_decorators.dtype_to_torch
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    if axis != 0 or not endpoint:
        raise NotImplementedError
    return asarray(torch.logspace(start, stop, num, base=base, dtype=dtype))


@_decorators.dtype_to_torch
def arange(start=None, stop=None, step=1, dtype=None, *, like=None):
    _util.subok_not_ok(like)
    start, stop, step = _helpers.ndarrays_to_tensors(start, stop, step)
    result = _impl.arange(start, stop, step, dtype=dtype)
    return asarray(result)


@_decorators.dtype_to_torch
def empty(shape, dtype=float, order="C", *, like=None):
    _util.subok_not_ok(like)
    if order != "C":
        raise NotImplementedError
    if dtype is None:
        dtype = _dtypes_impl.default_float_dtype
    result = torch.empty(shape, dtype=dtype)
    return asarray(result)


# NB: *_like function deliberately deviate from numpy: it has subok=True
# as the default; we set subok=False and raise on anything else.
@asarray_replacer()
@_decorators.dtype_to_torch
def empty_like(prototype, dtype=None, order="K", subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != "K":
        raise NotImplementedError
    result = _impl.empty_like(prototype, dtype=dtype, shape=shape)
    return result


@_decorators.dtype_to_torch
def full(shape, fill_value, dtype=None, order="C", *, like=None):
    _util.subok_not_ok(like)
    if order != "C":
        raise NotImplementedError
    fill_value = asarray(fill_value).get()
    result = _impl.full(shape, fill_value, dtype=dtype)
    return asarray(result)


@asarray_replacer()
@_decorators.dtype_to_torch
def full_like(a, fill_value, dtype=None, order="K", subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != "K":
        raise NotImplementedError
    result = _impl.full_like(a, fill_value, dtype=dtype, shape=shape)
    return result


@_decorators.dtype_to_torch
def ones(shape, dtype=None, order="C", *, like=None):
    _util.subok_not_ok(like)
    if order != "C":
        raise NotImplementedError
    if dtype is None:
        dtype = _dtypes_impl.default_float_dtype
    result = torch.ones(shape, dtype=dtype)
    return asarray(result)


@asarray_replacer()
@_decorators.dtype_to_torch
def ones_like(a, dtype=None, order="K", subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != "K":
        raise NotImplementedError
    result = _impl.ones_like(a, dtype=dtype, shape=shape)
    return result


@_decorators.dtype_to_torch
def zeros(shape, dtype=None, order="C", *, like=None):
    _util.subok_not_ok(like)
    if order != "C":
        raise NotImplementedError
    if dtype is None:
        dtype = _dtypes_impl.default_float_dtype
    result = torch.zeros(shape, dtype=dtype)
    return asarray(result)


@asarray_replacer()
@_decorators.dtype_to_torch
def zeros_like(a, dtype=None, order="K", subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != "K":
        raise NotImplementedError
    result = _impl.zeros_like(a, dtype=dtype, shape=shape)
    return result


@_decorators.dtype_to_torch
def eye(N, M=None, k=0, dtype=float, order="C", *, like=None):
    _util.subok_not_ok(like)
    if order != "C":
        raise NotImplementedError
    result = _impl.eye(N, M, k, dtype)
    return asarray(result)


def identity(n, dtype=None, *, like=None):
    _util.subok_not_ok(like)
    return asarray(torch.eye(n, dtype=dtype))


def diag(v, k=0):
    v_tensor = asarray(v).get()
    result = torch.diag(v_tensor, k)
    return asarray(result)


###### misc/unordered


def _xy_helper_corrcoef(x_tensor, y_tensor=None, rowvar=True):
    """Prepate inputs for cov and corrcoef."""

    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/function_base.py#L2636
    if y_tensor is not None:
        # make sure x and y are at least 2D
        ndim_extra = 2 - x_tensor.ndim
        if ndim_extra > 0:
            x_tensor = x_tensor.view((1,) * ndim_extra + x_tensor.shape)
        if not rowvar and x_tensor.shape[0] != 1:
            x_tensor = x_tensor.mT
        x_tensor = x_tensor.clone()

        ndim_extra = 2 - y_tensor.ndim
        if ndim_extra > 0:
            y_tensor = y_tensor.view((1,) * ndim_extra + y_tensor.shape)
        if not rowvar and y_tensor.shape[0] != 1:
            y_tensor = y_tensor.mT
        y_tensor = y_tensor.clone()

        x_tensor = _impl.concatenate((x_tensor, y_tensor), axis=0)

    return x_tensor


@_decorators.dtype_to_torch
def corrcoef(x, y=None, rowvar=True, bias=NoValue, ddof=NoValue, *, dtype=None):
    if bias is not None or ddof is not None:
        # deprecated in NumPy
        raise NotImplementedError

    x_tensor, y_tensor = _helpers.to_tensors_or_none(x, y)
    tensor = _xy_helper_corrcoef(x_tensor, y_tensor, rowvar)
    result = _impl.corrcoef(tensor, dtype=dtype)
    return asarray(result)


@_decorators.dtype_to_torch
def cov(
    m,
    y=None,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights=None,
    aweights=None,
    *,
    dtype=None,
):

    m_tensor, y_tensor, fweights_tensor, aweights_tensor = _helpers.to_tensors_or_none(
        m, y, fweights, aweights
    )
    m_tensor = _xy_helper_corrcoef(m_tensor, y_tensor, rowvar)

    result = _impl.cov(
        m_tensor, bias, ddof, fweights_tensor, aweights_tensor, dtype=dtype
    )
    return asarray(result)


def bincount(x, /, weights=None, minlength=0):
    if not isinstance(x, ndarray) and x == []:
        # edge case allowed by numpy
        x = asarray([], dtype=int)

    x_tensor, weights_tensor = _helpers.to_tensors_or_none(x, weights)
    result = _impl.bincount(x_tensor, weights_tensor, minlength)
    return asarray(result)


# YYY: pattern: sequence of arrays
def where(condition, x=None, y=None, /):
    selector = (x is None) == (y is None)
    if not selector:
        raise ValueError("either both or neither of x and y should be given")
    condition = asarray(condition).get()
    if x is None and y is None:
        return tuple(asarray(_) for _ in torch.where(condition))
    x = asarray(condition).get()
    y = asarray(condition).get()
    return asarray(torch.where(condition, x, y))


###### module-level queries of object properties


def ndim(a):
    a = asarray(a).get()
    return a.ndim


def shape(a):
    a = asarray(a).get()
    return tuple(a.shape)


def size(a, axis=None):
    a = asarray(a).get()
    if axis is None:
        return a.numel()
    else:
        return a.shape[axis]


###### shape manipulations and indexing


def transpose(a, axes=None):
    arr = asarray(a)
    return arr.transpose(axes)


def reshape(a, newshape, order="C"):
    arr = asarray(a)
    return arr.reshape(*newshape, order=order)


def ravel(a, order="C"):
    arr = asarray(a)
    return arr.ravel(order=order)


def squeeze(a, axis=None):
    arr = asarray(a)
    return arr.squeeze(axis)


def expand_dims(a, axis):
    a = asarray(a)
    shape = _util.expand_shape(a.shape, axis)
    tensor = a.get().view(shape)  # never copies
    return ndarray._from_tensor_and_base(tensor, a)


@asarray_replacer()
def flip(m, axis=None):
    return _flips.flip(m, axis)


@asarray_replacer()
def flipud(m):
    return _flips.flipud(m)


@asarray_replacer()
def fliplr(m):
    return _flips.fliplr(m)


@asarray_replacer()
def rot90(m, k=1, axes=(0, 1)):
    return _flips.rot90(m, k, axes)


@asarray_replacer()
def broadcast_to(array, shape, subok=False):
    _util.subok_not_ok(subok=subok)
    return torch.broadcast_to(array, size=shape)


from torch import broadcast_shapes


# YYY: pattern: tuple of arrays as input, tuple of arrays as output; cf nonzero
def broadcast_arrays(*args, subok=False):
    _util.subok_not_ok(subok=subok)
    tensors = _helpers.to_tensors(*args)
    res = torch.broadcast_tensors(*tensors)
    return tuple(asarray(_) for _ in res)


@asarray_replacer()
def moveaxis(a, source, destination):
    source = _util.normalize_axis_tuple(source, a.ndim, "source")
    destination = _util.normalize_axis_tuple(destination, a.ndim, "destination")
    return asarray(torch.moveaxis(a, source, destination))


def swapaxes(a, axis1, axis2):
    arr = asarray(a)
    return arr.swapaxes(axis1, axis2)


@asarray_replacer()
def rollaxis(a, axis, start=0):
    return _flips.rollaxis(a, axis, start)


def unravel_index(indices, shape, order="C"):
    # cf https://github.com/pytorch/pytorch/pull/66687
    # this version is from
    # https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/3
    if order != "C":
        raise NotImplementedError
    result = []
    for index in indices:
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        result.append(tuple(reversed(out)))
    return result


def ravel_multi_index(multi_index, dims, mode="raise", order="C"):
    # XXX: not available in pytorch, implement
    return sum(idx * dim for idx, dim in zip(multi_index, dims))


def meshgrid(*xi, copy=True, sparse=False, indexing="xy"):
    xi_tensors = _helpers.to_tensors(*xi)
    output = _impl.meshgrid(*xi_tensors, copy=copy, sparse=sparse, indexing=indexing)
    return [asarray(t) for t in output]


def nonzero(a):
    arr = asarray(a)
    return arr.nonzero()


def flatnonzero(a):
    arr = asarray(a)
    return nonzero(arr.ravel())[0]


def argwhere(a):
    arr = asarray(a)
    tensor = arr.get()
    return asarray(torch.argwhere(tensor))


from ._decorators import emulate_out_arg
from ._ndarray import axis_keepdims_wrapper

count_nonzero = emulate_out_arg(axis_keepdims_wrapper(_reductions.count_nonzero))


@asarray_replacer()
def roll(a, shift, axis=None):
    if axis is not None:
        axis = _util.normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        if not isinstance(shift, tuple):
            shift = (shift,) * len(axis)
    return a.roll(shift, axis)


def round_(a, decimals=0, out=None):
    arr = asarray(a)
    return arr.round(decimals, out=out)


around = round_
round = round_


def clip(a, a_min, a_max, out=None):
    arr = asarray(a)
    return arr.clip(a_min, a_max, out=out)


###### tri{l, u} and related
@asarray_replacer()
def tril(m, k=0):
    return m.tril(k)


@asarray_replacer()
def triu(m, k=0):
    return m.triu(k)


def tril_indices(n, k=0, m=None):
    result = _impl.tril_indices(n, k, m)
    return tuple(asarray(t) for t in result)


def triu_indices(n, k=0, m=None):
    result = _impl.triu_indices(n, k, m)
    return tuple(asarray(t) for t in result)


def tril_indices_from(arr, k=0):
    tensor = asarray(arr).get()
    result = _impl.tril_indices_from(tensor, k)
    return tuple(asarray(t) for t in result)


def triu_indices_from(arr, k=0):
    tensor = asarray(arr).get()
    result = _impl.triu_indices_from(tensor, k)
    return tuple(asarray(t) for t in result)


@_decorators.dtype_to_torch
def tri(N, M=None, k=0, dtype=float, *, like=None):
    _util.subok_not_ok(like)
    result = _impl.tri(N, M, k, dtype)
    return asarray(result)


###### reductions
def argmax(a, axis=None, out=None, *, keepdims=NoValue):
    arr = asarray(a)
    return arr.argmax(axis=axis, out=out, keepdims=keepdims)


def argmin(a, axis=None, out=None, *, keepdims=NoValue):
    arr = asarray(a)
    return arr.argmin(axis=axis, out=out, keepdims=keepdims)


def amax(a, axis=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    arr = asarray(a)
    return arr.max(axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)


max = amax


def amin(a, axis=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    arr = asarray(a)
    return arr.min(axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)


min = amin


def ptp(a, axis=None, out=None, keepdims=NoValue):
    arr = asarray(a)
    return arr.ptp(axis=axis, out=out, keepdims=keepdims)


def all(a, axis=None, out=None, keepdims=NoValue, *, where=NoValue):
    arr = asarray(a)
    return arr.all(axis=axis, out=out, keepdims=keepdims, where=where)


def any(a, axis=None, out=None, keepdims=NoValue, *, where=NoValue):
    arr = asarray(a)
    return arr.any(axis=axis, out=out, keepdims=keepdims, where=where)


def mean(a, axis=None, dtype=None, out=None, keepdims=NoValue, *, where=NoValue):
    arr = asarray(a)
    return arr.mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)


# YYY: pattern: initial=...


def sum(
    a, axis=None, dtype=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue
):
    arr = asarray(a)
    return arr.sum(
        axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where
    )


def prod(
    a, axis=None, dtype=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue
):
    arr = asarray(a)
    return arr.prod(
        axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where
    )


product = prod


def cumprod(a, axis=None, dtype=None, out=None):
    arr = asarray(a)
    return arr.cumprod(axis=axis, dtype=dtype, out=out)


cumproduct = cumprod


def cumsum(a, axis=None, dtype=None, out=None):
    arr = asarray(a)
    return arr.cumsum(axis=axis, dtype=dtype, out=out)


# YYY: pattern : ddof


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue):
    arr = asarray(a)
    return arr.std(
        axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where
    )


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue):
    arr = asarray(a)
    return arr.var(
        axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where
    )


def average(a, axis=None, weights=None, returned=False, *, keepdims=NoValue):

    if weights is None:
        result = mean(a, axis=axis, keepdims=keepdims)
        if returned:
            scl = result.dtype.type(a.size / result.size)
            return result, scl
        return result

    a_tensor, w_tensor = _helpers.to_tensors(a, weights)

    result, wsum = _reductions.average(a_tensor, axis, w_tensor)

    # keepdims
    if keepdims:
        result = _util.apply_keepdims(result, axis, a_tensor.ndim)

    # returned
    if returned:
        scl = wsum
        if scl.shape != result.shape:
            scl = torch.broadcast_to(scl, result.shape).clone()

        return asarray(result), asarray(scl)
    else:
        return asarray(result)


def percentile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None,
):
    return quantile(
        a, asarray(q) / 100.0, axis, out, overwrite_input, method, keepdims=keepdims
    )


def quantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None,
):
    if interpolation is not None:
        raise ValueError("'interpolation' argument is deprecated; use 'method' instead")

    a_tensor, q_tensor = _helpers.to_tensors(a, q)
    result = _reductions.quantile(a_tensor, q_tensor, axis, method)

    # keepdims
    if keepdims:
        result = _util.apply_keepdims(result, axis, a_tensor.ndim)
    return _helpers.result_or_out(result, out, promote_scalar=True)


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    return quantile(
        a, 0.5, axis=axis, overwrite_input=overwrite_input, out=out, keepdims=keepdims
    )


@asarray_replacer()
def nanmean(a, axis=None, dtype=None, out=None, keepdims=NoValue, *, where=NoValue):
    if where is not NoValue:
        raise NotImplementedError
    if dtype is None:
        dtype = a.dtype
    if axis is None:
        result = a.nanmean(dtype=dtype)
        if keepdims:
            result = torch.full(a.shape, result, dtype=result.dtype)
    else:
        result = a.nanmean(dtype=dtype, dim=axis, keepdim=bool(keepdims))
    if out is not None:
        out.copy_(result)
    return result


def nanmin():
    raise NotImplementedError


def nanmax():
    raise NotImplementedError


def nanvar():
    raise NotImplementedError


def nanstd():
    raise NotImplementedError


def nanargmin():
    raise NotImplementedError


def nanargmax():
    raise NotImplementedError


def nansum():
    raise NotImplementedError


def nanprod():
    raise NotImplementedError


def nancumsum():
    raise NotImplementedError


def nancumprod():
    raise NotImplementedError


def nanmedian():
    raise NotImplementedError


def nanquantile():
    raise NotImplementedError


def nanpercentile():
    raise NotImplementedError


def diff(a, n=1, axis=-1, prepend=NoValue, append=NoValue):

    if n == 0:
        # match numpy and return the input immediately
        return a

    a_tensor, prepend_tensor, append_tensor = _helpers.to_tensors_or_none(
        a, prepend, append
    )

    result = _impl.diff(
        a_tensor,
        n=n,
        axis=axis,
        prepend_tensor=prepend_tensor,
        append_tensor=append_tensor,
    )
    return asarray(result)


@asarray_replacer()
def argsort(a, axis=-1, kind=None, order=None):
    result = _impl.argsort(a, axis, kind, order)
    return result


##### math functions


@asarray_replacer()
def angle(z, deg=False):
    result = _impl.angle(z, deg)
    return result


@asarray_replacer()
def sinc(x):
    return torch.sinc(x)


def real(a):
    arr = asarray(a)
    return arr.real


def imag(a):
    arr = asarray(a)
    return arr.imag


@asarray_replacer()
def real_if_close(a, tol=100):
    result = _impl.real_if_close(a, tol=tol)
    return result


@asarray_replacer()
def iscomplex(x):
    result = _impl.iscomplex(x)
    return result  # XXX: missing .item on a zero-dim value; a case for array_or_scalar(value) ?


@asarray_replacer()
def isreal(x):
    result = _impl.isreal(x)
    return result


@asarray_replacer()
def iscomplexobj(x):
    return torch.is_complex(x)


@asarray_replacer()
def isrealobj(x):
    return not torch.is_complex(x)


@asarray_replacer()
def isneginf(x, out=None):
    return torch.isneginf(x, out=out)


@asarray_replacer()
def isposinf(x, out=None):
    return torch.isposinf(x, out=out)


@asarray_replacer()
def i0(x):
    return torch.special.i0(x)


def isscalar(a):
    # XXX: this is a stub
    try:
        arr = asarray(a)
        return arr.size == 1
    except Exception:
        return False


def isclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    a_t, b_t = _helpers.to_tensors(a, b)
    result = _impl.isclose(a_t, b_t, rtol, atol, equal_nan=equal_nan)
    return asarray(result)


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a_t, b_t = _helpers.to_tensors(a, b)
    result = _impl.isclose(a_t, b_t, rtol, atol, equal_nan=equal_nan)
    return result.all()


def array_equal(a1, a2, equal_nan=False):
    a1_t, a2_t = _helpers.to_tensors(a1, a2)
    result = _impl.tensor_equal(a1_t, a2_t, equal_nan)
    return result


def array_equiv(a1, a2):
    a1_t, a2_t = _helpers.to_tensors(a1, a2)
    result = _impl.tensor_equiv(a1_t, a2_t)
    return result


def common_type():
    raise NotImplementedError


def mintypecode():
    raise NotImplementedError


def nan_to_num():
    raise NotImplementedError


def asfarray():
    raise NotImplementedError


###### mapping from numpy API objects to wrappers from this module ######

# All is in the mapping dict in _mapping.py
