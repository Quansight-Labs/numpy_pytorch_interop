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
    res = torch.atleast_1d([asarray(a).get() for a in arys])
    if len(res) == 1:
        return asarray(res[0])
    else:
        return list(asarray(_) for _ in res)


def atleast_2d(*arys):
    res = torch.atleast_2d([asarray(a).get() for a in arys])
    if len(res) == 1:
        return asarray(res[0])
    else:
        return list(asarray(_) for _ in res)


def atleast_3d(*arys):
    res = torch.atleast_3d([asarray(a).get() for a in arys])
    if len(res) == 1:
        return asarray(res[0])
    else:
        return list(asarray(_) for _ in res)


def vstack(tup, *, dtype=None, casting="same_kind"):
    arrs = atleast_2d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return concatenate(arrs, 0, dtype=dtype, casting=casting)


row_stack = vstack


def hstack(tup, *, dtype=None, casting="same_kind"):
    arrs = atleast_1d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
    if arrs and arrs[0].ndim == 1:
        return concatenate(arrs, 0, dtype=dtype, casting=casting)
    else:
        return concatenate(arrs, 1, dtype=dtype, casting=casting)


def dstack(tup, *, dtype=None, casting="same_kind"):
    # XXX: in numpy 1.24 dstack does not have dtype and casting keywords
    # but {h,v}stack do.  Hence add them here for consistency.
    arrs = atleast_3d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return concatenate(arrs, 2, dtype=dtype, casting=casting)


def column_stack(tup, *, dtype=None, casting="same_kind"):
    # XXX: in numpy 1.24 column_stack does not have dtype and casting keywords
    # but row_stack does. (because row_stack is an alias for vstack, really).
    # Hence add these keywords here for consistency.
    arrays = []
    for v in tup:
        arr = asarray(v)
        if arr.ndim < 2:
            arr = array(arr, copy=False, ndmin=2).T
        arrays.append(arr)
    return concatenate(arrays, 1, dtype=dtype, casting=casting)


def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    arrays = [asarray(arr) for arr in arrays]
    if not arrays:
        raise ValueError("need at least one array to stack")

    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError("all input arrays must have the same shape")

    result_ndim = arrays[0].ndim + 1
    axis = _util.normalize_axis_index(axis, result_ndim)

    sl = (slice(None),) * axis + (newaxis,)
    expanded_arrays = [arr[sl] for arr in arrays]
    return concatenate(
        expanded_arrays, axis=axis, out=out, dtype=dtype, casting=casting
    )


def array_split(ary, indices_or_sections, axis=0):
    tensor = asarray(ary).get()
    base = ary if isinstance(ary, ndarray) else None
    axis = _util.normalize_axis_index(axis, tensor.ndim)

    result = _impl.split_helper(tensor, indices_or_sections, axis)

    return tuple(maybe_set_base(x, base) for x in result)


def split(ary, indices_or_sections, axis=0):
    tensor = asarray(ary).get()
    base = ary if isinstance(ary, ndarray) else None
    axis = _util.normalize_axis_index(axis, tensor.ndim)

    result = _impl.split_helper(tensor, indices_or_sections, axis, strict=True)

    return tuple(maybe_set_base(x, base) for x in result)


def hsplit(ary, indices_or_sections):
    tensor = asarray(ary).get()
    base = ary if isinstance(ary, ndarray) else None

    if tensor.ndim == 0:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")

    axis = 1 if tensor.ndim > 1 else 0

    result = _impl.split_helper(tensor, indices_or_sections, axis, strict=True)

    return tuple(maybe_set_base(x, base) for x in result)


def vsplit(ary, indices_or_sections):
    tensor = asarray(ary).get()
    base = ary if isinstance(ary, ndarray) else None

    if tensor.ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    result = _impl.split_helper(tensor, indices_or_sections, 0, strict=True)

    return tuple(maybe_set_base(x, base) for x in result)


def dsplit(ary, indices_or_sections):
    tensor = asarray(ary).get()
    base = ary if isinstance(ary, ndarray) else None

    if tensor.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    result = _impl.split_helper(tensor, indices_or_sections, 2, strict=True)

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


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    if axis != 0 or not endpoint:
        raise NotImplementedError
    tstart, tstop = torch.as_tensor([start, stop])
    base = torch.pow(tstop / tstart, 1.0 / (num - 1))
    result = torch.logspace(
        torch.log(tstart) / torch.log(base),
        torch.log(tstop) / torch.log(base),
        num,
        base=base,
    )
    return asarray(result)


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    if axis != 0 or not endpoint:
        raise NotImplementedError
    return asarray(torch.logspace(start, stop, num, base=base, dtype=dtype))


def arange(start=None, stop=None, step=1, dtype=None, *, like=None):
    _util.subok_not_ok(like)
    if step == 0:
        raise ZeroDivisionError
    if stop is None and start is None:
        raise TypeError
    if stop is None:
        # XXX: this breaks if start is passed as a kwarg:
        # arange(start=4) should raise (no stop) but doesn't
        start, stop = 0, start
    if start is None:
        start = 0

    if dtype is None:
        dtype = _dtypes.default_int_type()
        dtype = result_type(start, stop, step, dtype)
    torch_dtype = _dtypes.torch_dtype_from(dtype)
    start, stop, step = _helpers.ndarrays_to_tensors(start, stop, step)

    try:
        return asarray(torch.arange(start, stop, step, dtype=torch_dtype))
    except RuntimeError:
        raise ValueError("Maximum allowed size exceeded")


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
def empty_like(prototype, dtype=None, order="K", subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != "K":
        raise NotImplementedError
    torch_dtype = None if dtype is None else _dtypes.torch_dtype_from(dtype)
    result = torch.empty_like(prototype, dtype=torch_dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


@_decorators.dtype_to_torch
def full(shape, fill_value, dtype=None, order="C", *, like=None):
    _util.subok_not_ok(like)
    if order != "C":
        raise NotImplementedError

    fill_value = asarray(fill_value).get()
    if dtype is None:
        dtype = fill_value.dtype

    if not isinstance(shape, (tuple, list)):
        shape = (shape,)

    result = torch.full(shape, fill_value, dtype=dtype)

    return asarray(result)


@asarray_replacer()
def full_like(a, fill_value, dtype=None, order="K", subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != "K":
        raise NotImplementedError
    torch_dtype = None if dtype is None else _dtypes.torch_dtype_from(dtype)
    result = torch.full_like(a, fill_value, dtype=torch_dtype)
    if shape is not None:
        result = result.reshape(shape)
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
def ones_like(a, dtype=None, order="K", subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != "K":
        raise NotImplementedError
    torch_dtype = None if dtype is None else _dtypes.torch_dtype_from(dtype)
    result = torch.ones_like(a, dtype=torch_dtype)
    if shape is not None:
        result = result.reshape(shape)
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
def zeros_like(a, dtype=None, order="K", subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != "K":
        raise NotImplementedError
    torch_dtype = None if dtype is None else _dtypes.torch_dtype_from(dtype)
    result = torch.zeros_like(a, dtype=torch_dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


@_decorators.dtype_to_torch
def eye(N, M=None, k=0, dtype=float, order="C", *, like=None):
    _util.subok_not_ok(like)
    if order != "C":
        raise NotImplementedError
    if M is None:
        M = N
    z = torch.zeros(N, M, dtype=dtype)
    z.diagonal(k).fill_(1)
    return asarray(z)


def identity(n, dtype=None, *, like=None):
    _util.subok_not_ok(like)
    return asarray(torch.eye(n, dtype=dtype))


def diag(v, k=0):
    v_tensor = asarray(v).get()
    result = torch.diag(v_tensor, k)
    return asarray(result)


###### misc/unordered


@_decorators.dtype_to_torch
def corrcoef(x, y=None, rowvar=True, bias=NoValue, ddof=NoValue, *, dtype=None):
    if bias is not None or ddof is not None:
        # deprecated in NumPy
        raise NotImplementedError

    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/function_base.py#L2636
    if y is not None:
        x = array(x, ndmin=2)
        if not rowvar and x.shape[0] != 1:
            x = x.T

        y = array(y, ndmin=2)
        if not rowvar and y.shape[0] != 1:
            y = y.T

        x = concatenate((x, y), axis=0)

    x_tensor = asarray(x).get()
    result = _impl.corrcoef(x_tensor, rowvar, dtype=dtype)
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
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/function_base.py#L2636
    if y is not None:
        m = array(m, ndmin=2)
        if not rowvar and m.shape[0] != 1:
            m = m.T

        y = array(y, ndmin=2)
        if not rowvar and y.shape[0] != 1:
            y = y.T

        m = concatenate((m, y), axis=0)

#    if ddof is None:
#        if bias == 0:
#            ddof = 1
#        else:
#            ddof = 0

    m_tensor, fweights_tensor, aweights_tensor = _helpers.to_tensors_or_none(
        m, fweights, aweights
    )
    result = _impl.cov(m_tensor, bias, ddof, fweights_tensor, aweights_tensor, dtype=dtype)
    return asarray(result)


@_decorators.dtype_to_torch
def concatenate(ar_tuple, axis=0, out=None, dtype=None, casting="same_kind"):
    if ar_tuple == ():
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
    tensors = _helpers.to_tensors(*ar_tuple)
    result = _impl.concatenate(tensors, axis, out, dtype, casting)
    return _helpers.result_or_out(result, out)


def bincount(x, /, weights=None, minlength=0):
    if not isinstance(x, ndarray) and x == []:
        # edge case allowed by numpy
        x = asarray([], dtype=int)

    x_tensor, weights_tensor = _helpers.to_tensors_or_none(x, weights)
    int_dtype = _dtypes_impl.default_int_dtype
    (x_tensor,) = _util.cast_dont_broadcast((x_tensor,), int_dtype, casting="safe")

    result = torch.bincount(x_tensor, weights_tensor, minlength)
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
    res = torch.broadcast_tensors(*[asarray(a).get() for a in args])
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


# YYY: pattern: return sequence
def tril_indices(n, k=0, m=None):
    if m is None:
        m = n
    tensor_2 = torch.tril_indices(n, m, offset=k)
    return tuple(asarray(_) for _ in tensor_2)


def triu_indices(n, k=0, m=None):
    if m is None:
        m = n
    tensor_2 = torch.tril_indices(n, m, offset=k)
    return tuple(asarray(_) for _ in tensor_2)


# YYY: pattern: array in, sequence of arrays out
def tril_indices_from(arr, k=0):
    arr = asarray(arr).get()
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    tensor_2 = torch.tril_indices(arr.shape[0], arr.shape[1], offset=k)
    return tuple(asarray(_) for _ in tensor_2)


def triu_indices_from(arr, k=0):
    arr = asarray(arr).get()
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    tensor_2 = torch.tril_indices(arr.shape[0], arr.shape[1], offset=k)
    return tuple(asarray(_) for _ in tensor_2)


@_decorators.dtype_to_torch
def tri(N, M=None, k=0, dtype=float, *, like=None):
    _util.subok_not_ok(like)
    if M is None:
        M = N
    tensor = torch.ones((N, M), dtype=dtype)
    tensor = torch.tril(tensor, diagonal=k)
    return asarray(tensor)


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
    if order is not None:
        raise NotImplementedError
    stable = True if kind == "stable" else False
    if axis is None:
        axis = -1
    return torch.argsort(a, stable=stable, dim=axis, descending=False)


##### math functions


@asarray_replacer()
def angle(z, deg=False):
    result = torch.angle(z)
    if deg:
        result *= 180 / torch.pi
    return asarray(result)


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
    if not torch.is_complex(a):
        return a
    if torch.abs(torch.imag) < tol * torch.finfo(a.dtype).eps:
        return torch.real(a)
    else:
        return a


@asarray_replacer()
def iscomplex(x):
    if torch.is_complex(x):
        return torch.as_tensor(x).imag != 0
    result = torch.zeros_like(x, dtype=torch.bool)
    return result[()]


@asarray_replacer()
def isreal(x):
    if torch.is_complex(x):
        return torch.as_tensor(x).imag == 0
    result = torch.zeros_like(x, dtype=torch.bool)
    return result[()]


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
    a, b = _helpers.to_tensors(a, b)
    dtype = result_type(a, b)
    torch_dtype = dtype.type.torch_dtype
    a = a.to(torch_dtype)
    b = b.to(torch_dtype)
    return asarray(torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    arr_res = isclose(a, b, rtol, atol, equal_nan)
    return arr_res.all()


def array_equal(a1, a2, equal_nan=False):
    a1_t, a2_t = _helpers.to_tensors(a1, a2)
    result = _impl.tensor_equal(a1_t, a2_t, equal_nan)
    return result


def array_equiv(a1, a2):
    a1_t, a2_t = _helpers.to_tensors(a1, a2)
    try:
        a1_t, a2_t = torch.broadcast_tensors(a1_t, a2_t)
    except RuntimeError:
        # failed to broadcast => not equivalent
        return False
    return _impl.tensor_equal(a1_t, a2_t)


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
