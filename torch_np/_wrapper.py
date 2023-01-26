"""A thin pytorch / numpy compat layer.

Things imported from here have numpy-compatible signatures but operate on
pytorch tensors.
"""
#import numpy as np

import torch

from ._detail import _util
from ._detail import _reductions

from . import _dtypes
from . import _helpers
from ._ndarray import ndarray, asarray, array, asarray_replacer, newaxis
from ._ndarray import can_cast, result_type




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
#
#  5. handle the out= arg: verify dimensions, handle dtype (blocked on dtype decision)


NoValue = None



###### array creation routines


def copy(a, order='K', subok=False):
    a = asarray(a)
    _util.subok_not_ok(subok=subok)
    if order != 'K':
        raise NotImplementedError
    # XXX: ndarray.copy only accepts order='C'
    return a.copy(order='C')


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


def vstack(tup, *, dtype=None, casting='same_kind'):
    arrs = atleast_2d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return concatenate(arrs, 0, dtype=dtype, casting=casting)


row_stack = vstack


def hstack(tup, *, dtype=None, casting='same_kind'):
    arrs = atleast_1d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
    if arrs and arrs[0].ndim == 1:
        return concatenate(arrs, 0, dtype=dtype, casting=casting)
    else:
        return concatenate(arrs, 1, dtype=dtype, casting=casting)


def dstack(tup, *, dtype=None, casting='same_kind'):
    # XXX: in numpy 1.24 dstack does not have dtype and casting keywords
    # but {h,v}stack do.  Hence add them here for consistency.
    arrs = atleast_3d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return concatenate(arrs, 2, dtype=dtype, casting=casting)


def column_stack(tup, *, dtype=None, casting='same_kind'):
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


def stack(arrays, axis=0, out=None, *, dtype=None, casting='same_kind'):
    arrays = [asarray(arr) for arr in arrays]
    if not arrays:
        raise ValueError('need at least one array to stack')

    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError('all input arrays must have the same shape')

    result_ndim = arrays[0].ndim + 1
    axis = _util.normalize_axis_index(axis, result_ndim)

    sl = (slice(None),) * axis + (newaxis,)
    expanded_arrays = [arr[sl] for arr in arrays]
    return concatenate(expanded_arrays, axis=axis, out=out,
                       dtype=dtype, casting=casting)



def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
             axis=0):
    if axis != 0 or retstep or not endpoint:
        raise NotImplementedError
    # XXX: raises TypeError if start or stop are not scalars
    return asarray(torch.linspace(start, stop, num, dtype=dtype))


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    if axis != 0 or not endpoint:
        raise NotImplementedError
    tstart, tstop = torch.as_tensor([start, stop])
    base = torch.pow(tstop / tstart, 1./(num-1))
    result = torch.logspace(torch.log(tstart)/torch.log(base),
                            torch.log(tstop)/torch.log(base), num, base=base)
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


def empty(shape, dtype=float, order='C', *, like=None):
    _util.subok_not_ok(like)
    if order != 'C':
        raise NotImplementedError
    torch_dtype = _dtypes.torch_dtype_from(dtype)
    return asarray(torch.empty(shape, dtype=torch_dtype))


# NB: *_like function deliberately deviate from numpy: it has subok=True
# as the default; we set subok=False and raise on anything else.
@asarray_replacer()
def empty_like(prototype, dtype=None, order='K', subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != 'K':
        raise NotImplementedError
    torch_dtype = None if dtype is None else _dtypes.torch_dtype_from(dtype)
    result = torch.empty_like(prototype, dtype=torch_dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def full(shape, fill_value, dtype=None, order='C', *, like=None):
    _util.subok_not_ok(like)
    if order != 'C':
        raise NotImplementedError
    if isinstance(fill_value, ndarray):
        fill_value = fill_value.get()
    torch_dtype = _dtypes.torch_dtype_from(dtype)
    return asarray(torch.full(shape, fill_value, dtype=torch_dtype))


@asarray_replacer()
def full_like(a, fill_value, dtype=None, order='K', subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != 'K':
        raise NotImplementedError
    torch_dtype = None if dtype is None else _dtypes.torch_dtype_from(dtype)
    result = torch.full_like(a, fill_value, dtype=torch_dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def ones(shape, dtype=None, order='C', *, like=None):
    _util.subok_not_ok(like)
    if order != 'C':
        raise NotImplementedError
    torch_dtype = _dtypes.torch_dtype_from(dtype)
    return asarray(torch.ones(shape, dtype=torch_dtype))


@asarray_replacer()
def ones_like(a, dtype=None, order='K', subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != 'K':
        raise NotImplementedError
    torch_dtype = None if dtype is None else _dtypes.torch_dtype_from(dtype)
    result = torch.ones_like(a, dtype=torch_dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


# XXX: dtype=float
def zeros(shape, dtype=float, order='C', *, like=None):
    _util.subok_not_ok(like)
    if order != 'C':
        raise NotImplementedError
    torch_dtype = _dtypes.torch_dtype_from(dtype)
    return asarray(torch.zeros(shape, dtype=torch_dtype))


@asarray_replacer()
def zeros_like(a, dtype=None, order='K', subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != 'K':
        raise NotImplementedError
    torch_dtype = None if dtype is None else _dtypes.torch_dtype_from(dtype)
    result = torch.zeros_like(a, dtype=torch_dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


# XXX: dtype=float
def eye(N, M=None, k=0, dtype=float, order='C', *, like=None):
    _util.subok_not_ok(like)
    if order != 'C':
        raise NotImplementedError
    if M is None:
        M = N
    z = torch.zeros(N, M, dtype=dtype)
    z.diagonal(k).fill_(1)
    return asarray(z)


def identity(n, dtype=None, *, like=None):
    _util.subok_not_ok(like)
    return asarray(torch.eye(n, dtype=dtype))


###### misc/unordered




@asarray_replacer()
def corrcoef(x, y=None, rowvar=True, bias=NoValue, ddof=NoValue, *, dtype=None):
    if bias is not None or ddof is not None:
        # deprecated in NumPy
        raise NotImplementedError
    if y is not None:
        # go figure what it means, XXX
        raise NotImplementedError

    if rowvar is False:
        x = x.T
    if dtype is not None:
        x = x.to(dtype)
    return torch.corrcoef(x)


def concatenate(ar_tuple, axis=0, out=None, dtype=None, casting="same_kind"):
    if ar_tuple == ():
        # XXX: RuntimeError in torch, ValueError in numpy
        raise ValueError("need at least one array to concatenate")

    tensors = _helpers.to_tensors(*ar_tuple)

    # np.concatenate ravels if axis=None
    tensors, axis = _util.axis_none_ravel(*tensors, axis=axis)

    if out is not None:
        if not isinstance(out, ndarray):
            raise ValueError("'out' must be an array")

        if dtype is not None:
            # mimic numpy
            raise TypeError("concatenate() only takes `out` or `dtype` as an "
                            "argument, but both were provided.")

    # figure out the type of the inputs and outputs
    if out is None and dtype is None:
        out_dtype = None
    else:
        out_dtype = out.dtype if dtype is None else _dtypes.dtype(dtype)
        out_dtype = out_dtype.type.torch_dtype

        # cast input arrays if necessary; do not broadcast them agains `out`
        tensors = _util.cast_dont_broadcast(tensors, out_dtype, casting)

    try:
        result = torch.cat(tensors, axis)
    except (IndexError, RuntimeError):
        raise _util.AxisError

    return _helpers.result_or_out(result, out)



@asarray_replacer()
def bincount(x, /, weights=None, minlength=0):
    return torch.bincount(x, weights, minlength)

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


def reshape(a, newshape, order='C'):
    arr = asarray(a)
    return arr.reshape(*newshape, order=order)


def ravel(a, order='C'):
    arr = asarray(a)
    return arr.ravel(order=order)


def squeeze(a, axis=None):
    arr = asarray(a)
    return arr.squeeze(axis)


def expand_dims(a, axis):
    a = asarray(a)
    shape = _util.expand_shape(a.shape, axis)
    tensor = a.get().view(shape)    # never copies
    return ndarray._from_tensor_and_base(tensor, a)


@asarray_replacer()
def flip(m, axis=None):
    # XXX: semantic difference: np.flip returns a view, torch.flip copies
    if axis is None:
        axis = tuple(range(m.ndim))
    else:
        axis = _util.normalize_axis_tuple(axis, m.ndim)
    return torch.flip(m, axis)


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
    return asarray(torch.moveaxis(a, source, destination))


def unravel_index(indices, shape, order='C'):
# cf https://github.com/pytorch/pytorch/pull/66687
# this version is from
# https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/3
    if order != 'C':
        raise NotImplementedError
    result = []
    for index in indices:
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        result.append(tuple(reversed(out)))
    return result


def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    # XXX: not available in pytorch, implement
    return sum(idx*dim for idx, dim in zip(multi_index, dims))


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


from ._ndarray import axis_keepdims_wrapper
from ._decorators import emulate_out_arg
count_nonzero = emulate_out_arg(axis_keepdims_wrapper(_reductions.count_nonzero))


@asarray_replacer()
def roll(a, shift, axis=None):
    return a.roll(shift, axis)


@asarray_replacer()
def round_(a, decimals=0, out=None):
    if torch.is_floating_point(a):
        return torch.round(a, decimals=decimals, out=out)
    else:
        return a

around = round_


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


def tri(N, M=None, k=0, dtype=float, *, like=None):
    _util.subok_not_ok(like)
    tensor = torch.tril(torch.ones((N, M), dtype=dtype), diagonal=k)
    return asarray(tensor)

###### reductions
def argmax(a, axis=None, out=None, *, keepdims=NoValue):
    arr = asarray(a)
    return arr.argmax(axis=axis, out=out, keepdims=keepdims)


def argmin(a, axis=None, out=None, *, keepdims=NoValue):
    arr = asarray(a)
    return arr.argmin(axis=axis, out=out, keepdims=keepdims)


def amax(a, axis=None, out=None, keepdims=NoValue, initial=NoValue,
         where=NoValue):
    arr = asarray(a)
    return arr.max(axis=axis, out=out, keepdims=keepdims, initial=initial,
                    where=where)

max = amax


def amin(a, axis=None, out=None, keepdims=NoValue, initial=NoValue,
         where=NoValue):
    arr = asarray(a)
    return arr.min(axis=axis, out=out, keepdims=keepdims, initial=initial,
                    where=where)

min = amin


def all(a, axis=None, out=None, keepdims=NoValue, *, where=NoValue):
    arr = asarray(a)
    return arr.all(axis=axis, out=out, keepdims=keepdims, where=where)


def any(a, axis=None, out=None, keepdims=NoValue, *, where=NoValue):
    arr = asarray(a)
    return arr.any(axis=axis, out=out, keepdims=keepdims, where=where)


def mean(a, axis=None, dtype=None, out=None, keepdims=NoValue, *, where=NoValue):
    arr = asarray(a)
    return arr.mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)


#YYY: pattern: initial=...

def sum(a, axis=None, dtype=None, out=None, keepdims=NoValue,
        initial=NoValue, where=NoValue):
    arr = asarray(a)
    return arr.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                   initial=initial, where=where)


def prod(a, axis=None, dtype=None, out=None, keepdims=NoValue,
         initial=NoValue, where=NoValue):
    arr = asarray(a)
    return arr.prod(axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                   initial=initial, where=where)


#YYY: pattern : ddof

def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue):
    arr = asarray(a)
    return arr.std(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue):
    arr = asarray(a)
    return arr.var(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where)


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


@asarray_replacer()
def argsort(a, axis=-1, kind=None, order=None):
    if order is not None:
        raise NotImplementedError
    stable = True if kind == 'stable' else False
    if axis is None:
        axis = -1
    return torch.argsort(a, stable=stable, dim=axis, descending=False)


##### math functions

@asarray_replacer()
def angle(z, deg=False):
    result = torch.angle(z)
    if deg:
        result *= 180 / torch.pi
    return result


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


def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    a, b = _helpers.to_tensors(a, b)
    dtype = result_type(a, b)
    torch_dtype = dtype.type.torch_dtype
    a = a.to(torch_dtype)
    b = b.to(torch_dtype)
    return asarray(torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

###### mapping from numpy API objects to wrappers from this module ######

# All is in the mapping dict in _mapping.py


