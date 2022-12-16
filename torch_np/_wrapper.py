"""A thin pytorch / numpy compat layer.

Things imported from here have numpy-compatible signatures but operate on
pytorch tensors.
"""
import numpy as np


from . import _util
from ._ndarray import ndarray, asarray, array, asarray_replacer


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


@asarray_replacer()
def copy(a, order='K', subok=False):
    _util.subok_not_ok(subok=subok)
    if order != 'K':
        raise NotImplementedError
    return torch.clone(a)


def atleast_1d(*arys):
    res = torch.atleast_1d([asarray(a).get() for a in arys])
    if len(res) == 1:
        return asarray(res[0])
    else:
        return tuple(asarray(_) for _ in res)


def atleast_2d(*arys):
    res = torch.atleast_2d([asarray(a).get() for a in arys])
    if len(res) == 1:
        return asarray(res[0])
    else:
        return tuple(asarray(_) for _ in res)


def atleast_3d(*arys):
    res = torch.atleast_3d([asarray(a).get() for a in arys])
    if len(res) == 1:
        return asarray(res[0])
    else:
        return tuple(asarray(_) for _ in res)


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


def arange(start, stop=None, step=1, dtype=None, *, like=None):
    _util.subok_not_ok(like)
    if stop is None:
        # arange(stop)
        stop = start
        start = 0
    return asarray(torch.arange(start, stop, step, dtype=dtype))


def empty(shape, dtype=float, order='C', *, like=None):
    _util.subok_not_ok(like)
    if order != 'C':
        raise NotImplementedError
    return asarray(torch.empty(shape, dtype=dtype))


# NB: *_like function deliberately deviate from numpy: it has subok=True
# as the default; we set subok=False and raise on anything else.
@asarray_replacer()
def empty_like(prototype, dtype=None, order='K', subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != 'K':
        raise NotImplementedError
    result = torch.empty_like(prototype, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def full(shape, fill_value, dtype=None, order='C', *, like=None):
    _util.subok_not_ok(like)
    if order != 'C':
        raise NotImplementedError
    return asarray(torch.full(shape, fill_value, dtype=dtype))


@asarray_replacer()
def full_like(a, fill_value, dtype=None, order='K', subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != 'K':
        raise NotImplementedError
    result = torch.full_like(a, fill_value, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def ones(shape, dtype=None, order='C', *, like=None):
    _util.subok_not_ok(like)
    if order != 'C':
        raise NotImplementedError
    return asarray(torch.ones(shape, dtype=dtype))


@asarray_replacer()
def ones_like(a, dtype=None, order='K', subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != 'K':
        raise NotImplementedError
    result = torch.ones_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


# XXX: dtype=float
def zeros(shape, dtype=float, order='C', *, like=None):
    _util.subok_not_ok(like)
    if order != 'C':
        raise NotImplementedError
    return asarray(torch.zeros(shape, dtype=dtype))


@asarray_replacer()
def zeros_like(a, dtype=None, order='K', subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != 'K':
        raise NotImplementedError
    result = torch.zeros_like(a, dtype=dtype)
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


#YYY: pattern: initial=...
@asarray_replacer()
def prod(a, axis=None, dtype=None, out=None, keepdims=NoValue,
         initial=NoValue, where=NoValue):
    if initial is not None or where is not None:
        raise NotImplementedError
    if axis is None:
        if keepdims is not None:
            raise NotImplementedError
        return torch.prod(a, dtype=dtype)
    elif _util.is_sequence(axis):
        raise NotImplementedError
    return torch.prod(a, dim=axis, dtype=dtype, keepdim=bool(keepdims), out=out)


@asarray_replacer()
def sum(a, axis=None, dtype=None, out=None, keepdims=NoValue,
        initial=NoValue, where=NoValue):
    if initial is not None or where is not None:
        raise NotImplementedError
    if axis is None:
        if keepdims is not None:
            raise NotImplementedError
        return torch.sum(a, dtype=dtype)
    elif _util.is_sequence(axis):
        raise NotImplementedError
    return torch.sum(a, dim=axis, dtype=dtype, keepdim=bool(keepdims), out=out)


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
    if casting != "same_kind":
        raise NotImplementedError
    tensors = tuple(asarray(ar).get() for ar in ar_tuple)
    if dtype is not None:
        # XXX: map numpy dtypes
        tensors = tuple(ar.type(dtype) for ar in ar_typle)
    return asarray(torch.cat(tensors, axis, out=out))



def stack(arrays, axis=0, out=None):
    tensors = tuple(asarray(ar).get() for ar in arrays)
    return asarray(torch.stack(tensors, axis, out=out))


@asarray_replacer()
def squeeze(a, axis=None):
    if axis is None:
        return torch.squeeze(a)
    else:
        return torch.squeeze(a, axis)


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

@asarray_replacer()
def reshape(a, newshape, order='C'):
    if order != 'C':
        raise NotImplementedError
    return torch.reshape(a, newshape)


@asarray_replacer()
def expand_dims(a, axis):
    # taken from numpy 1.23.x
    if type(axis) not in (list, tuple):
        axis = (axis,)
    out_ndim = len(axis) + a.ndim
    axis = _util.normalize_axis_tuple(axis, out_ndim)
    shape_it = iter(a.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
    return a.reshape(shape)


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


# YYY : pattern: array_like input, tuple of arrays as output; cf broadcast_arrays
def nonzero(a):
    a = asarray(a).get()
    return tuple(asarray(_) for _ in a.nonzero(as_tuple=True))


@asarray_replacer()
def transpose(a, axes=None):
    if axes is None:
        axes = tuple(range(a.ndim))[::-1]
    return a.permute(axes)


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

# YYY: pattern : argmax, argmin

@asarray_replacer()
def argmax(a, axis=None, out=None, *, keepdims=NoValue):
    if axis is None:
        result = torch.argmax(a, keepdim=bool(keepdims))
    else:
        result = torch.argmax(a, axis, keepdim=bool(keepdims))
    if out is not None:
        out.copy_(result)
    return result


@asarray_replacer()
def argmin(a, axis=None, out=None, *, keepdims=NoValue):
    if axis is None:
        result = torch.argmin(a, keepdim=bool(keepdims))
    else:
        result = torch.argmin(a, axis, keepdim=bool(keepdims))
    if out is not None:
        out.copy_(result)
    return result


# YYY: pattern all, any

@asarray_replacer()
def all(a, axis=None, out=None, keepdims=NoValue, *, where=NoValue):
    if where is not None:
        raise NotImplementedError
    if axis is None:
        result = torch.all(a)
        # pytorch does not support keepdims=True and no axis
        if keepdims:
            result = torch.full(a.shape, result, dtype=result.dtype)
    else:
        result = torch.all(a, axis, keepdim=bool(keepdims))
    if out is not None:
        out.copy_(result)
    return result


@asarray_replacer()
def any(a, axis=None, out=None, keepdims=NoValue, *, where=NoValue):
    if where is not None:
        raise NotImplementedError
    if axis is None:
        result = a.any()
        if keepdims:
            result = torch.full(a.shape, result, dtype=result.dtype)
    else:
        result = a.any(axis, keepdim=bool(keepdims))
    if out is not None:
        out.copy_(result)
    return result


# YYY: pattern: dtype kwarg, None not accepted
@asarray_replacer()
def mean(a, axis=None, dtype=None, out=None, keepdims=NoValue, *, where=NoValue):
    if where is not None:
        raise NotImplementedError
    if dtype is None:
        dtype = a.dtype
    if axis is None:
        result = a.mean(dtype=dtype)
        if keepdims:
            result = torch.full(a.shape, result, dtype=result.dtype)
    else:
        result = a.mean(dtype=dtype, dim=axis, keepdim=bool(keepdims))
    if out is not None:
        out.copy_(result)
    return result


@asarray_replacer()
def nanmean(a, axis=None, dtype=None, out=None, keepdims=NoValue, *, where=NoValue):
    if where is not None:
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


# YYY: pattern : std, var
@asarray_replacer()
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue):
    if where is not None:
        raise NotImplementedError
    if dtype is not None:
        raise NotImplementedError 
    if ddof == 0:
        unbiased = False
    elif ddof == 1:
        unbiased = True
    else:
        raise ValueError(f"std: ddof = {ddof}")
    if not torch.is_floating_point:
        a = a * 1.0
    return torch.std(a, axis, keepdim=bool(keepdims), out=out)


@asarray_replacer()
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue):
    if where is not None:
        raise NotImplementedError
    if dtype is not None:
        raise NotImplementedError 
    if ddof == 0:
        unbiased = False
    elif ddof == 1:
        unbiased = True
    else:
        raise ValueError(f"std: ddof = {ddof}")
    if not torch.is_floating_point:
        a = a * 1.0
    return torch.var(a, axis, keepdim=bool(keepdims), out=out)



@asarray_replacer()
def argsort(a, axis=-1, kind=None, order=None):
    if order is not None:
        raise NotImplementedError
    stable = True if kind == 'stable' else False
    if axis is None:
        axis = -1
    return torch.argsort(a, stable=stable, dim=axis, descending=False)


##### math functions

from ._unary_ufuncs import *
abs = absolute

from ._binary_ufuncs import *


@asarray_replacer()
def angle(z, deg=False):
    result = torch.angle(z)
    if deg:
        result *= 180 / torch.pi
    return result


@asarray_replacer()
def real(a):
    return torch.real(a)


@asarray_replacer()
def imag(a):
    # torch.imag raises on real-valued inputs
    if torch.is_complex(a):
        return torch.imag(a) 
    else:
        return torch.zeros_like(a)


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


###### mapping from numpy API objects to wrappers from this module ######

# All is in the mapping dict in _mapping.py


