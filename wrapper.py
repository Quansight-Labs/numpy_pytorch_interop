"""A thin pytorch / numpy compat layer.

Things imported from here have numpy-compatible signatures but operate on
pytorch tensors.
"""
import numpy as np
import torch

import _util

# Things to decide on (punt for now)
#
# 1. Q: What are the return types of wrapper functions: plain torch.Tensors or
#       wrapper ndarrays.
#    A: Tensors, apparently
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
#    A: raise ValueErrors?
#       initial=... can be useful though, punt on
#       where=...   punt on for now


# TODO
# 1. Mapping of the numpy layout ('C', 'K' etc) to torch layout / memory_format.
# 2. np.dtype <-> torch.dtype
# 3. numpy type casting rules (to be cleaned up in numpy: follow old or new)


NoValue = None

###### array creation routines

def asarray(a, dtype=None, order=None, *, like=None):
    _util.subok_no_ok(like)
    if order is not None:
        raise NotImplementedError
    return torch.asarray(a, dtype=dtype)


def array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0,
          like=None):
    _util.subok_not_ok(like, subok)
    if order != 'K':
        raise NotImplementedError
    result = torch.asarray(object, dtype=dtype, copy=copy)
    ndim_extra = ndmin - result.ndim
    if ndim_extra > 0:
        result = result.reshape((1,)*ndim_extra + result.shape)
    return result


def copy(a, order='K', subok=False):
    _util.subok_not_ok(subok=subok)
    if order != 'K':
        raise NotImplementedError
    return torch.clone(a)


from torch import atleast_1d, atleast_2d, atleast_3d


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
             axis=0):
    if axis !=0 or retstep or not endpoint:
        raise NotImplementedError
    # XXX: raises TypeError if start or stop are not scalars
    return torch.linspace(start, stop, num, dtype=dtype)


def empty(shape, dtype=float, order='C', *, like=None):
    _util.subok_not_ok(like)
    if order != 'C':
        raise NotImplementedError
    return torch.empty(shape, dtype=dtype)


# NB: *_like function deliberately deviate from numpy: it has subok=True
# as the default; we set subok=False and raise on anything else.
def empty_like(prototype, dtype=None, order='K', subok=False, shape=None):
    _util.subok_not_ok(subok=subok)
    if order != 'K':
        raise NotImplementedError
    result = torch.empty(prototype, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def full(shape, fill_value, dtype=None, order='C', *, like=None):
    _util.subok_not_ok(like)
    if order != 'C':
        raise NotImplementedError
    return torch.full(shape, fill_value, dtype=dtype)


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
    return torch.ones(shape, dtype=dtype)


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
    return torch.zeros(shape, dtype=dtype)


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
    s = M - k if k >= 0 else N + k
    z = torch.zeros(N, M, dtype=dtype)
    z.diagonal(k).fill_(1)
    return z


def identity(n, dtype=None, *, like=None):
    _util.subok_not_ok(like)
    return torch.eye(n, dtype=dtype)


###### misc/unordered

def prod(a, axis=None, dtype=None, out=None, keepdims=NoValue,
         initial=NoValue, where=NoValue):
    if initial is not None or where is not None:
        raise NotImplementedError
    if axis is None:
        if keepdims is not None:
            raise NotImplementedError
        return torch.prod(torch.as_tensor(a), dtype=dtype)
    elif _util.is_sequence(axis):
        raise NotImplementedError
    return torch.prod(torch.as_tensor(a), dim=axis, dtype=dtype, keepdim=bool(keepdims), out=out)


def corrcoef(x, y=None, rowvar=True, bias=NoValue, ddof=NoValue, *, dtype=None):
    if bias is not None or ddof is not None:
        # deprecated in NumPy
        raise NotImplementedError
    if rowvar is False:
        x = x.T
    if y is not None:
        raise NotImplementedError
    if dtype is not None:
        x = x.type(dtype)
    return torch.corrcoef(x)


def concatenate(ar_tuple, axis=0, out=None, dtype=None, casting="same_kind"):
    if casting != "same_kind":
        raise NotImplementedError   # XXX
    if dtype is not None:
        # XXX: map numpy dtypes
        ar_tuple = tuple(ar.type(dtype) for ar in ar_typle)
    return torch.cat(ar_tuple, axis, out=out)


def squeeze(a, axis=None):
    if axis is None:
        return torch.squeeze(a)
    else:
        return torch.squeeze(a, axis)


def bincount(x, /, weights=None, minlength=0):
    return torch.bincount(x, weights, minlength)


###### module-level queries of object properties

def ndim(a):
    return torch.as_tensor(a).ndim


def shape(a):
    return tuple(torch.as_tensor(a).shape)


def size(a, axis=None):
    if axis is None:
        return torch.as_tensor(a).numel()
    else:
        return torch.as_tensor(a).shape[axis]


###### shape manipulations and indexing

def reshape(a, newshape, order='C'):
    if order != 'C':
        raise NotImplementedError
    return torch.reshape(a, newshape)


def broadcast_to(array, shape, subok=False):
    _util.subok_not_ok(subok=subok)
    return torch.broadcast_to(array, shape)


from torch import broadcast_shapes


def broadcast_arrays(*args, subok=False):
    _util.subok_not_ok(subok=subok)
    return torch.broadcast_tensors(*args)


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



###### reductions

def argmax(a, axis=None, out=None, *, keepdims=NoValue):
    if axis is None:
        result = torch.argmax(a, keepdims=bool(keepdims))
    else:
        result = torch.argmax(a, axis, keepdims=bool(keepdims))
    if out is not None:
        out.copy_(result)
    return result


##### math functions

from _unary_ufuncs import *
abs = absolute

from _binary_ufuncs import *


def angle(z, deg=False):
    result = torch.angle(z)
    if deg:
        result *= 180 / torch.pi
    return result

from torch import imag, real


def real_if_close(a, tol=100):
    if not torch.is_complex(a):
        return a
    if torch.abs(torch.imag) < tol * torch.finfo(a.dtype).eps:
        return torch.real(a)
    else:
        return a


def iscomplex(x):
    if torch.is_complex(x):
        return torch.as_tensor(x).imag != 0
    result = torch.zeros_like(x, dtype=torch.bool)
    return result[()]


def isreal(x):
    if torch.is_complex(x):
        return torch.as_tensor(x).imag == 0
    result = torch.zeros_like(x, dtype=torch.bool)
    return result[()]


def iscomplexobj(x):
    return torch.is_complex(x)


def isrealobj(x):
    return not torch.is_complex(x)


def isneginf(x, out=None):
    return torch.isneginf(x, out=out)


def isposinf(x, out=None):
    return torch.isposinf(x, out=out)


def i0(x):
    return torch.special.i0(x)


###### mapping from numpy API objects to wrappers from this module ######

# All is in the mapping dict in _mapping.py

##################### ndarray class ###########################

class ndarray:
    def __init__(self, *args, **kwds):
        self._tensor = torch.Tensor(*args, **kwds)

    @property
    def shape(self):
        return tuple(self._tensor.shape)

    @property
    def size(self):
        return self._tensor.numel()

    @property
    def ndim(self):
        return self._tensor.ndim

    @property
    def dtype(self):
        return self._tensor.dtype

    @property
    def strides(self):
        return self._tensor.stride()   # XXX: byte strides

    ### arithmetic ###

    def __add__(self, other):
        return self._tensor__add__(other)

    def __iadd__(self, other):
        return self._tensor.__add__(other)

    def __sub__(self, other):
        return self._tensor.__sub__(other)

    def __mul__(self, other):
        return self._tensor.__mul__(other)

    ### methods to match namespace functions
    def squeeze(self, axis=None):
        return squeeze(self._tensor, axis)

    def argmax(self, axis=None, out=None, *, keepdims=NoValue):
        return argmax(self._tensor, axis, out=out, keepdims=keepdims)

    def reshape(self, shape, order='C'):
        return reshape(self._tensor, shape, order)



