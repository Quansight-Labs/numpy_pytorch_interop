import functools

import torch

import _util

NoValue = None

##################### ndarray class ###########################

class ndarray:
    def __init__(self):
        self._tensor = torch.Tensor()

    def get(self):
        return self._tensor

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

    # ctors
    def astype(self, dtype):
        newt = ndarray()
        newt._tensor = self._tensor.to(dtype)
        return newt

    # niceties
    def __str__(self):
        return str(self._tensor).replace("tensor", "array_w")

    __repr__ = __str__


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




def asarray(a, dtype=None, order=None, *, like=None):
    _util.subok_not_ok(like)
    if order is not None:
        raise NotImplementedError

    if isinstance(a, ndarray):
        return a

    # This and array(...) are the only places which talk to ndarray directly.
    # The rest goes through asarray (preferred) or array.
    out = ndarray()
    out._tensor = torch.asarray(a, dtype=dtype)
    return out



def asarray_replacer_1(func):
    """Asarray the *first* arg, so that the wrapped `func` operates on a
       torch.Tensor. Then asarray the result.
    """
    @functools.wraps(func)
    def wrapped(x, *args, **kwds):
        x_tensor = asarray(x).get()
        return asarray(func(x_tensor, *args, **kwds))
    return wrapped

