# this file is autogenerated via gen_ufuncs.py
# do not edit manually!

import torch

from . import _util
from ._ndarray import asarray_replacer

from ._ndarray import asarray, ndarray, can_cast
from . import _dtypes
from . import _helpers


def sin(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or not where:
        raise NotImplementedError

    # XXX: dtype=... parameter is silently ignored

    x_array = asarray(x)

    arrays = (x_array,)
    x_tensor, = _helpers.check_bcast(arrays, out, casting)

    result = torch.sin(x_tensor)

    if out is not None:
        out_tensor = out.get()
        out_tensor.copy_(result)
        return out
    else:
        return asarray(result)

'''
    # XXX: or this, which one is better for TorchInductor?
    # result = {torch_stanza}
    if out is not None:
        torch.sin(x_tensor, out=out_tensor)
        return out
    else:
        result = torch.sin(x_tensor)
        return asarray(result)
'''


#################################


@asarray_replacer()
def absolute(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.absolute(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def absolute(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.absolute(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def arccos(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.arccos(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def arccosh(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.arccosh(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def arcsin(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.arcsin(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def arcsinh(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.arcsinh(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def arctan(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.arctan(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def arctanh(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.arctanh(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def cbrt(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.pow(x, 1/3, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def ceil(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.ceil(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def conjugate(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.conj_physical(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def conjugate(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.conj_physical(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def cos(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.cos(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def cosh(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.cosh(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def deg2rad(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.deg2rad(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def degrees(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.rad2deg(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def exp(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.exp(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def exp2(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.exp2(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def expm1(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.expm1(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def fabs(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.absolute(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def floor(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.floor(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def isfinite(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.isfinite(x)
    if dtype is not None:
        result = result.to(dtype)
    
    if out is not None:
        out[...] = result

    return result



@asarray_replacer()
def isinf(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.isinf(x)
    if dtype is not None:
        result = result.to(dtype)
    
    if out is not None:
        out[...] = result

    return result



@asarray_replacer()
def isnan(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.isnan(x)
    if dtype is not None:
        result = result.to(dtype)
    
    if out is not None:
        out[...] = result

    return result



@asarray_replacer()
def log(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.log(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def log10(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.log10(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def log1p(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.log1p(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def log2(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.log2(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def logical_not(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.logical_not(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def negative(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.negative(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def positive(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = +x
    if dtype is not None:
        result = result.to(dtype)
    
    if out is not None:
        out[...] = result

    return result



@asarray_replacer()
def rad2deg(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.rad2deg(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def radians(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.deg2rad(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def reciprocal(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.reciprocal(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def rint(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.round(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def sign(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.sign(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def signbit(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.signbit(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result







@asarray_replacer()
def sinh(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.sinh(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def sqrt(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.sqrt(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def square(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.square(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def tan(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.tan(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def tanh(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.tanh(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



@asarray_replacer()
def trunc(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = torch.trunc(x, out=out)
    if dtype is not None:
        result = result.to(dtype)
    
    return result



__all__ = ['absolute', 'absolute', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctanh', 'cbrt', 'ceil', 'conjugate', 'conjugate', 'cos', 'cosh', 'deg2rad', 'degrees', 'exp', 'exp2', 'expm1', 'fabs', 'floor', 'isfinite', 'isinf', 'isnan', 'log', 'log10', 'log1p', 'log2', 'logical_not', 'negative', 'positive', 'rad2deg', 'radians', 'reciprocal', 'rint', 'sign', 'signbit', 'sin', 'sinh', 'sqrt', 'square', 'tan', 'tanh', 'trunc']
