# this file is autogenerated via gen_ufuncs.py
# do not edit manually!

import torch

from . import _util
from ._ndarray import asarray_replacer


from ._ndarray import asarray, ndarray
from  . import _dtypes
from ._wrapper import can_cast


def add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or not where:
        raise NotImplementedError

    # XXX: dtype=... parameter is silently ignored

    x1_array = asarray(x1)
    x2_array = asarray(x2)

    if out is not None:
        if not isinstance(out, ndarray):
            raise TypeError("Return arrays must be of ArrayType")

        for arr in (x1_array, x2_array):
            # check dtypes of x and out
            if not can_cast(arr.dtype, out.dtype, casting=casting):
                raise TypeError(f"Cannot cast array data from {x.dtype} to"
                                 " {out_dtype} according to the rule '{casting}'")
            tensor = arr.get()

            # `out` broadcasts `x`
            if arr.shape != out.shape:
                tensor = torch.broadcast_to(tensor, out.shape)

            # cast x if needed
            if arr.dtype != out.dtype:
                tensor = tensor.to(_dtypes.torch_dtype_from(out.dtype))

    result = torch.add(x1_array.get(), x2_array.get())

    if out is not None:
        out_tensor = out.get()
        out_tensor.copy_(result)
        return out
    else:
        return asarray(result)






#####################################

'''
@asarray_replacer("two")
def add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.add(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result
'''


@asarray_replacer("two")
def arctan2(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.arctan2(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def bitwise_and(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.bitwise_and(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def bitwise_or(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.bitwise_or(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def bitwise_xor(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.bitwise_xor(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def copysign(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.copysign(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def divide(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.divide(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def equal(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.eq(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def float_power(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.float_power(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def floor_divide(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.floor_divide(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def fmax(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.fmax(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def fmin(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.fmin(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def fmod(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.fmod(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def gcd(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.gcd(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def greater(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.greater(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def greater_equal(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.greater_equal(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def heaviside(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.heaviside(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def hypot(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.hypot(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def lcm(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.lcm(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def ldexp(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.ldexp(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def left_shift(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.left_shift(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def less(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.less(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def less_equal(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.less_equal(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def logaddexp(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.logaddexp(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def logaddexp2(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.logaddexp2(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def logical_and(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.logical_and(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def logical_or(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.logical_or(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def logical_xor(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.logical_xor(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def matmul(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.matmul(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def maximum(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.maximum(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def minimum(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.minimum(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def remainder(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.remainder(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def multiply(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.multiply(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def nextafter(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.nextafter(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def not_equal(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.not_equal(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def power(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.pow(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def remainder(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.remainder(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def right_shift(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.right_shift(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def subtract(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.subtract(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result



@asarray_replacer("two")
def divide(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.divide(x1, x2, out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result

