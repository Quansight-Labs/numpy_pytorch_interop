""" Implementation of reduction operations, to be wrapped into arrays, dtypes etc
in the 'public' layer.

Anything here only deals with torch objects, e.g. "dtype" is a torch.dtype instance etc
"""

import torch
from . import _util
from . import _scalar_types

NoValue = None


def _atleast_float(dtype, other_dtype):
    """Cast bools and ints and floats to a default float; complex to default complex.
    """
    if dtype is None:
        dtype = other_dtype
    if not dtype.is_floating_point:
        if dtype.is_complex:
            sctype = _scalar_types.default_complex_type
        else:
            sctype = _scalar_types.default_float_type
        dtype = sctype.torch_dtype
    return dtype


def count_nonzero(a, axis=None):
    # XXX: this all should probably be generalized to a sum(a != 0, dtype=bool)
    try:
        return a.count_nonzero(axis)
    except RuntimeError:
        raise ValueError
    return tensor


def any(tensor, axis=None, *, where=NoValue):
    if where is not None:
        raise NotImplementedError

    axis = _util.allow_only_single_axis(axis)

    if axis is None:
        result = tensor.any()
    else:
        result = tensor.any(axis)
    return result


def all(tensor, axis=None, *, where=NoValue):
    if where is not None:
        raise NotImplementedError

    axis = _util.allow_only_single_axis(axis)

    if axis is None:
        result = tensor.all()
    else:
        result = tensor.all(axis)
    return result


def max(tensor, axis=None, initial=NoValue,
         where=NoValue):
    if where is not None:
        raise NotImplementedError
    if initial is not None:
        raise NotImplementedError

    result = tensor.amax(axis)
    return result


def min(tensor, axis=None, initial=NoValue, where=NoValue):
    if where is not None:
        raise NotImplementedError
    if initial is not None:
        raise NotImplementedError

    result = tensor.amin(axis)
    return result


def sum(tensor, axis=None, dtype=None, initial=NoValue, where=NoValue):
    if initial is not None or where is not None:
        raise NotImplementedError

    assert dtype is None or isinstance(dtype, torch.dtype)

    if dtype == torch.bool:
        dtype = _scalar_types.default_int_type.dtype

    if axis is None:
        result = tensor.sum(dtype=dtype)
    else:
        result = tensor.sum(dtype=dtype, dim=axis)

    return result


def prod(tensor, axis=None, dtype=None, initial=NoValue, where=NoValue):
    if initial is not None or where is not None:
        raise NotImplementedError

    axis = _util.allow_only_single_axis(axis)

    if dtype == torch.bool:
        dtype = _scalar_types.default_int_type.dtype

    if axis is None:
        result = tensor.prod(dtype=dtype)
    else:
        result = tensor.prod(dtype=dtype, dim=axis)

    return result


def mean(tensor, axis=None, dtype=None, *, where=NoValue):
    if where is not None:
        raise NotImplementedError

    dtype = _atleast_float(dtype, tensor.dtype)

    if axis is None:
        result = tensor.mean(dtype=dtype)
    else:
        result = tensor.mean(dtype=dtype, dim=axis)

    return result


def std(tensor, axis=None, dtype=None, ddof=0, *, where=NoValue):
    if where is not None:
        raise NotImplementedError

    dtype = _atleast_float(dtype, tensor.dtype)

    tensor = tensor.to(dtype)
    result = tensor.std(dim=axis, correction=ddof)

    return result


def var(tensor, axis=None, dtype=None, ddof=0, *, where=NoValue):
    if where is not None:
        raise NotImplementedError

    dtype = _atleast_float(dtype, tensor.dtype)

    tensor = tensor.to(dtype)
    result = tensor.var(dim=axis, correction=ddof)

    return result

