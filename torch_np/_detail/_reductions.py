""" Implementation of reduction operations, to be wrapped into arrays, dtypes etc
in the 'public' layer.

Anything here only deals with torch objects, e.g. "dtype" is a torch.dtype instance etc
"""

import torch
from . import _util
from . import _scalar_types

NoValue = None


def _atleast_float(dtype, other_dtype):
    """Return a dtype that is real or complex floating-point.

    For inputs that are boolean or integer dtypes, this returns the default
    float dtype; inputs that are complex get converted to the default complex
    dtype; real floating-point dtypes (`float*`) get passed through unchanged
    """
    if dtype is None:
        dtype = other_dtype
    if not (dtype.is_floating_point or dtype.is_complex):
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


def argmax(tensor, axis=None):
    axis = _util.allow_only_single_axis(axis)
    tensor = torch.argmax(tensor, axis)
    return tensor

def argmin(tensor, axis=None):
    axis = _util.allow_only_single_axis(axis)
    tensor = torch.argmin(tensor, axis)
    return tensor


def any(tensor, axis=None, *, where=NoValue):
    if where is not NoValue:
        raise NotImplementedError

    axis = _util.allow_only_single_axis(axis)

    if axis is None:
        result = tensor.any()
    else:
        result = tensor.any(axis)
    return result


def all(tensor, axis=None, *, where=NoValue):
    if where is not NoValue:
        raise NotImplementedError

    axis = _util.allow_only_single_axis(axis)

    if axis is None:
        result = tensor.all()
    else:
        result = tensor.all(axis)
    return result


def max(tensor, axis=None, initial=NoValue, where=NoValue):
    if initial is not NoValue or where is not NoValue:
        raise NotImplementedError

    result = tensor.amax(axis)
    return result


def min(tensor, axis=None, initial=NoValue, where=NoValue):
    if initial is not NoValue or where is not NoValue:
        raise NotImplementedError

    result = tensor.amin(axis)
    return result


def sum(tensor, axis=None, dtype=None, initial=NoValue, where=NoValue):
    if initial is not NoValue or where is not NoValue:
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
    if initial is not NoValue or where is not NoValue:
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
    if where is not NoValue:
        raise NotImplementedError

    dtype = _atleast_float(dtype, tensor.dtype)

    if axis is None:
        result = tensor.mean(dtype=dtype)
    else:
        result = tensor.mean(dtype=dtype, dim=axis)

    return result


def std(tensor, axis=None, dtype=None, ddof=0, *, where=NoValue):
    if where is not NoValue:
        raise NotImplementedError

    dtype = _atleast_float(dtype, tensor.dtype)

    if dtype is not None:
        tensor = tensor.to(dtype)
    result = tensor.std(dim=axis, correction=ddof)

    return result


def var(tensor, axis=None, dtype=None, ddof=0, *, where=NoValue):
    if where is not NoValue:
        raise NotImplementedError

    dtype = _atleast_float(dtype, tensor.dtype)

    if dtype is not None:
        tensor = tensor.to(dtype)
    result = tensor.var(dim=axis, correction=ddof)

    return result

