import functools

import torch

from . import _dtypes

def dtype_to_torch(func):
##    @functools.wraps  # XXX: why
    def wrapped(*args, dtype=None, **kwds):

        torch_dtype = None
        if dtype is not None:
            dtype = _dtypes.dtype(dtype)
            torch_dtype = dtype._scalar_type.torch_dtype

        return func(*args, dtype=torch_dtype, **kwds)
    return wrapped
