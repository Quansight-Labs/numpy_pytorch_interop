import functools

import torch

from . import _dtypes, _helpers
from ._detail import _util


def out_shape_dtype(func):
    """Handle out=... kwarg for ufuncs.

    With ufuncs, `out` array can typcast and broadcast ufunc arguments, hence
    extract the shape and dtype of the tensor which backs the `out` array
    and pass these through.
    """

    @functools.wraps(func)
    def wrapped(*args, out=None, **kwds):
        if out is not None:
            kwds.update({"out_shape_dtype": (out.get().dtype, out.get().shape)})
        result_tensor = func(*args, **kwds)
        return _helpers.result_or_out(result_tensor, out)

    return wrapped
