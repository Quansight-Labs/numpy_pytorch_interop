import functools

import torch

from . import _dtypes
from . import _helpers


def dtype_to_torch(func):
##    @functools.wraps  # XXX: why
    def wrapped(*args, dtype=None, **kwds):
        torch_dtype = None
        if dtype is not None:
            dtype = _dtypes.dtype(dtype)
            torch_dtype = dtype._scalar_type.torch_dtype
        return func(*args, dtype=torch_dtype, **kwds)
    return wrapped


def emulate_out_arg(func):
    """Simulate the out=... handling *for functions which do not need it*.

    With this decorator, the inner function just does not see the out array.
    """
    def wrapped(*args, out=None, **kwds):
        from ._ndarray import ndarray
        if out is not None:
            if not isinstance(out, ndarray):
                raise TypeError("Return arrays must be of ArrayType")        
        result_tensor = func(*args, **kwds)
        return _helpers.result_or_out(result_tensor, out)

    return wrapped


def out_shape_dtype(func):
    """Handle out=... kwarg for ufuncs.

    With ufuncs, `out` array can typcast and broadcast ufunc arguments, hence
    extract the shape and dtype of the tensor which backs the `out` array
    and pass these through.
    """
    def wrapped(*args, out=None, **kwds):
        from ._ndarray import ndarray
        if out is not None:
            if not isinstance(out, ndarray):
                raise TypeError("Return arrays must be of ArrayType")
            kwds.update({'out_shape_dtype': (out.get().dtype, out.get().shape)})
        result_tensor = func(*args, **kwds)
        return _helpers.result_or_out(result_tensor, out)

    return wrapped


def deco_unary_ufunc_from_impl(impl_func):
    @functools.wraps(impl_func)
    @dtype_to_torch
    @out_shape_dtype
    def wrapped(x1, *args, **kwds):
        from ._ndarray import asarray
        x1_tensor = asarray(x1).get()
        result = impl_func((x1_tensor,), *args, **kwds)
        return result
    return wrapped


# TODO: deduplicate with _ndarray/asarray_replacer,
# and _wrapper/concatenate et al
def deco_ufunc_from_impl(impl_func):
    @functools.wraps(impl_func)
    @dtype_to_torch
    @out_shape_dtype
    def wrapped(x1, x2, *args, **kwds):
        from ._ndarray import asarray
        x1_tensor = asarray(x1).get()
        x2_tensor = asarray(x2).get()
        return impl_func((x1_tensor, x2_tensor), *args, **kwds)
    return wrapped
