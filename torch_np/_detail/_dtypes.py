"""Dtypes/scalar type implementtaions with torch dtypes.

Here `dtype` is always a torch.dtype, this module knows nothing about
scalar types, wrapper dtypes or anything like that. PyTorch only.
"""
import builtins

import torch


#### defaults : mimic NumPy
default_scalar_dtype = torch.float64
default_int_dtype = torch.int64
default_float_dtype = torch.float64
default_complex_dtype = torch.complex128
##########################


def get_default_dtype_for(dtype):
    """Default scalar type given sctype category."""
    if dtype == torch.bool:
        return dtype
    if dtype.is_complex:
        return default_complex_dtype
    if dtype.is_floating_point:
        return default_float_dtype
    # else, it must be (some) integer
    return default_int_dtype


# XXX: is it ever used? cf _detail/reductions.py::_atleast_float(...)
def float_or_default(sctype, enforce_float=False):
    """bool -> int; int -> float"""
    if issubclass(sctype, bool_):
        sctype = default_int_type
    if enforce_float and issubclass(sctype, integer):
        sctype = default_float_type
    return sctype


from . import _casting_dicts as _cd


def _can_cast_sctypes(from_sctype, to_sctype, casting):
    return _can_cast_impl(from_sctype.torch_dtype, to_sctype.torch_dtype, casting)


def _can_cast_impl(from_torch_dtype, to_torch_dtype, casting):
    return _cd._can_cast_dict[casting][from_torch_dtype][to_torch_dtype]


'''
__all__ = list(_names.keys())
__all__ += [
    "intp",
    "int_",
    "intc",
    "byte",
    "short",
    "longlong",
    "ubyte",
    "half",
    "single",
    "double",
    "csingle",
    "cdouble",
    "float_",
]
__all__ += ["sctypes"]
__all__ += [
    "generic",
    "number",
    "integer",
    "signedinteger",
    "unsignedinteger",
    "inexact",
    "floating",
    "complexfloating",
]
'''
