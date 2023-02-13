"""Dtypes/scalar type implementtaions with torch dtypes.

Here `dtype` is always a torch.dtype, this module knows nothing about
scalar types, wrapper dtypes or anything like that. PyTorch only.
"""
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


from . import _casting_dicts as _cd


def can_cast_impl(from_torch_dtype, to_torch_dtype, casting):
    return _cd._can_cast_dict[casting][from_torch_dtype][to_torch_dtype]


def result_type_impl(dtypes):
    # NB: torch dtypes here
    dtyp = dtypes[0]
    if len(dtypes) == 1:
        return dtyp

    for curr in dtypes[1:]:
        dtyp = _cd._result_type_dict[dtyp][curr]

    return dtyp
