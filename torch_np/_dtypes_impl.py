"""Dtypes/scalar type implementtaions with torch dtypes.

Here `dtype` is always a torch.dtype, this module knows nothing about
scalar types, wrapper dtypes or anything like that. PyTorch only.
"""
from collections import namedtuple

import torch

#### defaults : mimic NumPy, allow user control
DefaultDTypes = namedtuple(
    "DefaultDTypes", ["float_dtype", "complex_dtype", "int_dtype"]
)

# a global state: NumPy defaults
default_dtypes_numpy = DefaultDTypes(
    float_dtype=torch.float64, complex_dtype=torch.complex128, int_dtype=torch.int64
)

default_dtypes = default_dtypes_numpy


def get_default_dtype_for(dtype):
    """Default scalar type given sctype category."""
    if dtype == torch.bool:
        return dtype
    if dtype.is_complex:
        return default_dtypes.complex_dtype
    if dtype.is_floating_point:
        return default_dtypes.float_dtype
    # else, it must be (some) integer
    return default_dtypes.int_dtype


from . import _casting_dicts as _cd


def can_cast_impl(from_torch_dtype, to_torch_dtype, casting):
    return _cd._can_cast_dict[casting][from_torch_dtype][to_torch_dtype]


def result_type_impl(*tensors):
    # NB: torch dtypes here
    dtyp = tensors[0].dtype
    if len(tensors) == 1:
        return dtyp

    for curr in tensors[1:]:
        dtyp = _cd._result_type_dict[dtyp][curr.dtype]

    return dtyp


# ### NEP 50 helpers ###

categories = [(torch.bool,),
              (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
              (torch.float16, torch.float32, torch.float64),
              (torch.complex64, torch.complex128)]


def category(dtyp):
    for j, cat in enumerate(categories):
        if dtyp in cat:
            return j
    raise ValueError(f"unknown dtype {dtyp}")


dtype_for_cat = {0: torch.bool,
                 1: torch.int64,
                 2: torch.float64,
                 3: torch.complex128}
