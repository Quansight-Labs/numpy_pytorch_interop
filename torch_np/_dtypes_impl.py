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

SCALAR_TYPES = (int, bool, float, complex)


def _dtype_for_scalar(py_type):
    return {
        bool: torch.bool,
        int: torch.int64,
        float: torch.float64,
        complex: torch.complex128,
    }[py_type]


categories = [
    (torch.bool,),
    (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
    (torch.float16, torch.float32, torch.float64),
    (torch.complex64, torch.complex128),
]


def category(dtyp):
    for j, cat in enumerate(categories):
        if dtyp in cat:
            return j
    raise ValueError(f"unknown dtype {dtyp}")


dtype_for_cat = {0: torch.bool, 1: torch.int64, 2: torch.float64, 3: torch.complex128}


def nep50_to_tensors(x1, x2):
    """If either of inputs is a python scalar, type-promote with NEP 50.

    NB: NEP 50 mandates RuntimeWarnings on some overflows. We do not emit them:
    we either raise OverflowError or just do the computation.
    """

    x1_type, x2_type = type(x1), type(x2)
    x1_is_weak = x1_type in SCALAR_TYPES
    x2_is_weak = x2_type in SCALAR_TYPES
    if x1_is_weak and x2_is_weak:
        # two scalars: promote
        x1 = torch.as_tensor(x1, dtype=_dtype_for_scalar(x1_type))
        x2 = torch.as_tensor(x2, dtype=_dtype_for_scalar(x2_type))
        return x1, x2
    elif not (x1_is_weak or x2_is_weak):
        # two tensors: nothing to do here
        return x1, x2
    else:
        # scalar <op> scalar: NEP 50
        weak, not_weak = (x1, x2) if x1_is_weak else (x2, x1)

        # find the dtype for the weak's type
        weak_dtype = _dtype_for_scalar(type(weak))

        cat_weak = category(weak_dtype)
        cat_not_weak = category(not_weak.dtype)

        dt = not_weak.dtype if cat_weak <= cat_not_weak else dtype_for_cat[cat_weak]

        # special-case complex + float32
        if weak_dtype.is_complex and not_weak.dtype == torch.float32:
            dt = torch.complex64

        # finally, can cast make `weak` into a 0D tensor
        weak_ = torch.as_tensor(weak, dtype=dt)

        # detect uint overflow: in PyTorch, uint8(-1) wraps around to 255,
        # while NEP50 mandates an exception.
        if weak_.dtype == torch.uint8 and weak_.item() != weak:
            raise OverflowError(f"Python integer {weak} out of bounds for {weak_.dtype}")

        return (weak_, not_weak) if x1_is_weak else (not_weak, weak_)
