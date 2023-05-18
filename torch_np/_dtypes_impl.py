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


def _category(dtype):
    return {
        torch.bool: 0,
        # int
        torch.uint8: 1,
        torch.int8: 1,
        torch.int16: 1,
        torch.int32: 1,
        torch.int64: 1,
        # float
        torch.float16: 2,
        torch.float32: 2,
        torch.float64: 2,
        # complex
        torch.complex64: 3,
        torch.complex128: 3,
    }[dtype]


def nep50_to_tensors(x1, x2, handle_weaks):
    """If either of inputs is a python scalar, type-promote with NEP 50."""

    def to_tensor(scalar, dtype=None):
        if dtype is None:
            dtype = _dtype_for_scalar(type(scalar))
            dtype = get_default_dtype_for(dtype)
        return torch.as_tensor(scalar, dtype=dtype)

    x1_is_weak = not isinstance(x1, torch.Tensor)
    x2_is_weak = not isinstance(x2, torch.Tensor)
    if not handle_weaks or (x1_is_weak and x2_is_weak):
        x1 = to_tensor(x1) if x1_is_weak else x1
        x2 = to_tensor(x2) if x2_is_weak else x2
        return x1, x2

    # scalar <op> tensor: NEP 50
    assert x1_is_weak != x2_is_weak

    weak, not_weak = (x1, x2) if x1_is_weak else (x2, x1)

    # find the dtype for the weak's type
    weak_dtype = _dtype_for_scalar(type(weak))

    cat_weak = _category(weak_dtype)
    cat_not_weak = _category(not_weak.dtype)

    dt = not_weak.dtype if cat_weak <= cat_not_weak else None

    # special-case complex + float32
    if weak_dtype.is_complex and not_weak.dtype == torch.float32:
        dt = torch.complex64

    # detect overflows: in PyTorch, uint8(-1) wraps around to 255,
    # while NEP50 mandates an exception.
    #
    # Note that we only check if each element of the binop overflows,
    # not the result. Consider, e.g. `uint8(100) + 200`. Operands are OK
    # in uint8, but the result overflows and wrap around 255.
    # Numpy emits a RuntimeWarning, PyTorch does not, and we do not either.
    if cat_weak == 1 and cat_not_weak == 1:
        # integers
        iinfo = torch.iinfo(not_weak.dtype)
        if not (iinfo.min <= weak <= iinfo.max):
            raise OverflowError(
                f"Python integer {weak} out of bounds for {not_weak.dtype}"
            )

    # finally, can make `weak` into a 0D tensor
    weak = to_tensor(weak, dt)

    return (weak, not_weak) if x1_is_weak else (not_weak, weak)
