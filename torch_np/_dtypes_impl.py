"""Dtypes/scalar type implementtaions with torch dtypes.

Here `dtype` is always a torch.dtype, this module knows nothing about
scalar types, wrapper dtypes or anything like that. PyTorch only.
"""
import torch

#### defaults : mimic NumPy, allow user control


class DefaultDTypes:
    """A bag class for dtype defaults."""

    def __init__(self, fp_dtype="numpy", int_dtype="pytorch"):

        if not (fp_dtype in ["numpy", "pytorch"] or isinstance(fp_dtype, torch.dtype)):
            raise TypeError(f"failed to interpter {fp_dtype} as torch.dtype.")
        if not (
            int_dtype in ["numpy", "pytorch"] or isinstance(int_dtype, torch.dtype)
        ):
            raise TypeError(f"failed to interpter {int_dtype} as torch.dtype.")

        if fp_dtype == "numpy":
            self.float_dtype = torch.float64
        elif fp_dtype == "pytorch":
            self.float_dtype = torch.float32
        else:
            self.float_dtype = fp_dtype

        self.complex_dtype = {
            torch.float64: torch.complex128,
            torch.float32: torch.complex64,
            torch.float16: torch.complex64,
        }[self.float_dtype]

        if int_dtype in ["numpy", "pytorch"]:
            self.int_dtype = torch.int64
        else:
            self.int_dtype = int_dtype


# a global state: NumPy defaults
default_dtypes = DefaultDTypes()


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


def result_type_impl(dtypes):
    # NB: torch dtypes here
    dtyp = dtypes[0]
    if len(dtypes) == 1:
        return dtyp

    for curr in dtypes[1:]:
        dtyp = _cd._result_type_dict[dtyp][curr]

    return dtyp
