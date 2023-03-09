"""Export torch work functions for binary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `torch_np/_binary_ufuncs.py` module.
"""

import torch

from . import _dtypes_impl, _util


from torch import (add, arctan2, bitwise_and, bitwise_or, bitwise_xor, copysign, divide, 
    float_power, floor_divide, fmax, fmin, fmod, gcd, greater, greater_equal, heaviside,
    hypot, lcm, ldexp, less, less_equal, logaddexp, logaddexp2, logical_and,
    logical_or, logical_xor, maximum, minimum, remainder, multiply, nextafter, not_equal,
    remainder, subtract, divide)


# renames
from torch import (eq as equal, pow as power, bitwise_right_shift as right_shift,
        bitwise_left_shift as left_shift,)


# work around torch limitations w.r.t. numpy
def matmul(x, y):
    # work around RuntimeError: expected scalar type Int but found Double
    dtype = _dtypes_impl.result_type_impl((x.dtype, y.dtype))
    x = _util.cast_if_needed(x, dtype)
    y = _util.cast_if_needed(y, dtype)
    result = torch.matmul(x, y)
    return result

