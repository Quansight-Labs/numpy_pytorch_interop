"""Export torch work functions for unary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `torch_np/_unary_ufuncs.py` module.
"""

import torch

from torch import (arccos, arccosh, arcsin, arcsinh, arctan, arctanh, ceil,
    cos, cosh, deg2rad, exp, exp2, expm1,
    floor, isfinite, isinf, isnan, log, log10, log1p, log2, logical_not,
    negative, rad2deg, reciprocal, sign, signbit,
    sin, sinh, sqrt, square, tan, tanh, trunc)

# renames
from torch import (conj_physical as conjugate, round as rint, bitwise_not as invert, rad2deg as degrees,
    deg2rad as radians, absolute as fabs, )

# special cases: torch does not export these names
def cbrt(x):
    return torch.pow(x, 1 / 3)


def positive(x):
    return +x


def absolute(x):
    # work around torch.absolute not impl for bools
    if x.dtype == torch.bool:
        return x
    return torch.absolute(x)


abs = absolute
conj = conjugate

