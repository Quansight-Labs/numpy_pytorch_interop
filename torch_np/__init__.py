from ._binary_ufuncs import *
from ._dtypes import *
from ._getlimits import errstate, finfo, iinfo
from ._ndarray import can_cast, newaxis, result_type
from ._scalar_types import *
from ._unary_ufuncs import *
from ._util import AxisError, UFuncTypeError
from ._wrapper import *

# from . import testing


inf = float("inf")
nan = float("nan")


#### HACK HACK HACK ####
import torch

torch.set_default_dtype(torch.float64)
del torch
