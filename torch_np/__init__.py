from . import linalg, random
from ._dtypes import *
from ._funcs import *
from ._getlimits import errstate, finfo, iinfo
from ._ndarray import array, asarray, can_cast, ndarray, newaxis, result_type
from ._ufuncs import *
from ._util import AxisError, UFuncTypeError

# from . import testing

alltrue = all
sometrue = any

inf = float("inf")
nan = float("nan")
from math import pi, e  # isort: skip

False_ = asarray(False, bool_)
True_ = asarray(True, bool_)
