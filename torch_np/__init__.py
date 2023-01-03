from ._dtypes import *
from ._scalar_types import *
from ._wrapper import *
from . import testing

from ._unary_ufuncs import *
from ._binary_ufuncs import *
from ._ndarray import can_cast, result_type, newaxis
from ._util import AxisError


inf = float('inf')
nan = float('nan')
