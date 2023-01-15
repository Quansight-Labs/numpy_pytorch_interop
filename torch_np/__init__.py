from ._dtypes import *
from ._detail._scalar_types import *
from ._wrapper import *
#from . import testing

from ._unary_ufuncs import *
from ._binary_ufuncs import *
from ._ndarray import can_cast, result_type, newaxis
from ._detail._util import AxisError, UFuncTypeError
from ._getlimits import iinfo, finfo
from ._getlimits import errstate

inf = float('inf')
nan = float('nan')

