""" Define the supported dtypes and numpy <--> torch dtype mapping, define casting rules. 
"""

# TODO: 1. define torch_np dtypes, make this work without numpy.
#       2. mimic numpy's various aliases (np.half == np.float16, dtype='i8' etc)
#       3. convert from python types: np.ones(3, dtype=float) etc

import builtins
import torch

from ._detail import _scalar_types


__all__ = ['dtype', 'DType', 'typecodes', 'issubdtype']


# Define analogs of numpy dtypes supported by pytorch.


def dtype(arg):
    if arg is None:
        arg = _scalar_types.default_scalar_type
    return DType(arg)


def torch_dtype_from(dtype_arg):
    return dtype(dtype_arg).torch_dtype


class DType:
    def __init__(self, arg):
        # a pytorch object?
        if isinstance(arg, torch.dtype):
            sctype = _scalar_types._torch_dtypes[arg]
        elif isinstance(arg, torch.Tensor):
            sctype = _scalar_types._torch_dtypes[arg.dtype]
        # a scalar type?
        elif issubclass_(arg, _scalar_types.generic):
            sctype = arg
        # a dtype already?
        elif isinstance(arg, DType):
            sctype = arg._scalar_type
        # a has a right attribute?
        elif hasattr(arg, 'dtype'):
            sctype = arg.dtype._scalar_type
        else:
            sctype = _scalar_types.sctype_from_string(arg)
        self._scalar_type = sctype

    @property
    def name(self):
        return self._scalar_type.name

    @property
    def type(self):
        return self._scalar_type

    @property
    def typecode(self):
        return self._scalar_type.typecode

    def __eq__(self, other):
        if isinstance(other, DType):
            return self._scalar_type == other._scalar_type
        try:
            other_instance = DType(other)
        except TypeError:
            return False
        return self._scalar_type == other_instance._scalar_type

    @property
    def torch_dtype(self):
        return self._scalar_type.torch_dtype

    def __hash__(self):
        return hash(self._scalar_type.name)

    def __repr__(self):
        return f'dtype("{self.name}")'

    __str__ = __repr__

    @property
    def itemsize(self):
        elem = self.type(1)
        return elem.get().element_size()

    def __getstate__(self):
        return self._scalar_type

    def __setstate__(self, value):
        self._scalar_type = value


typecodes = {'All': 'efdFDBbhil?',
        'AllFloat': 'efdFD',
        'AllInteger': 'Bbhil',
        'Integer': 'bhil',
        'UnsignedInteger': 'B',
        'Float': 'efd',
        'Complex': 'FD',
}



# ### Defaults and dtype discovery

def default_int_type():
    return dtype(_scalar_types.default_int_type)


def default_float_type():
    return dtype(_scalar_types.default_float_type)


def default_complex_type():
    return dtype(_scalar_types.default_complex_type)


def is_floating(dtyp):
    dtyp = dtype(dtyp)
    return issubclass(dtyp.type, _scalar_types.floating)


def is_integer(dtyp):
    dtyp = dtype(dtyp)
    return issubclass(dtyp.type, _scalar_types.integer)


def get_default_dtype_for(dtyp):
    sctype = dtype(dtyp).type
    return _scalar_types.get_default_type_for(sctype)


def issubclass_(arg, klass):
    try:
        return issubclass(arg, klass)
    except TypeError:
        return False


def issubdtype(arg1, arg2):
    # cf https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numerictypes.py#L356-L420
    if not issubclass_(arg1, _scalar_types.generic):
        arg1 = dtype(arg1).type
    if not issubclass_(arg2, _scalar_types.generic):
        arg2 = dtype(arg2).type
    return issubclass(arg1, arg2)


def can_cast(from_dtype, to_dtype, casting):
    from_sctype = dtype(from_dtype).type.torch_dtype
    to_sctype = dtype(to_dtype).type.torch_dtype

    return _scalar_types._can_cast_impl(from_sctype, to_sctype, casting)


# XXX : used in _ndarray.py/result_type, clean up
from ._detail._casting_dicts import _result_type_dict

