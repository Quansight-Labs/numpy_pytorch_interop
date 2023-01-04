""" Define the supported dtypes and numpy <--> torch dtype mapping, define casting rules. 
"""

# TODO: 1. define torch_np dtypes, make this work without numpy.
#       2. mimic numpy's various aliases (np.half == np.float16, dtype='i8' etc)
#       3. convert from python types: np.ones(3, dtype=float) etc

import builtins
import torch

from . import _scalar_types


__all__ = ['dtype_from_torch', 'dtype', 'typecodes', 'issubdtype']


# Define analogs of numpy dtypes supported by pytorch.

class dtype:
    def __init__(self, name):
        if isinstance(name, dtype):
            _name = name.name
        elif name in python_types_dict:
            _name = python_types_dict[name]
        elif name in dt_names:
            _name = name
        elif name in typecode_chars_dict:
            _name = typecode_chars_dict[name]
        elif name in dt_aliases_dict:
            _name = dt_aliases_dict[name]
       # the check must come last, so that 'name' is not a string
        elif issubclass(name, _scalar_types.generic):
            _name = name.name
        else:
            raise TypeError(f"data type '{name}' not understood")
        self._name = _name

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return _scalar_types._typemap[self._name]

    @property
    def typecode(self):
        return _typecodes_from_dtype_dict[self._name]

    def __eq__(self, other):
        if isinstance(other, dtype):
            return self._name == other.name
        else:
            try:
                other_instance = dtype(other)
            except TypeError:
                return False
            return self._name == other_instance.name

    def __repr__(self):
        return f'dtype("{self.name}")'

    __str__ = __repr__


dt_names = ['float16', 'float32', 'float64',
         'complex64', 'complex128',
         'uint8',
         'int8',
         'int16',
         'int32',
         'int64',
         'bool']


dt_aliases_dict = {
    'i1' : 'int8',
    'i2' : 'int16',
    'i4' : 'int32',
    'i8' : 'int64',
    'b'  : 'int8',   # XXX: srsly?
    'f2' : 'float16',
    'f4' : 'float32',
    'f8' : 'float64',
    'c8' : 'complex64',
    'c16': 'complex128',
    '?'  : 'bool',
}


python_types_dict = {
    int: 'int64',
    float: 'float64',
    builtins.bool: 'bool'
}


typecode_chars_dict = {
    'e': 'float16',
    'f': 'float32',
    'd': 'float64',
    'F': 'complex64',
    'D': 'complex128',
    'B': 'uint8',
    'b': 'int8',
    'h': 'int16',
    'i': 'int32',
    'l': 'int64',
    '?': 'bool'
}

# reverse mapping
_typecodes_from_dtype_dict = {typecode_chars_dict[key]: key
                                for key in typecode_chars_dict}


typecodes = {'All': 'efdFDBbhil?',
        'AllFloat': 'efdFD',
        'AllInteger': 'Bbhil',
}


# Map the torch-suppored subset dtypes to local analogs
# "quantized" types not available in numpy, skip
_dtype_from_torch_dict = {
        # floating-point
        torch.float16: 'float16',
        torch.float32: 'float32',
        torch.float64 : 'float64',
        # np.complex32 does not exist
        torch.complex64: 'complex64',
        torch.complex128: 'complex128',
        # integer, unsigned (unit8 only, torch.uint32 etc do not exist)
        torch.uint8: 'uint8',
        # integer
        torch.int8: 'int8',
        torch.int16: 'int16',
        torch.int32: 'int32',
        torch.int64: 'int64',
        # boolean
        torch.bool : 'bool'
}


# reverse mapping
_torch_dtype_from_dtype_dict = {_dtype_from_torch_dict[key]: key
                                for key in _dtype_from_torch_dict}


def dtype_from_torch(torch_dtype):
    try:
        name = _dtype_from_torch_dict[torch_dtype]
        return dtype(name)
    except KeyError:
        # mimic numpy: >>> np.dtype('unknown') -->  TypeError
        raise TypeError


def torch_dtype_from(dtyp):
    if dtyp is None:
        return None
    name = dtype(dtyp).name
    try:
        return _torch_dtype_from_dtype_dict[name]
    except KeyError:
        # mimic numpy: >>> np.dtype('unknown') -->  TypeError
        raise TypeError


def default_int_type():
    return dtype('int64')


def default_float_type():
    return dtype('float64')


def is_floating(dtyp):
    dtyp = dtype(dtyp)
    return dtyp.typecode in typecodes['AllFloat']

def is_integer(dtyp):
    dtyp = dtype(dtyp)
    return dtyp.typecode in typecodes['AllInteger']



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


# The casting below is defined *with dtypes only*, so no value-based casting!

# These two dicts are autogenerated with autogen/gen_dtypes.py,
# using numpy version 1.23.5.

_can_cast_dict = {
'no': {'float16': {'float16': True, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float32': {'float16': False, 'float32': True, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float64': {'float16': False, 'float32': False, 'float64': True, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex64': {'float16': False, 'float32': False, 'float64': False, 'complex64': True, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex128': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'uint8': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': True, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'int8': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': True, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'int16': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': True, 'int32': False, 'int64': False, 'bool': False}, 'int32': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': True, 'int64': False, 'bool': False}, 'int64': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': True, 'bool': False}, 'bool': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': True}},

'equiv': {'float16': {'float16': True, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float32': {'float16': False, 'float32': True, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float64': {'float16': False, 'float32': False, 'float64': True, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex64': {'float16': False, 'float32': False, 'float64': False, 'complex64': True, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex128': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'uint8': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': True, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'int8': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': True, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'int16': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': True, 'int32': False, 'int64': False, 'bool': False}, 'int32': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': True, 'int64': False, 'bool': False}, 'int64': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': True, 'bool': False}, 'bool': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': True}},

'safe': {'float16': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float32': {'float16': False, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float64': {'float16': False, 'float32': False, 'float64': True, 'complex64': False, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex64': {'float16': False, 'float32': False, 'float64': False, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex128': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'uint8': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': False, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int8': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int16': {'float16': False, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int32': {'float16': False, 'float32': False, 'float64': True, 'complex64': False, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': True, 'int64': True, 'bool': False}, 'int64': {'float16': False, 'float32': False, 'float64': True, 'complex64': False, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': True, 'bool': False}, 'bool': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}},

'same_kind': {'float16': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float32': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float64': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex64': {'float16': False, 'float32': False, 'float64': False, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex128': {'float16': False, 'float32': False, 'float64': False, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'uint8': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int8': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int16': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int32': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int64': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'bool': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}},

'unsafe': {'float16': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'float32': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'float64': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'complex64': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'complex128': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'uint8': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'int8': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'int16': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'int32': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'int64': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'bool': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}}
}


_result_type_dict = {
'float16': {'float16': 'float16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'float16', 'int8': 'float16', 'int16': 'float32', 'int32': 'float64', 'int64': 'float64', 'bool': 'float16'}, 
'float32': {'float16': 'float32', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'float32', 'int8': 'float32', 'int16': 'float32', 'int32': 'float64', 'int64': 'float64', 'bool': 'float32'},
'float64': {'float16': 'float64', 'float32': 'float64', 'float64': 'float64', 'complex64': 'complex128', 'complex128': 'complex128', 'uint8': 'float64', 'int8': 'float64', 'int16': 'float64', 'int32': 'float64', 'int64': 'float64', 'bool': 'float64'},
'complex64': {'float16': 'complex64', 'float32': 'complex64', 'float64': 'complex128', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'complex64', 'int8': 'complex64', 'int16': 'complex64', 'int32': 'complex128', 'int64': 'complex128', 'bool': 'complex64'},
'complex128': {'float16': 'complex128', 'float32': 'complex128', 'float64': 'complex128', 'complex64': 'complex128', 'complex128': 'complex128', 'uint8': 'complex128', 'int8': 'complex128', 'int16': 'complex128', 'int32': 'complex128', 'int64': 'complex128', 'bool': 'complex128'},
'uint8': {'float16': 'float16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'uint8', 'int8': 'int16', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'bool': 'uint8'},
'int8': {'float16': 'float16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'int16', 'int8': 'int8', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'bool': 'int8'},
'int16': {'float16': 'float32', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'int16', 'int8': 'int16', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'bool': 'int16'},
'int32': {'float16': 'float64', 'float32': 'float64', 'float64': 'float64', 'complex64': 'complex128', 'complex128': 'complex128', 'uint8': 'int32', 'int8': 'int32', 'int16': 'int32', 'int32': 'int32', 'int64': 'int64', 'bool': 'int32'},
'int64': {'float16': 'float64', 'float32': 'float64', 'float64': 'float64', 'complex64': 'complex128', 'complex128': 'complex128', 'uint8': 'int64', 'int8': 'int64', 'int16': 'int64', 'int32': 'int64', 'int64': 'int64', 'bool': 'int64'},
'bool': {'float16': 'float16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'uint8', 'int8': 'int8', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'bool': 'bool'}}

########################## end autogenerated part

