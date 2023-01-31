"""Replicate the NumPy scalar type hierarchy
"""
import abc
import builtins

import torch


class generic(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self):
        return self.__class__.__name__

    def __new__(self, value):
        #
        # Yes, a call to np.float32(4) produces a zero-dim array.
        #
        from .. import _ndarray

        if isinstance(value, str) and value in ["inf", "nan"]:
            value = {"inf": torch.inf, "nan": torch.nan}[value]

        if isinstance(value, _ndarray.ndarray):
            tensor = value.get()
        else:
            try:
                tensor = torch.as_tensor(value, dtype=self.torch_dtype)
            except RuntimeError as e:
                if "Overflow" in str(e):
                    raise OverflowError(e.args)
                raise e
        #
        # With numpy:
        # >>> a = np.ones(3)
        # >>> np.float64(a) is a        # True
        # >>> np.float64(a[0]) is a[0]  # False
        #
        # A reasonable assumption is that the second case is more common,
        # and here we follow the second approach and create a new object
        # *for all inputs*.
        #
        return _ndarray.ndarray._from_tensor_and_base(tensor, None)


##### these are abstract types


class number(generic):
    pass


class integer(number):
    pass


class inexact(number):
    pass


class signedinteger(integer):
    pass


class unsignedinteger(integer):
    pass


class floating(inexact):
    pass


class complexfloating(inexact):
    pass


# ##### concrete types

# signed integers


class int8(signedinteger):
    name = "int8"
    typecode = "b"
    torch_dtype = torch.int8


class int16(signedinteger):
    name = "int16"
    typecode = "h"
    torch_dtype = torch.int16


class int32(signedinteger):
    name = "int32"
    typecode = "i"
    torch_dtype = torch.int32


class int64(signedinteger):
    name = "int64"
    typecode = "l"
    torch_dtype = torch.int64


# unsigned integers


class uint8(unsignedinteger):
    name = "uint8"
    typecode = "B"
    torch_dtype = torch.uint8


# floating point


class float16(floating):
    name = "float16"
    typecode = "e"
    torch_dtype = torch.float16


class float32(floating):
    name = "float32"
    typecode = "f"
    torch_dtype = torch.float32


class float64(floating):
    name = "float64"
    typecode = "d"
    torch_dtype = torch.float64


class complex64(complexfloating):
    name = "complex64"
    typecode = "F"
    torch_dtype = torch.complex64


class complex128(complexfloating):
    name = "complex128"
    typecode = "D"
    torch_dtype = torch.complex128


class bool_(generic):
    name = "bool_"
    typecode = "?"
    torch_dtype = torch.bool


# name aliases : FIXME (OS, bitness)
intp = int64
int_ = int64
intc = int32

byte = int8
short = int16
longlong = int64  # XXX: is this correct?

ubyte = uint8

half = float16
single = float32
double = float64
float_ = float64

csingle = complex64
cdouble = complex128


# Replicate this NumPy-defined way of grouping scalar types,
# cf tests/core/test_scalar_methods.py
sctypes = {
    "int": [int8, int16, int32, int64],
    "uint": [
        uint8,
    ],
    "float": [float16, float32, float64],
    "complex": [complex64, complex128],
    "others": [bool_],
}


_names = {st.name: st for cat in sctypes for st in sctypes[cat]}
_typecodes = {st.typecode: st for cat in sctypes for st in sctypes[cat]}
_torch_dtypes = {st.torch_dtype: st for cat in sctypes for st in sctypes[cat]}

_aliases = {
    "u1": uint8,
    "i1": int8,
    "i2": int16,
    "i4": int32,
    "i8": int64,
    "b": int8,  # XXX: srsly?
    "f2": float16,
    "f4": float32,
    "f8": float64,
    "c8": complex64,
    "c16": complex128,
    # numpy-specific trailing underscore
    "bool_": bool_,
}


_python_types = {
    int: int64,
    float: float64,
    complex: complex128,
    builtins.bool: bool_,
    # also allow stringified names of python types
    int.__name__: int64,
    float.__name__: float64,
    complex.__name__: complex128,
    builtins.bool.__name__: bool_,
}


def sctype_from_string(s):
    """Normalize a string value: a type 'name' or a typecode or a width alias."""
    if s in _names:
        return _names[s]
    if s in _typecodes:
        return _typecodes[s]
    if s in _aliases:
        return _aliases[s]
    if s in _python_types:
        return _python_types[s]
    raise TypeError(f"data type '{s}' not understood")


def sctype_from_torch_dtype(torch_dtype):
    return _torch_dtypes[torch_dtype]


#### default : mimic NumPy
default_scalar_type = float64
default_int_type = int64
default_float_type = float64
default_complex_type = complex128
##########################


def get_default_type_for(sctype):
    """Default scalar type given sctype category."""
    if issubclass(sctype, integer):
        result = default_int_type
    elif issubclass(sctype, floating):
        result = default_float_type
    elif issubclass(sctype, complexfloating):
        result = default_complex_type
    elif issubclass(sctype, bool_):
        result = bool_
    else:
        raise RuntimeError("cannot be here with sctype= %s" % sctype)
    return result


# XXX: is it ever used? cf _detail/reductions.py::_atleast_float(...)
def float_or_default(sctype, enforce_float=False):
    """bool -> int; int -> float"""
    if issubclass(sctype, bool_):
        sctype = default_int_type
    if enforce_float and issubclass(sctype, integer):
        sctype = default_float_type
    return sctype


from . import _casting_dicts as _cd


def _can_cast_sctypes(from_sctype, to_sctype, casting):
    return _can_cast_impl(from_sctype.torch_dtype, to_sctype.torch_dtype, casting)


def _can_cast_impl(from_torch_dtype, to_torch_dtype, casting):
    return _cd._can_cast_dict[casting][from_torch_dtype][to_torch_dtype]


__all__ = list(_names.keys())
__all__ += [
    "intp",
    "int_",
    "intc",
    "byte",
    "short",
    "longlong",
    "ubyte",
    "half",
    "single",
    "double",
    "csingle",
    "cdouble",
    "float_",
]
__all__ += ["sctypes"]
__all__ += [
    "generic",
    "number",
    "integer",
    "signedinteger",
    "unsignedinteger",
    "inexact",
    "floating",
    "complexfloating",
]
