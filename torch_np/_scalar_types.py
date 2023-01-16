"""Replicate the NumPy scalar type hierarchy
"""

import abc

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
        from . import _dtypes, _ndarray

        torch_dtype = _dtypes.torch_dtype_from(self.name)

        if isinstance(value, str) and value in ["inf", "nan"]:
            value = {"inf": torch.inf, "nan": torch.nan}[value]

        if isinstance(value, _ndarray.ndarray):
            tensor = value.get()
        else:
            try:
                tensor = torch.as_tensor(value, dtype=torch_dtype)
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


class int16(signedinteger):
    name = "int16"


class int32(signedinteger):
    name = "int32"


class int64(signedinteger):
    name = "int64"


# unsigned integers


class uint8(unsignedinteger):
    name = "uint8"


# floating point


class float16(floating):
    name = "float16"


class float32(floating):
    name = "float32"


class float64(floating):
    name = "float64"


class complex64(complexfloating):
    name = "complex64"


class complex128(complexfloating):
    name = "complex128"


class bool_(generic):
    name = "bool"


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


_typemap = {
    "int8": int8,
    "int16": int16,
    "int32": int32,
    "int64": int64,
    "uint8": uint8,
    "float16": float16,
    "float32": float32,
    "float64": float64,
    "complex64": complex64,
    "complex128": complex128,
    "bool": bool_,
}


# Replicate this -- yet another --- NumPy-defined way of grouping scalar types,
# cf tests/core/test_scalar_methods.py
sctypes = {
    "int": [int8, int16, int32, int64],
    "uint": [
        uint8,
    ],
    "float": [float16, float32, float64],
    "complex": [complex64, complex128],
    "others": [bool],
}


__all__ = list(_typemap.keys())
__all__.remove("bool")

__all__ += [
    "bool_",
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
