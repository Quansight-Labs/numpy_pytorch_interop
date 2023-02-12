""" Define the supported dtypes and numpy <--> torch dtype mapping, define casting rules. 
"""
import torch

from ._detail import _dtypes_impl, _scalar_types

__all__ = ["dtype", "DType", "typecodes", "issubdtype"]


# Define analogs of numpy dtypes supported by pytorch.


def dtype(arg):
    if arg is None:
        arg = _dtypes_impl.default_scalar_dtype
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
        elif hasattr(arg, "dtype"):
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


typecodes = {
    "All": "efdFDBbhil?",
    "AllFloat": "efdFD",
    "AllInteger": "Bbhil",
    "Integer": "bhil",
    "UnsignedInteger": "B",
    "Float": "efd",
    "Complex": "FD",
}


# ### Defaults and dtype discovery


def default_int_type():
    return dtype(_dtypes_impl.default_int_dtype)


def default_float_type():
    return dtype(_dtypes_impl.default_float_dtype)


def default_complex_type():
    return dtype(_dtypes_impl.default_complex_dtype)


def get_default_dtype_for(dtyp):
    torch_dtype = dtype(dtyp).torch_dtype
    return _dtypes_impl.get_default_type_for(torch_dtype)


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


# XXX : used in _ndarray.py/result_type, clean up
from ._detail._casting_dicts import _result_type_dict
