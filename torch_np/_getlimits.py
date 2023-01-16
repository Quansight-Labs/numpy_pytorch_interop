import torch

from . import _dtypes


def finfo(dtyp):
    torch_dtype = _dtypes.torch_dtype_from(dtyp)
    return torch.finfo(torch_dtype)


def iinfo(dtyp):
    torch_dtype = _dtypes.torch_dtype_from(dtyp)
    return torch.iinfo(torch_dtype)


import contextlib


# FIXME: this is only a stub
@contextlib.contextmanager
def errstate(*args, **kwds):
    yield
