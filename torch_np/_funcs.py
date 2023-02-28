import torch

from . import _decorators, _helpers
from ._detail import _dtypes_impl, _flips, _util
from ._detail import implementations as _impl


def nonzero(a):
    (tensor,) = _helpers.to_tensors(a)
    result = tensor.nonzero(as_tuple=True)
    return _helpers.tuple_arrays_from(result)


def argwhere(a):
    (tensor,) = _helpers.to_tensors(a)
    result = torch.argwhere(tensor)
    return _helpers.array_from(result)


def clip(a, min=None, max=None, out=None):
    # np.clip requires both a_min and a_max not None, while ndarray.clip allows
    # one of them to be None. Follow the more lax version.
    # Also min/max as arg names: follow numpy naming.
    tensor, t_min, t_max = _helpers.to_tensors_or_none(a, min, max)
    result = _impl.clip(tensor, t_min, t_max)
    return _helpers.result_or_out(result, out)


def repeat(a, repeats, axis=None):
    tensor, t_repeats = _helpers.to_tensors(a, repeats)  # XXX: scalar repeats
    result = torch.repeat_interleave(tensor, t_repeats, axis)
    return _helpers.array_from(result)


# ### diag et al ###


def diagonal(a, offset=0, axis1=0, axis2=1):
    (tensor,) = _helpers.to_tensors(a)
    result = _impl.diagonal(tensor, offset, axis1, axis2)
    return _helpers.array_from(result)


@_decorators.dtype_to_torch
def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    (tensor,) = _helpers.to_tensors(a)
    result = _impl.trace(tensor, offset, axis1, axis2, dtype)
    return _helpers.result_or_out(result, out)


@_decorators.dtype_to_torch
def eye(N, M=None, k=0, dtype=float, order="C", *, like=None):
    _util.subok_not_ok(like)
    if order != "C":
        raise NotImplementedError
    result = _impl.eye(N, M, k, dtype)
    return _helpers.array_from(result)


@_decorators.dtype_to_torch
def identity(n, dtype=None, *, like=None):
    _util.subok_not_ok(like)
    result = torch.eye(n, dtype=dtype)
    return _helpers.array_from(result)


def diag(v, k=0):
    (tensor,) = _helpers.to_tensors(v)
    result = torch.diag(tensor, k)
    return _helpers.array_from(result)


def diagflat(v, k=0):
    (tensor,) = _helpers.to_tensors(v)
    result = torch.diagflat(tensor, k)
    return _helpers.array_from(result)


def diag_indices(n, ndim=2):
    result = _impl.diag_indices(n, ndim)
    return _helpers.tuple_arrays_from(result)


def diag_indices_from(arr):
    (tensor,) = _helpers.to_tensors(arr)
    result = _impl.diag_indices_from(tensor)
    return _helpers.tuple_arrays_from(result)


def fill_diagonal(a, val, wrap=False):
    tensor, t_val = _helpers.to_tensors(a, val)
    result = _impl.fill_diagonal(tensor, t_val, wrap)
    return _helpers.array_from(result)


def vdot(a, b, /):
    t_a, t_b = _helpers.to_tensors(a, b)
    result = _impl.vdot(t_a, t_b)
    return result.item()


def dot(a, b, out=None):
    t_a, t_b = _helpers.to_tensors(a, b)
    dtype = _dtypes_impl.result_type_impl((t_a.dtype, t_b.dtype))
    t_a = t_a.to(dtype)
    t_b = t_b.to(dtype)
    result = _impl.dot(t_a, t_b)
    return _helpers.result_or_out(result, out)


# ### sort and partition ###


def sort(a, axis=-1, kind=None, order=None):
    (tensor,) = _helpers.to_tensors(a)
    result = _impl.sort(tensor, axis, kind, order)
    return _helpers.array_from(result)


def argsort(a, axis=-1, kind=None, order=None):
    (tensor,) = _helpers.to_tensors(a)
    result = _impl.argsort(tensor, axis, kind, order)
    return _helpers.array_from(result)


def searchsorted(a, v, side="left", sorter=None):
    a_t, v_t, sorter_t = _helpers.to_tensors_or_none(a, v, sorter)
    result = torch.searchsorted(a_t, v_t, side=side, sorter=sorter_t)
    return _helpers.array_from(result)


# ### swap/move/roll axis ###


def moveaxis(a, source, destination):
    (tensor,) = _helpers.to_tensors(a)
    result = _impl.moveaxis(tensor, source, destination)
    return _helpers.array_from(result)


def swapaxes(a, axis1, axis2):
    (tensor,) = _helpers.to_tensors(a)
    result = _flips.swapaxes(tensor, axis1, axis2)
    return _helpers.array_from(result)


def rollaxis(a, axis, start=0):
    (tensor,) = _helpers.to_tensors(a)
    result = _flips.rollaxis(a, axis, start)
    return _helpers.array_from(result)


# ### shape manipulations ###


def squeeze(a, axis=None):
    (tensor,) = _helpers.to_tensors(a)
    result = _impl.squeeze(tensor, axis)
    return _helpers.array_from(result, a)


def reshape(a, newshape, order="C"):
    (tensor,) = _helpers.to_tensors(a)
    result = _impl.reshape(tensor, newshape, order=order)
    return _helpers.array_from(result, a)


def transpose(a, axes=None):
    (tensor,) = _helpers.to_tensors(a)
    result = _impl.transpose(tensor, axes)
    return _helpers.array_from(result, a)


def ravel(a, order="C"):
    (tensor,) = _helpers.to_tensors(a)
    result = _impl.ravel(tensor)
    return _helpers.array_from(result, a)


# leading underscore since arr.flatten exists but np.flatten does not
def _flatten(a, order="C"):
    (tensor,) = _helpers.to_tensors(a)
    result = _impl._flatten(tensor)
    return _helpers.array_from(result, a)


# ### Type/shape etc queries ###


def real(a):
    (tensor,) = _helpers.to_tensors(a)
    result = torch.real(tensor)
    return _helpers.array_from(result)


def imag(a):
    (tensor,) = _helpers.to_tensors(a)
    result = _impl.imag(tensor)
    return _helpers.array_from(result)


def round_(a, decimals=0, out=None):
    (tensor,) = _helpers.to_tensors(a)
    result = _impl.round(tensor, decimals)
    return _helpers.result_or_out(result, out)


around = round_
round = round_
