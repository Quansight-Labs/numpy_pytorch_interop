import typing

import torch

from . import _decorators, _helpers
from ._detail import _dtypes_impl, _flips, _util
from ._detail import implementations as _impl

################################## normalizations

ArrayLike = typing.TypeVar("ArrayLike")
DTypeLike = typing.TypeVar("DTypeLike")
SubokLike = typing.TypeVar("SubokLike")


import inspect

from . import _dtypes


def normalize_array_like(x, name=None):
    (tensor,) = _helpers.to_tensors(x)
    return tensor


def normalize_dtype(dtype, name=None):
    # cf _decorators.dtype_to_torch
    torch_dtype = None
    if dtype is not None:
        dtype = _dtypes.dtype(dtype)
        torch_dtype = dtype.torch_dtype
    return torch_dtype


def normalize_subok_like(arg, name):
    if arg:
        raise ValueError(f"'{name}' parameter is not supported.")


normalizers = {
    ArrayLike: normalize_array_like,
    DTypeLike: normalize_dtype,
    SubokLike: normalize_subok_like,
}

import functools


def normalizer(func):
    @functools.wraps(func)
    def wrapped(*args, **kwds):
        sig = inspect.signature(func)

        dct = {}
        # loop over positional parameters and actual arguments
        for arg, (name, parm) in zip(args, sig.parameters.items()):
            print(arg, name, parm.annotation)
            normalizer = normalizers.get(parm.annotation, None)
            if normalizer:
                dct[name] = normalizer(arg, name)
            else:
                # untyped arguments pass through
                dct[name] = arg

        # normalize keyword arguments
        for name, arg in kwds.items():
            if not name in sig.parameters:
                # unknown kwarg, bail out
                raise TypeError(
                    f"{func.__name__}() got an unexpected keyword argument '{name}'."
                )

            print("kw: ", name, sig.parameters[name].annotation)
            parm = sig.parameters[name]
            normalizer = normalizers.get(parm.annotation, None)
            if normalizer:
                dct[name] = normalizer(kwds[name], name)
            else:
                dct[name] = arg

        ba = sig.bind(**dct)
        ba.apply_defaults()

        # Now that all parameters have been consumed, check:
        # Anything that has not been bound is unexpected positional arg => raise.
        # If there are too few actual arguments, this fill fail in func(*ba.args) below
        if len(args) > len(ba.args):
            raise TypeError(
                f"{func.__name__}() takes {len(ba.args)} positional argument but {len(args)} were given."
            )

        # TODO:
        # 2. [LOOKS OK] extra unknown args -- error out : nonzero([2, 0, 3], oops=42)
        # 3. [LOOKS OK] optional (tensor_or_none) : untyped => pass through
        # 4. [LOOKS OK] DTypeLike : positional or kw
        # 5. axes : live in _impl or in types? several ways of handling them
        # 6. keepdims : peel off, postprocess
        # 7. OutLike : normal & keyword-only, peel off, postprocess

        # finally, pass normalized arguments through
        result = func(*ba.args)
        return result

    return wrapped


##################################


@normalizer
def nonzero(a: ArrayLike):
    #   (tensor,) = _helpers.to_tensors(a)
    result = a.nonzero(as_tuple=True)
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


@normalizer
def trace(a: ArrayLike, offset=0, axis1=0, axis2=1, dtype: DTypeLike = None, out=None):
    #    (tensor,) = _helpers.to_tensors(a)
    result = _impl.trace(a, offset, axis1, axis2, dtype)
    return _helpers.result_or_out(result, out)


@normalizer
def eye(N, M=None, k=0, dtype: DTypeLike = float, order="C", *, like: SubokLike = None):
    #    _util.subok_not_ok(like)
    if order != "C":
        raise NotImplementedError
    result = _impl.eye(N, M, k, dtype)
    return _helpers.array_from(result)


@normalizer
def identity(n, dtype: DTypeLike = None, *, like: SubokLike = None):
    ##   _util.subok_not_ok(like)
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
