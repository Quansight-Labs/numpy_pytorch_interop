from __future__ import annotations

from typing import Optional, Union

import torch

from . import _binary_ufuncs_impl, _dtypes_impl, _helpers, _unary_ufuncs_impl, _util
from ._normalizations import (
    ArrayLike,
    CastingModes,
    DTypeLike,
    NotImplementedType,
    OutArray,
    Scalar,
    normalizer,
)


def _ufunc_postprocess(result, out, casting):
    if out is not None:
        result = _util.typecast_tensor(result, out.dtype.torch_dtype, casting)
        result = torch.broadcast_to(result, out.shape)
    return result


# ############# Binary ufuncs ######################

_binary = [
    name
    for name in dir(_binary_ufuncs_impl)
    if not name.startswith("_") and name not in ["torch", "matmul", "divmod"]
]


NEP50_FUNCS = ("add", "sub", "mul", "floordiv", "mod", "and", "or", "xor", "lshift", "rshift")


def deco_binary_ufunc(torch_func):
    """Common infra for binary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """

    @normalizer
    def wrapped(
        x1: Union[ArrayLike, Scalar],
        x2: Union[ArrayLike, Scalar],
        /,
        out: Optional[OutArray] = None,
        *,
        where=True,
        casting: Optional[CastingModes] = "same_kind",
        order="K",
        dtype: Optional[DTypeLike] = None,
        subok: NotImplementedType = False,
        signature=None,
        extobj=None,
    ):
        x1, x2 = _dtypes_impl.nep50_to_tensors(x1, x2, torch_func.__name__ in NEP50_FUNCS)

        if dtype is None:
            dtype = _dtypes_impl.result_type_impl(x1, x2)
        x1, x2 = _util.typecast_tensors((x1, x2), dtype, casting)

        result = torch_func(x1, x2)

        result = _ufunc_postprocess(result, out, casting)
        return result

    wrapped.__qualname__ = torch_func.__name__
    wrapped.__name__ = torch_func.__name__

    return wrapped


#
# matmul's signature is _slightly_ different from other ufuncs:
# - no where=...
# - additional axis=..., axes=...
# - no NEP50 scalars in or out
#
@normalizer
def matmul(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    out: Optional[OutArray] = None,
    *,
    casting: Optional[CastingModes] = "same_kind",
    order: NotImplementedType = "K",
    dtype: Optional[DTypeLike] = None,
    subok: NotImplementedType = False,
    signature: NotImplementedType = None,
    extobj: NotImplementedType = None,
    axes: NotImplementedType = None,
    axis: NotImplementedType = None,
):

    if dtype is None:
        dtype = _dtypes_impl.result_type_impl(x1, x2)
    x1, x2 = _util.typecast_tensors((x1, x2), dtype, casting)

    result = _binary_ufuncs_impl.matmul(x1, x2)

    result = _ufunc_postprocess(result, out, casting)
    return result


#
# nin=2, nout=2
#
@normalizer
def divmod(
    x1: ArrayLike,
    x2: ArrayLike,
    out1: Optional[OutArray] = None,
    out2: Optional[OutArray] = None,
    /,
    out: tuple[Optional[OutArray], Optional[OutArray]] = (None, None),
    *,
    where: NotImplementedType = True,
    casting: Optional[CastingModes] = "same_kind",
    order: NotImplementedType = "K",
    dtype: Optional[DTypeLike] = None,
    subok: NotImplementedType = False,
    signature: NotImplementedType = None,
    extobj: NotImplementedType = None,
):
    # make sure we either have no out arrays at all, or there is either
    # out1, out2, or out=tuple, but not both
    num_outs = sum(x is not None for x in [out1, out2])
    if num_outs == 1:
        raise ValueError("both out1 and out2 need to be provided")
    elif num_outs == 2:
        o1, o2 = out
        if o1 is not None or o2 is not None:
            raise TypeError(
                "cannot specify 'out' as both a positional and keyword argument"
            )
    else:
        out1, out2 = out

    if dtype is None:
        dtype = _dtypes_impl.result_type_impl(x1, x2)
    x1, x2 = _util.typecast_tensors((x1, x2), dtype, casting)

    quot, rem = _binary_ufuncs_impl.divmod(x1, x2)

    quot = _ufunc_postprocess(quot, out1, casting)
    rem = _ufunc_postprocess(rem, out2, casting)
    return quot, rem


#
# Attach ufuncs to this module, for a further export to the public namespace in __init__.py
#
for name in _binary:
    ufunc = getattr(_binary_ufuncs_impl, name)
    vars()[name] = deco_binary_ufunc(ufunc)


def modf(x, /, *args, **kwds):
    quot, rem = divmod(x, 1, *args, **kwds)
    return rem, quot


_binary = _binary + ["divmod", "modf", "matmul"]


# ############# Unary ufuncs ######################


_unary = [
    name
    for name in dir(_unary_ufuncs_impl)
    if not name.startswith("_") and name != "torch"
]


def deco_unary_ufunc(torch_func):
    """Common infra for unary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """

    @normalizer
    def wrapped(
        x: ArrayLike,
        /,
        out: Optional[OutArray] = None,
        *,
        where=True,
        casting: Optional[CastingModes] = "same_kind",
        order="K",
        dtype: Optional[DTypeLike] = None,
        subok: NotImplementedType = False,
        signature=None,
        extobj=None,
    ):
        if dtype is not None:
            x = _util.typecast_tensor(x, dtype, casting)
        result = torch_func(x)
        result = _ufunc_postprocess(result, out, casting)
        return result

    wrapped.__qualname__ = torch_func.__name__
    wrapped.__name__ = torch_func.__name__

    return wrapped


#
# Attach ufuncs to this module, for a further export to the public namespace in __init__.py
#
for name in _unary:
    ufunc = getattr(_unary_ufuncs_impl, name)
    vars()[name] = deco_unary_ufunc(ufunc)


__all__ = _binary + _unary
