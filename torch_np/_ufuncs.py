from typing import Optional

import torch

from . import _binary_ufuncs_impl, _helpers, _unary_ufuncs_impl
from ._detail import _dtypes_impl, _util
from ._normalizations import ArrayLike, DTypeLike, OutArray, SubokLike, normalizer


def _ufunc_preprocess(tensors, where, casting, order, dtype, subok, signature, extobj):
    if order != "K" or not where or signature or extobj:
        raise NotImplementedError

    if dtype is None:
        dtype = _dtypes_impl.result_type_impl([t.dtype for t in tensors])

    tensors = _util.typecast_tensors(tensors, dtype, casting)

    return tensors


def _ufunc_postprocess(result, out, casting):
    if out is not None:
        (result,) = _util.typecast_tensors((result,), out.dtype.torch_dtype, casting)
        result = torch.broadcast_to(result, out.shape)
    return result


# ############# Binary ufuncs ######################

_binary = [
    name
    for name in dir(_binary_ufuncs_impl)
    if not name.startswith("_") and name not in ["torch", "matmul"]
]


def deco_binary_ufunc(torch_func):
    """Common infra for binary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """

    def wrapped(
        x1: ArrayLike,
        x2: ArrayLike,
        /,
        out: Optional[OutArray] = None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype: DTypeLike = None,
        subok: SubokLike = False,
        signature=None,
        extobj=None,
    ):
        tensors = _ufunc_preprocess(
            (x1, x2), where, casting, order, dtype, subok, signature, extobj
        )
        result = torch_func(*tensors)

        result = _ufunc_postprocess(result, out, casting)
        return result

    wrapped.__qualname__ = torch_func.__name__
    wrapped.__name__ = torch_func.__name__

    return wrapped


#
# matmul's signature is _slightly_ different from other ufuncs:
# - no where=...
# - additional axis=..., axes=...
#
@normalizer
def matmul(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    out: Optional[OutArray] = None,
    *,
    casting="same_kind",
    order="K",
    dtype: DTypeLike = None,
    subok: SubokLike = False,
    signature=None,
    extobj=None,
    axes=None,
    axis=None,
):
    tensors = _ufunc_preprocess(
        (x1, x2), True, casting, order, dtype, subok, signature, extobj
    )
    if axis is not None or axes is not None:
        raise NotImplementedError

    result = _binary_ufuncs_impl.matmul(*tensors)

    result = _ufunc_postprocess(result, out, casting)
    return result


#
# nin=2, nout=2
#
def divmod(
    x1: ArrayLike,
    x2: ArrayLike,
    out1: Optional[OutArray] = None,
    out2: Optional[OutArray] = None,
    /,
    out: tuple[Optional[OutArray], Optional[OutArray]] = (None, None),
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype: DTypeLike = None,
    subok: SubokLike = False,
    signature=None,
    extobj=None,
):
    num_outs = sum(x is None for x in [out1, out2])
    if sum_outs == 1:
        raise ValueError("both out1 and out2 need to be provided")
    if sum_outs != 0 and out != (None, None):
        raise ValueError("Either provide out1 and out2, or out.")
    if out is not None:
        out1, out2 = out
    if out1.shape != out2.shape or out1.dtype != out2.dtype:
        raise ValueError("out1, out2 must be compatible")

    tensors = _ufunc_preprocess(
        (x1, x2), True, casting, order, dtype, subok, signature, extobj
    )

    quot, rem = _binary_ufuncs_impl.divmod(*tensors)

    quot = _ufunc_postprocess(quot, out1, casting)
    rem = _ufunc_postprocess(rem, out2, casting)
    return quot, rem


#
# For each torch ufunc implementation, decorate and attach the decorated name
# to this module. Its contents is then exported to the public namespace in __init__.py
#
for name in _binary:
    ufunc = getattr(_binary_ufuncs_impl, name)
    decorated = normalizer(deco_binary_ufunc(ufunc))
    vars()[name] = decorated


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

    def wrapped(
        x: ArrayLike,
        /,
        out: Optional[OutArray] = None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype: DTypeLike = None,
        subok: SubokLike = False,
        signature=None,
        extobj=None,
    ):
        tensors = _ufunc_preprocess(
            (x,), where, casting, order, dtype, subok, signature, extobj
        )
        result = torch_func(*tensors)
        result = _ufunc_postprocess(result, out, casting)
        return result

    wrapped.__qualname__ = torch_func.__name__
    wrapped.__name__ = torch_func.__name__

    return wrapped


#
# For each torch ufunc implementation, decorate and attach the decorated name
# to this module. Its contents is then exported to the public namespace in __init__.py
#
for name in _unary:
    ufunc = getattr(_unary_ufuncs_impl, name)
    decorated = normalizer(deco_unary_ufunc(ufunc))
    vars()[name] = decorated


__all__ = _binary + _unary
