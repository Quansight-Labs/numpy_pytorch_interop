from typing import Optional

import torch

from . import _binary_ufuncs_impl, _helpers, _unary_ufuncs_impl
from ._normalizations import ArrayLike, DTypeLike, NDArray, SubokLike, normalizer

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
        out: Optional[NDArray] = None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype: DTypeLike = None,
        subok: SubokLike = False,
        signature=None,
        extobj=None,
    ):
        tensors = _helpers.ufunc_preprocess(
            (x1, x2), out, where, casting, order, dtype, subok, signature, extobj
        )
        # now broadcast input tensors against the out=... array
        if out is not None:
            # XXX: need to filter out noop broadcasts if t.shape == out.shape?
            shape = out.shape
            tensors = tuple(torch.broadcast_to(t, shape) for t in tensors)

        result = torch_func(*tensors)
        return result

    return wrapped


#
# matmul is special in that its `out=...` array does not broadcast x1 and x2.
# E.g. consider x1.shape = (5, 2) and x2.shape = (2, 3). Then `out.shape` is (5, 3).
#
@normalizer
def matmul(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    out: Optional[NDArray] = None,
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
    tensors = _helpers.ufunc_preprocess(
        (x1, x2), out, True, casting, order, dtype, subok, signature, extobj
    )
    if axis is not None or axes is not None:
        raise NotImplementedError

    # NB: do not broadcast input tensors against the out=... array
    result = _binary_ufuncs_impl.matmul(*tensors)
    return result


def divmod(
    x1: ArrayLike,
    x2: ArrayLike,
    out1: Optional[NDArray] = None,
    out2: Optional[NDArray] = None,
    /,
    out: tuple[Optional[NDArray], Optional[NDArray]] = (None, None),
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

    tensors = _helpers.ufunc_preprocess(
        (x1, x2), out, True, casting, order, dtype, subok, signature, extobj
    )

    result = _binary_ufuncs_impl.divmod(*tensors)

    return quot, rem


#
# For each torch ufunc implementation, decorate and attach the decorated name
# to this module. Its contents is then exported to the public namespace in __init__.py
#
for name in _binary:
    ufunc = getattr(_binary_ufuncs_impl, name)
    decorated = normalizer(deco_binary_ufunc(ufunc))

    decorated.__qualname__ = name  # XXX: is this really correct?
    decorated.__name__ = name
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
        out: Optional[NDArray] = None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype: DTypeLike = None,
        subok: SubokLike = False,
        signature=None,
        extobj=None,
    ):
        tensors = _helpers.ufunc_preprocess(
            (x,), out, where, casting, order, dtype, subok, signature, extobj
        )
        # now broadcast the input tensor against the out=... array
        if out is not None:
            # XXX: need to filter out noop broadcasts if t.shape == out.shape?
            shape = out.shape
            tensors = tuple(torch.broadcast_to(t, shape) for t in tensors)
        result = torch_func(*tensors)
        return result

    return wrapped


#
# For each torch ufunc implementation, decorate and attach the decorated name
# to this module. Its contents is then exported to the public namespace in __init__.py
#
for name in _unary:
    ufunc = getattr(_unary_ufuncs_impl, name)
    decorated = normalizer(deco_unary_ufunc(ufunc))

    decorated.__qualname__ = name  # XXX: is this really correct?
    decorated.__name__ = name
    vars()[name] = decorated


__all__ = _binary + _unary
