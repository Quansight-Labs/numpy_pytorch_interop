from typing import Optional

import torch

from . import _helpers
from ._detail import _binary_ufuncs
from ._normalizations import ArrayLike, DTypeLike, NDArray, SubokLike, normalizer

__all__ = [
    name
    for name in dir(_binary_ufuncs)
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
    result = _binary_ufuncs.matmul(*tensors)
    return result


#
# For each torch ufunc implementation, decorate and attach the decorated name
# to this module. Its contents is then exported to the public namespace in __init__.py
#
for name in __all__:
    ufunc = getattr(_binary_ufuncs, name)
    decorated = normalizer(deco_binary_ufunc(ufunc))

    decorated.__qualname__ = name  # XXX: is this really correct?
    decorated.__name__ = name
    vars()[name] = decorated


def modf(x, /, *args, **kwds):
    quot, rem = divmod(x, 1, *args, **kwds)
    return rem, quot


__all__ = __all__ + ["divmod", "modf", "matmul"]
