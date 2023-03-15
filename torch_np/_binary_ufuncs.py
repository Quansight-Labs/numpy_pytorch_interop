from typing import Optional

from . import _helpers
from ._detail import _binary_ufuncs
from ._normalizations import (
    ArrayLike,
    DTypeLike,
    NDArray,
    OutArray,
    SubokLike,
    normalizer,
)

__all__ = [
    name for name in dir(_binary_ufuncs) if not name.startswith("_") and name != "torch"
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
    ) -> OutArray:
        tensors = _helpers.ufunc_preprocess(
            (x1, x2), out, where, casting, order, dtype, subok, signature, extobj
        )
        result = torch_func(*tensors)
        return result, out

    return wrapped


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


# a stub implementation of divmod, should be improved after
# https://github.com/pytorch/pytorch/issues/90820 is fixed in pytorch
#
# Implementation details: we just call two ufuncs which have been created
# just above, for x1 // x2 and x1 % x2.
# This means we are normalizing x1, x2 in each of the ufuncs --- note that there
# is no @normalizer on divmod.


def divmod(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok: SubokLike = False,
    signature=None,
    extobj=None,
):
    out1, out2 = None, None
    if out is not None:
        out1, out2 = out

    kwds = dict(
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
        signature=signature,
        extobj=extobj,
    )

    # NB: use local names for
    quot = floor_divide(x1, x2, out=out1, **kwds)
    rem = remainder(x1, x2, out=out2, **kwds)

    quot = _helpers.result_or_out(quot.get(), out1)  # FIXME: .get() -> .tensor
    rem = _helpers.result_or_out(rem.get(), out2)

    return quot, rem


def modf(x, /, *args, **kwds):
    quot, rem = divmod(x, 1, *args, **kwds)
    return rem, quot


__all__ = __all__ + ["divmod", "modf"]
