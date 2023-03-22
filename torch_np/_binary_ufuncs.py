from typing import Optional

from . import _helpers
from ._detail import _binary_ufuncs
from ._normalizations import ArrayLike, DTypeLike, NDArray, SubokLike, normalizer

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
    ):
        tensors = _helpers.ufunc_preprocess(
            (x1, x2), out, where, casting, order, dtype, subok, signature, extobj
        )
        result = torch_func(*tensors)
        return _helpers.result_or_out(result, out)

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
