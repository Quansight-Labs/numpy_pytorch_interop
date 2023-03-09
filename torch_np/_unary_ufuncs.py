#from ._decorators import deco_unary_ufunc_from_impl
#from ._detail import _ufunc_impl


from ._detail import _unary_ufuncs

__all__ = [name for name in dir(_unary_ufuncs) if not name.startswith("_") and name != "torch"]


# TODO: consolidate normalizations
from ._funcs import normalizer, ArrayLike, SubokLike, DTypeLike
from ._detail import _util
from . import _helpers
#import torch


def deco_unary_ufunc(torch_func):
    """Common infra for unary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """
    def wrapped(
        x : ArrayLike,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype: DTypeLike=None,
        subok: SubokLike=False,
        signature=None,
        extobj=None
    ):
        if order != "K" or not where or signature or extobj:
            raise NotImplementedError

        # XXX: dtype=... parameter
        if dtype is not None:
            raise NotImplementedError

        out_shape_dtype = None
        if out is not None:
            out_shape_dtype = (out.get().dtype, out.get().shape)

        tensors = _util.cast_and_broadcast((x,), out_shape_dtype, casting)

        result = torch_func(*tensors)
        return _helpers.result_or_out(result, out)

    return wrapped

#
# For each torch ufunc implementation, decorate and attach the decorated name
# to this module. Its contents is then exported to the public namespace in __init__.py
#
for name in __all__:
    ufunc = getattr(_unary_ufuncs, name)
    decorated = normalizer(deco_unary_ufunc(ufunc))

    decorated.__qualname__ = name    # XXX: is this really correct?
    decorated.__name__ = name
    vars()[name] = decorated
