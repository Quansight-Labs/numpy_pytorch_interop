import torch

from . import _util
from . import _helpers

def deco_binary_ufunc(torch_func):
    """Common infra for binary ufuncs: receive arrays, sort out type casting,
       broadcasting, out array handling etc, and delegate to the
       pytorch function for actual work, then wrap the results into an array.

       x1, x2 are arrays! array_like -> array conversion is the caller responsibility.
    """
    def wrapped(x1, x2, /, out=None, *, where=True,
                casting='same_kind', order='K', dtype=None, subok=False, **kwds):
        _util.subok_not_ok(subok=subok)
        if order != 'K' or not where:
            raise NotImplementedError

        # XXX: dtype=... parameter is silently ignored

        arrays = (x1_array, x2_array)
        x1_tensor, x2_tensor = _helpers.cast_and_broadcast(arrays, out, casting)

        result = torch_func(x1_tensor, x2_tensor)

        return _helpers.result_or_out(result, out)
    return wrapped



add = deco_binary_ufunc(torch.add)
subtract = deco_binary_ufunc(torch.subtract)
multiply = deco_binary_ufunc(torch.multiply)

