import torch

from . import _util
from . import _helpers


def add(x1_array, x2_array, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or not where:
        raise NotImplementedError

    # XXX: dtype=... parameter is silently ignored

    arrays = (x1_array, x2_array)
    x1_tensor, x2_tensor = _helpers.cast_and_broadcast(arrays, out, casting)

    result = torch.add(x1_tensor, x2_tensor)

    return _helpers.result_or_out(result, out)

