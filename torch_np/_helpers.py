import operator

import torch
from . import _dtypes
from ._ndarray import ndarray, asarray

from ._detail import _util


def cast_and_broadcast(tensors, out, casting):
    """Cast dtypes of arrays to out.dtype and broadcast if needed.

    Parameters
    ----------
    arrays : sequence of arrays
        Each element is broadcast against `out` and typecast to out.dtype
    out : the "output" array
        Not modified.
    casting : str
        One of numpy casting modes

    Returns
    -------
    tensors : tuple of Tensors
        Each tensor is dtype-cast and broadcast agains `out`, as needed

    Notes
    -----
    The `out` arrays broadcasts and dtype-casts `arrays`, but not vice versa.

    """
    if out is None:
        return tensors
    else:
        if not isinstance(out, ndarray):
            raise TypeError("Return arrays must be of ArrayType")

        tensors = _util.cast_and_broadcast(tensors, out.dtype.type.torch_dtype, out.shape, casting)

    return tuple(tensors)


def result_or_out(result_tensor, out_array=None):
    """A helper for returns with out= argument."""
    if out_array is not None:
        if not isinstance(out_array, ndarray):
            raise TypeError("Return arrays must be of ArrayType")
        if result_tensor.shape != out_array.shape:
            raise ValueError("Bad size of the out array.")
        out_tensor = out_array.get()
        out_tensor.copy_(result_tensor)
        return out_array
    else:
        return asarray(result_tensor)


def ndarrays_to_tensors(*inputs):
    """Convert all ndarrays from `inputs` to tensors. (other things are intact)
    """
    return tuple([value.get() if isinstance(value, ndarray) else value
            for value in inputs])


def to_tensors(*inputs):
    """Convert all array_likes from `inputs` to tensors.
    """
    return tuple(asarray(value).get() for value in inputs)

