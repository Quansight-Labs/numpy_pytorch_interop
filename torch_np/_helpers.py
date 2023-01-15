import operator

import torch
from . import _dtypes
from ._ndarray import can_cast, ndarray, asarray
from ._detail import _util

def cast_and_broadcast(arrays, out, casting):
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
        return tuple(arr.get() for arr in arrays)
    else:
        if not isinstance(out, ndarray):
            raise TypeError("Return arrays must be of ArrayType")

        tensors = []
        for arr in arrays:
            # check dtypes of x and out
            if not can_cast(arr.dtype, out.dtype, casting=casting):
                raise TypeError(f"Cannot cast array data from {arr.dtype} to"
                                 " {out_dtype} according to the rule '{casting}'")
            tensor = arr.get()

            # `out` broadcasts `arr`
            if arr.shape != out.shape:
                tensor = torch.broadcast_to(tensor, out.shape)

            # cast arr if needed
            if arr.dtype != out.dtype:
                tensor = tensor.to(_dtypes.torch_dtype_from(out.dtype))

            tensors.append(tensor)

    return tuple(tensors)



def result_or_out(result_tensor, out_array=None):
    """A helper for returns with out= argument."""
    if out_array is not None:
        if result_tensor.shape != out_array.shape:
            raise ValueError
        out_tensor = out_array.get()
        out_tensor.copy_(result_tensor)
        return out_array
    else:
        return asarray(result_tensor)


def to_tensors_lax(*inputs):
    """Convert all ndarrays from `inputs` to tensors. (other things are intact)
    """
    return tuple([value.get() if isinstance(value, ndarray) else value
            for value in inputs])


def to_tensors(*inputs):
    """Convert all array_likes from `inputs` to tensors.
    """
    return tuple(asarray(value).get() for value in inputs)

