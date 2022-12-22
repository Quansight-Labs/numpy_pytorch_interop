import torch
from . import _dtypes
from ._ndarray import can_cast, ndarray, asarray

def check_bcast(arrays, out, casting):
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



def check_dtype(arrays, out_dtype, casting):
    """Dtype-cast arrays to dtype.
    """
    # check if we can dtype-cast all arguments
    tensors = []
    for arr in arrays:
        if not can_cast(arr.dtype, out_dtype, casting=casting):
            raise TypeError(f"Cannot cast array data from {arr.dtype} to"
                             " {out_dtype} according to the rule '{casting}'")
        tensor = arr.get()

        # cast arr if needed
        if arr.dtype != out_dtype:
            tensor = tensor.to(_dtypes.torch_dtype_from(out_dtype))

        tensors.append(tensor)

    return tuple(tensors)



def axis_none_ravel(*arrays, axis=None):
    """Ravel the arrays if axis is none."""
    if axis is None:
        arrays = tuple(ar.ravel() for ar in arrays)
        return arrays, 0
    else:
        return arrays, axis


def result_or_out(result, out=None):
    """A helper for returns with out= argument."""
    if out is not None:
        if result.shape != out.shape:
            raise ValueError
        out_tensor = out.get()
        out_tensor.copy_(result)
        return out
    else:
        return asarray(result)

