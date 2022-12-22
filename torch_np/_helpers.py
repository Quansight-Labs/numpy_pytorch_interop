import torch
from . import _dtypes
from ._ndarray import can_cast, ndarray

def check_bcast(arrays, out, casting):
    """Cast dtypes of arrays to out.dtype and broadcast if needed.

    Returns
    -------
    tensors : tuple of Tensors
        Each tensor is dtype-cast and broadcast agains `out`, as needed

    Notes
    -----
    1. `arrays` are modified in place.
    2. The `out` arrays broadcasts `arrays`, but not vice versa.
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
