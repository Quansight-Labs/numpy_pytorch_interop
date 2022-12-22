import torch
from . import _dtypes
from ._ndarray import can_cast, ndarray

def check_bcast(arrays, out, casting):
    """Check that arrays can be cast to out.dtype and broadcast if needed.

    Note that `out` broadcasts `arrays`, but not vice versa.
    """
    if out is not None:
        if not isinstance(out, ndarray):
            raise TypeError("Return arrays must be of ArrayType")

        for arr in arrays:
            # check dtypes of x and out
            if not can_cast(arr.dtype, out.dtype, casting=casting):
                raise TypeError(f"Cannot cast array data from {arr.dtype} to"
                                 " {out_dtype} according to the rule '{casting}'")
            tensor = arr.get()

            # `out` broadcasts `x`
            if arr.shape != out.shape:
                tensor = torch.broadcast_to(tensor, out.shape)

            # cast x if needed
            if arr.dtype != out.dtype:
                tensor = tensor.to(_dtypes.torch_dtype_from(out.dtype))

    return arrays
