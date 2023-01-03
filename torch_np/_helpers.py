import operator

import torch
from . import _dtypes
from ._ndarray import can_cast, ndarray, asarray
from . import _util

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



def cast_dont_broadcast(arrays, out_dtype, casting):
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


def apply_keepdims(tensor, axis, ndim):
    if axis is None:
        # tensor was a scalar
        tensor = torch.full((1,)*ndim, fill_value=tensor)
    else:
        shape = _util.expand_shape(tensor, axis)
        tensor = tensor.reshape(shape)
    return tensor


def standardize_axis_arg(axis, ndim):
    """Return axis as either None or a tuple of normalized axes."""
    if isinstance(axis, ndarray):
        axis = operator.index(axis)

    if axis is not None:
        if type(axis) not in (list, tuple):
            axis = (axis,)
        axis = _util.normalize_axis_tuple(axis, ndim)
    return axis


def allow_only_single_axis(axis):
    if axis is None:
        return axis
    if len(axis) != 1:
        raise NotImplementedError("does not handle tuple axis")
    return axis[0]


def to_tensors(*inputs):
    """Convert all ndarrays from `inputs` to tensors."""
    return tuple([value.get() if isinstance(value, ndarray) else value
            for value in inputs])
