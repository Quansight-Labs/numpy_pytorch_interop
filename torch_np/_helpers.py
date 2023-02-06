import torch

from . import _dtypes
from ._detail import _util
from ._ndarray import asarray, ndarray


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

        tensors = _util.cast_and_broadcast(
            tensors, out.dtype.type.torch_dtype, out.shape, casting
        )

    return tuple(tensors)


def result_or_out(result_tensor, out_array=None, promote_scalar=False):
    """A helper for returns with out= argument.

    If `promote_scalar is True`, then:
        if result_tensor.numel() == 1 and out is zero-dimensional,
            result_tensor is placed into the out array.
    This weirdness is used e.g. in `np.percentile`
    """
    if out_array is not None:
        if not isinstance(out_array, ndarray):
            raise TypeError("Return arrays must be of ArrayType")
        if result_tensor.shape != out_array.shape:
            can_fit = result_tensor.numel() == 1 and out_array.ndim == 0
            if promote_scalar and can_fit:
                result_tensor = result_tensor.squeeze()
            else:
                raise ValueError(
                    f"Bad size of the out array: out.shape = {out_array.shape}"
                    f" while result.shape = {result_tensor.shape}."
                )
        out_tensor = out_array.get()
        out_tensor.copy_(result_tensor)
        return out_array
    else:
        return asarray(result_tensor)


def ndarrays_to_tensors(*inputs):
    """Convert all ndarrays from `inputs` to tensors. (other things are intact)"""
    return tuple(
        [value.get() if isinstance(value, ndarray) else value for value in inputs]
    )


def to_tensors(*inputs):
    """Convert all array_likes from `inputs` to tensors."""
    return tuple(asarray(value).get() for value in inputs)


def to_tensors_or_none(*inputs):
    """Convert all array_likes from `inputs` to tensors. Nones pass through"""
    return tuple(None if value is None else asarray(value).get() for value in inputs)


def _outer(x, y):
    x_tensor, y_tensor = to_tensors(x, y)
    result = torch.outer(x_tensor, y_tensor)
    return asarray(result)
