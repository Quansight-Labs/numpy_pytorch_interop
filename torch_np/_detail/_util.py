"""Assorted utilities, which do not need anything other then torch and stdlib.

In particular, things here cannot import from _ndarray.py, cannot use the
name `ndarray`. Depending on a duck-typed "array" argument is allowed though.
The following methods are expected to exist:

- `arr.get()` to return a torch.Tensor
- `arr.ravel()` -- is only used by `axis_none_ravel`
"""

import operator

import torch

from . import _scalar_types

# https://github.com/numpy/numpy/blob/v1.23.0/numpy/distutils/misc_util.py#L497-L504
def is_sequence(seq):
    if isinstance(seq, str):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True


def subok_not_ok(like=None, subok=False):
    if like is not None:
        raise ValueError("like=... parameter is not supported.")
    if subok:
        raise ValueError("subok parameter is not supported.")


class AxisError(ValueError, IndexError):
    pass


class UFuncTypeError(TypeError, RuntimeError):
    pass


# a replica of the version in ./numpy/numpy/core/src/multiarray/common.h
def normalize_axis_index(ax, ndim, argname=None):
    if ax < -ndim or ax >= ndim:
        raise AxisError(f"axis {ax} is out of bounds for array of dimension {ndim}")
    if ax < 0:
        ax += ndim
    return ax


# from https://github.com/numpy/numpy/blob/main/numpy/core/numeric.py#L1378
def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    """
    Normalizes an axis argument into a tuple of non-negative integer axes.
    This handles shorthands such as ``1`` and converts them to ``(1,)``,
    as well as performing the handling of negative indices covered by
    `normalize_axis_index`.
    By default, this forbids axes from being specified multiple times.
    Used internally by multi-axis-checking logic.
    .. versionadded:: 1.13.0
    Parameters
    ----------
    axis : int, iterable of int
        The un-normalized index or indices of the axis.
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against.
    argname : str, optional
        A prefix to put before the error message, typically the name of the
        argument.
    allow_duplicate : bool, optional
        If False, the default, disallow an axis from being specified twice.
    Returns
    -------
    normalized_axes : tuple of int
        The normalized axis index, such that `0 <= normalized_axis < ndim`
    Raises
    ------
    AxisError
        If any axis provided is out of range
    ValueError
        If an axis is repeated
    See also
    --------
    normalize_axis_index : normalizing a single scalar axis
    """
    # Optimization to speed-up the most common cases.
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    # Going via an iterator directly is slower than via list comprehension.
    axis = tuple([normalize_axis_index(ax, ndim, argname) for ax in axis])
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError('repeated axis in `{}` argument'.format(argname))
        else:
            raise ValueError('repeated axis')
    return axis


def allow_only_single_axis(axis):
    if axis is None:
        return axis
    if len(axis) != 1:
        raise NotImplementedError("does not handle tuple axis")
    return axis[0]


def expand_shape(arr_shape, axis):
    # taken from numpy 1.23.x, expand_dims function
    if type(axis) not in (list, tuple):
        axis = (axis,)
    out_ndim = len(axis) + len(arr_shape)
    axis = normalize_axis_tuple(axis, out_ndim)
    shape_it = iter(arr_shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
    return shape


def apply_keepdims(tensor, axis, ndim):
    if axis is None:
        # tensor was a scalar
        tensor = torch.full((1,)*ndim, fill_value=tensor, dtype=tensor.dtype)
    else:
        shape = expand_shape(tensor.shape, axis)
        tensor = tensor.reshape(shape)
    return tensor


def axis_none_ravel(*tensors, axis=None):
    """Ravel the arrays if axis is none."""
    # XXX: is only used at `concatenate`. Inline unless reused more widely
    if axis is None:
        tensors = tuple(ar.ravel() for ar in tensors)
        return tensors, 0
    else:
        return tensors, axis


def cast_dont_broadcast(tensors, target_dtype, casting):
    """Dtype-cast tensors to target_dtype.

    Parameters
    ----------
    tensors : iterable
	tuple or list of torch.Tensors to typecast
    target_dtype : DType object
        The array dtype to cast all tensors to
    casting : str
        The casting mode, see `np.can_cast`

    Returns
    -------
    a tuple of torch.Tensors with dtype being the PyTorch counterpart
    of the `target_dtype`
    """
    # check if we can dtype-cast all arguments
    cast_tensors = []
    target_dtype_t = target_dtype.type.torch_dtype
    can_cast = _scalar_types._can_cast_impl

    for tensor in tensors:
        if not can_cast(tensor.dtype, target_dtype_t, casting=casting):
            raise TypeError(f"Cannot cast array data from {tensor.dtype} to"
                            f" {target_dtype} according to the rule '{casting}'")

        # cast if needed
        if tensor.dtype != target_dtype_t:
            tensor = tensor.to(target_dtype_t)
        cast_tensors.append(tensor)

    return tuple(cast_tensors)
