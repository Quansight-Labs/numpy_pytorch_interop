"""Assorted utilities, which do not need anything other then torch and stdlib.
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
    if not (-ndim <= ax < ndim):
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
            raise ValueError("repeated axis in `{}` argument".format(argname))
        else:
            raise ValueError("repeated axis")
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
    target_dtype : torch dtype object, optional
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
    can_cast = _scalar_types._can_cast_impl

    for tensor in tensors:
        if not can_cast(tensor.dtype, target_dtype, casting=casting):
            raise TypeError(f"Cannot cast array data from {tensor.dtype} to"
                            f" {target_dtype} according to the rule '{casting}'")

        # cast if needed
        if tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)
        cast_tensors.append(tensor)

    return tuple(cast_tensors)


def cast_and_broadcast(tensors, out_param, casting):
    """
    Parameters
    ----------
    tensors : iterable
	tuple or list of torch.Tensors to broadcast/typecast
    target_dtype : a torch.dtype object
        The torch dtype to cast all tensors to
    target_shape : tuple
        The tensor shape to broadcast all `tensors` to
    casting : str
        The casting mode, see `np.can_cast`

    Returns
    -------
    a tuple of torch.Tensors with dtype being the PyTorch counterpart
    of the `target_dtype` and `target_shape`
    """
    if out_param is None:
        return tensors

    target_dtype, target_shape = out_param

    can_cast = _scalar_types._can_cast_impl

    processed_tensors = []
    for tensor in tensors:
        # check dtypes of x and out
        if not can_cast(tensor.dtype, target_dtype, casting=casting):
            raise TypeError(f"Cannot cast array data from {tensor.dtype} to"
                            f" {target_dtype} according to the rule '{casting}'")

        # cast arr if needed
        if tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)

        # `out` broadcasts `tensor`
        if tensor.shape != target_shape:
            tensor = torch.broadcast_to(tensor, target_shape)

        processed_tensors.append(tensor)

    return tuple(processed_tensors)


def axis_keepdims(func, tensor, axis, keepdims, *args, **kwds):
    """Generically handle axis and keepdims arguments in reductions."""
    if axis is not None:
        if not isinstance(axis, (list, tuple)):
            axis = (axis,)
        axis = normalize_axis_tuple(axis, tensor.ndim)

    if axis == ():
        newshape = expand_shape(tensor.shape, axis=0)
        tensor = tensor.reshape(newshape)
        axis = (0,)

    result = func(tensor, axis=axis, *args, **kwds)

    if keepdims:
        result = apply_keepdims(result, axis, tensor.ndim)

    return result


def _coerce_to_tensor(obj, dtype=None, copy=False, ndmin=0):
    """The core logic of the array(...) function.

    Parameters
    ----------
    obj : tensor_like
        The thing to coerce
    dtype : torch.dtype object or None
        Coerce to this torch dtype
    copy : bool
        Copy or not

    Returns
    -------
    tensor : torch.Tensor
        a tensor object with requested dtype, ndim and copy semantics.

    Notes
    -----
    This is almost a "tensor_like" coersion function. Does not handle wrapper
    ndarrays (those should be handled in the ndarray-aware layer prior to
    invoking this function).
    """
    if isinstance(obj, torch.Tensor):
        tensor = obj
        base = None
    else:
        tensor = torch.as_tensor(obj)
        base = None

        # At this point, `tensor.dtype` is the pytorch default. Our default may
        # differ, so need to typecast. However, we cannot just do `tensor.to`,
        # because if our desired dtype is wider then pytorch's, `tensor`
        # may have lost precision:

        # int(torch.as_tensor(1e12)) - 1e12 equals -4096 (try it!)

        # Therefore, we treat `tensor.dtype` as a hint, and convert the
        # original object *again*, this time with an explicit dtype.
        sctype = _scalar_types.get_default_type_for(_scalar_types.sctype_from_torch_dtype(tensor.dtype))
        torch_dtype = sctype.torch_dtype

        tensor = torch.as_tensor(obj, dtype=torch_dtype)

    # type cast if requested
    if dtype is not None:
        tensor = tensor.to(dtype)

    # adjust ndim if needed
    ndim_extra = ndmin - tensor.ndim
    if ndim_extra > 0:
        tensor = tensor.view((1,)*ndim_extra + tensor.shape)

    # copy if requested
    if copy:
        tensor = tensor.clone()

    return tensor
