"""Assorted utilities, which do not need anything other then torch and stdlib.
"""

import operator

import torch

from . import _dtypes_impl


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


def cast_if_needed(tensor, dtype):
    # NB: no casting if dtype=None
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor


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
        shape = (1,) * ndim
        tensor = tensor.expand(shape).contiguous()
    else:
        shape = expand_shape(tensor.shape, axis)
        tensor = tensor.reshape(shape)
    return tensor


def axis_none_flatten(*tensors, axis=None):
    """Flatten the arrays if axis is None."""
    if axis is None:
        tensors = tuple(ar.flatten() for ar in tensors)
        return tensors, 0
    else:
        return tensors, axis


def typecast_tensor(t, target_dtype, casting):
    """Dtype-cast tensor to target_dtype.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to typecast
    target_dtype : torch dtype object
        The array dtype to cast all tensors to
    casting : str
        The casting mode, see `np.can_cast`

    Returns
    -------
    a torch.Tensors of the `target_dtype` dtype

    Raises
    ------
    ValueError
        if the argument cannot be cast according to the `casting` rule

    """
    # check if we can dtype-cast the argument
    can_cast = _dtypes_impl.can_cast_impl

    if not can_cast(t.dtype, target_dtype, casting=casting):
        raise TypeError(
            f"Cannot cast array data from {t.dtype} to"
            f" {target_dtype} according to the rule '{casting}'"
        ) 
    return cast_if_needed(t, target_dtype)


def typecast_tensors(tensors, target_dtype, casting):
    """Dtype-cast tensors to target_dtype.

    Apply `typecast_tensor` to each element of the `tensors` iterable.
    """
    return tuple(typecast_tensor(t, target_dtype, casting) for t in tensors)


def _dtype_for_scalar(py_type):
    return {bool: torch.bool, int: torch.int64, float: torch.float64}[py_type]


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
    obj_type = type(obj)

    if obj_type is torch.Tensor:
        tensor = obj
        is_weakly_typed = getattr(obj, "is_weakly_typed", False)

    elif obj_type in (bool, int, float):
        # Make python scalars weakly typed.
        is_weakly_typed = True
        dtype = dtype or _dtype_for_scalar(obj_type)
        tensor = torch.as_tensor(obj, dtype=dtype)        

    else:
        is_weakly_typed = False
        tensor = torch.as_tensor(obj)

        # tensor.dtype is the pytorch default, typically float32. If obj's elements
        # are not exactly representable in float32, we've lost precision:
        # >>> torch.as_tensor(1e12).item() - 1e12
        # -4096.0

        # Therefore, we treat `tensor.dtype` as a hint, and convert the
        # original object *again*, this time with an explicit dtype.
        torch_dtype = _dtypes_impl.get_default_dtype_for(tensor.dtype)
        tensor = torch.as_tensor(obj, dtype=torch_dtype)

    # type cast if requested
    tensor = cast_if_needed(tensor, dtype)

    # adjust ndim if needed
    ndim_extra = ndmin - tensor.ndim
    if ndim_extra > 0:
        tensor = tensor.view((1,) * ndim_extra + tensor.shape)

    # copy if requested
    if copy:
        tensor = tensor.clone()

    # Attach the flag *to the tensor* (will be used after normalizations)
    tensor.is_weakly_typed = is_weakly_typed
    return tensor
