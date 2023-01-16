import operator


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
            raise ValueError("repeated axis in `{}` argument".format(argname))
        else:
            raise ValueError("repeated axis")
    return axis


def expand_shape(arr_shape, axis):
    # taken from numpy 1.23.x, expand_dims function
    if type(axis) not in (list, tuple):
        axis = (axis,)
    out_ndim = len(axis) + len(arr_shape)
    axis = normalize_axis_tuple(axis, out_ndim)
    shape_it = iter(arr_shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
    return shape
