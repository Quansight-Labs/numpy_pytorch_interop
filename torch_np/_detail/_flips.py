"""Implementations of flip-based routines and related animals. 
"""

import torch

from . import _util


def flip(m_tensor, axis=None):
    # XXX: semantic difference: np.flip returns a view, torch.flip copies
    if axis is None:
        axis = tuple(range(m_tensor.ndim))
    else:
        axis = _util.normalize_axis_tuple(axis, m_tensor.ndim)
    return torch.flip(m_tensor, axis)


def flipud(m_tensor):
    return torch.flipud(m_tensor)


def fliplr(m_tensor):
    return torch.fliplr(m_tensor)


def rot90(m_tensor, k=1, axes=(0, 1)):
    axes = _util.normalize_axis_tuple(axes, m_tensor.ndim)
    return torch.rot90(m_tensor, k, axes)


def swapaxes(tensor, axis1, axis2):
    return torch.swapaxes(tensor, axis1, axis2)


# Straight vendor from:
# https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1259
#
# Also note this function in NumPy is mostly retained for backwards compat
# (https://stackoverflow.com/questions/29891583/reason-why-numpy-rollaxis-is-so-confusing)
# so let's not touch it unless hard pressed.
def rollaxis(tensor, axis, start=0):
    n = tensor.ndim
    axis = _util.normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise _util.AxisError(msg % ("start", -n, "start", n + 1, start))
    if axis < start:
        # it's been removed
        start -= 1
    if axis == start:
        # numpy returns a view, here we try returning the tensor itself
        # return tensor[...]
        return tensor
    axes = list(range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return tensor.view(axes)
