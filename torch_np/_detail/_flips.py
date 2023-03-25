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
