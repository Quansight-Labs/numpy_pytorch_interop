from typing import Sequence

import torch

from ._detail import _dtypes_impl, _util
from ._normalizations import ArrayLike, normalizer


class LinAlgError(Exception):
    pass


def _atleast_float_1(a):
    if not (a.dtype.is_floating_point or a.dtype.is_complex):
        a = a.to(_dtypes_impl.default_float_dtype)
    return a


def _atleast_float_2(a, b):
    dtyp = _dtypes_impl.result_type_impl((a.dtype, b.dtype))
    if not (dtyp.is_floating_point or dtyp.is_complex):
        dtyp = _dtypes_impl.default_float_dtype

    a = _util.cast_if_needed(a, dtyp)
    b = _util.cast_if_needed(b, dtyp)
    return a, b


def _prod(iterable):
    p = 1.0
    for x in iterable:
        p *= x
    return p


# ### Matrix and vector products ###


@normalizer
def matrix_power(a: ArrayLike, n):
    a = _atleat_float_1(a)
    return torch.linalg.matrix_power(a, n)


@normalizer
def multi_dot(inputs: Sequence[ArrayLike], *, out=None):
    return torch.linalg.multi_dot(inputs)


# ### Solving equations and inverting matrices ###


@normalizer
def solve(a: ArrayLike, b: ArrayLike):
    a, b = _atleast_float_2(a, b)
    return torch.linalg.solve(a, b)


@normalizer
def lstsq(a: ArrayLike, b: ArrayLike, rcond=None):
    a, b = _atleast_float_2(a, b)
    # NumPy is using gelsd: https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/umath_linalg.cpp#L3991
    return torch.linalg.lstsq(a, b, rcond=rcond, driver="gelsd")


@normalizer
def inv(a: ArrayLike):
    a = _atleast_float_1(a)
    try:
        result = torch.linalg.inv(a)
    except torch._C._LinAlgError as e:
        raise LinAlgError(*e.args)
    return result


@normalizer
def pinv(a: ArrayLike, rcond=1e-15, hermitian=False):
    a = _atleast_float_1(a)
    return torch.linalg.pinv(a, rtol=rcond, hermitian=hermitian)


@normalizer
def tensorsolve(a: ArrayLike, b: ArrayLike, axes=None):
    a, b = _atleast_float_2(a, b)
    return torch.linalg.tensorsolve(a, b, dims=axes)


@normalizer
def tensorinv(a: ArrayLike, ind=2):
    a = _atleast_float_1(a)
    return torch.linalg.tensorinv(a, ind=ind)


# ### Norms and other numbers ###


@normalizer
def det(a: ArrayLike):
    a = _atleast_float_1(a)
    return torch.linalg.det(a)


@normalizer
def slogdet(a: ArrayLike):
    a = _atleast_float_1(a)
    return torch.linalg.slogdet(a)


@normalizer
def cond(x: ArrayLike, p=None):
    x = _atleast_float_1(x)

    # check if empty
    # cf: https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1744
    if x.numel() == 0 and _prod(x.shape[-2:]) == 0:
        raise LinAlgError("cond is not defined on empty arrays")

    result = torch.linalg.cond(x, p=p)

    # Convert nans to infs unless the original array had nan entries
    # XXX: NumPy does this: https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1744
    # Do we want it? .any() synchronizes
    """
    nan_mask = torch.isnan(result)
    if nan_mask.any():
        nan_mask &= ~torch.isnan(x).any(axis=(-2, -1))  # XXX: any() w/ tuple axes
        if result.ndim > 0:
            result[nan_mask] = torch.inf
        elif nan_mask:
            result[()] = torch.inf
    """

    return result


@normalizer
def matrix_rank(a: ArrayLike, tol=None, hermitian=False):
    a = _atleast_float_1(a)

    if a.ndim < 2:
        return int(not (a == 0).all())

    if tol is None:
        # follow https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1885
        atol = 0
        rtol = max(a.shape[-2:]) * torch.finfo(a.dtype).eps
    else:
        atol, rtol = tol, 0
    return torch.linalg.matrix_rank(a, atol=atol, rtol=rtol, hermitian=hermitian)


@normalizer
def norm(x: ArrayLike, ord=None, axis=None, keepdims=False):
    x = _atleast_float_1(x)
    result = torch.linalg.norm(x, ord=ord, dim=axis)
    if keepdims:
        result = _util.apply_keepdims(result, axis, x.ndim)
    return result


# ### Decompositions ###


@normalizer
def cholesky(a: ArrayLike):
    a = _atleast_float_1(a)
    return torch.linalg.cholesky(a)


@normalizer
def qr(a: ArrayLike, mode="reduced"):
    a = _atleast_float_1(a)
    result = torch.linalg.qr(a, mode=mode)
    if mode == "r":
        # match NumPy
        result = result.R
    return result


@normalizer
def svd(a: ArrayLike, full_matrices=True, compute_uv=True, hermitian=False):
    a = _atleast_float_1(a)
    # NB: ignore the hermitian= argument (no pytorch equivalent)
    result = torch.linalg.svd(a, full_matrices=full_matrices)
    if not compute_uv:
        result = result.S
    return result


# ### Eigenvalues and eigenvectors ###


@normalizer
def eig(a: ArrayLike):
    a = _atleast_float_1(a)
    w, vt = torch.linalg.eig(a)

    if not a.is_complex():
        if w.is_complex() and (w.imag == 0).all():
            w = w.real
            vt = vt.real
    return w, vt


@normalizer
def eigh(a: ArrayLike, UPLO="L"):
    a = _atleast_float_1(a)
    return torch.linalg.eigh(a, UPLO=UPLO)


@normalizer
def eigvals(a: ArrayLike):
    a = _atleast_float_1(a)
    result = torch.linalg.eigvals(a)
    if not a.is_complex():
        if result.is_complex() and (result.imag == 0).all():
            result = result.real
    return result


@normalizer
def eigvalsh(a: ArrayLike, UPLO="L"):
    a = _atleast_float_1(a)
    return torch.linalg.eigvalsh(a, UPLO=UPLO)
