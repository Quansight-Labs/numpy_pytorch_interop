"""A thin pytorch / numpy compat layer.

Things imported from here have numpy-compatible signatures but operate on
pytorch tensors.
"""

from typing import Optional, Sequence

import torch

from . import _decorators
from . import _detail as _impl
from . import _dtypes, _funcs, _helpers
from ._detail import _dtypes_impl, _util
from ._ndarray import asarray
from ._normalizations import ArrayLike, DTypeLike, NDArray, SubokLike, normalizer

NoValue = _util.NoValue


# Things to decide on (punt for now)
#
# 1. Q: What are the return types of wrapper functions: plain torch.Tensors or
#       wrapper ndarrays.
#    A: Wrapper ndarrays.
#
# 2. Q: Default dtypes: numpy defaults to float64, pytorch defaults to float32
#    A: Stick to pytorch defaults?
#    NB: numpy recommends `dtype=float`?
#
# 3. Q: Masked arrays. Record, structured arrays.
#    A: Ignore for now
#
# 4. Q: What are the defaults for pytorch-specific args not present in numpy signatures?
#       device=..., requires_grad=... etc
#    A: ignore, keep whatever they are from inputs; test w/various combinations
#
# 5. Q: What is the useful action for numpy-specific arguments? e.g. like=...
#    A: like=... and subok=True both raise ValueErrors.
#       initial=... can be useful though, punt on
#       where=...   punt on for now


# TODO
# 1. Mapping of the numpy layout ('C', 'K' etc) to torch layout / memory_format.
# 2. np.dtype <-> torch.dtype
# 3. numpy type casting rules (to be cleaned up in numpy: follow old or new)
#
# 4. wrap/unwrap/wrap patterns:
#   - inputs are scalars, output is an array
#   - two-arg functions (second may be None)
#   - first arg is a sequence/tuple (_stack familty, concatenate, atleast_Nd etc)
#   - optional out arg


###### array creation routines


@normalizer
def copy(a: ArrayLike, order="K", subok: SubokLike = False):
    if order != "K":
        raise NotImplementedError
    tensor = a.clone()
    return tensor


@normalizer
def atleast_1d(*arys: ArrayLike):
    res = torch.atleast_1d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


@normalizer
def atleast_2d(*arys: ArrayLike):
    res = torch.atleast_2d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


@normalizer
def atleast_3d(*arys: ArrayLike):
    res = torch.atleast_3d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def _concat_check(tup, dtype, out):
    """Check inputs in concatenate et al."""
    if tup == ():
        # XXX: RuntimeError in torch, ValueError in numpy
        raise ValueError("need at least one array to concatenate")

    if out is not None:
        if dtype is not None:
            # mimic numpy
            raise TypeError(
                "concatenate() only takes `out` or `dtype` as an "
                "argument, but both were provided."
            )


@normalizer
def concatenate(
    ar_tuple: Sequence[ArrayLike],
    axis=0,
    out: Optional[NDArray] = None,
    dtype: DTypeLike = None,
    casting="same_kind",
):
    _concat_check(ar_tuple, dtype, out=out)
    result = _impl.concatenate(ar_tuple, axis, out, dtype, casting)
    return _helpers.result_or_out(result, out)


@normalizer
def vstack(tup: Sequence[ArrayLike], *, dtype: DTypeLike = None, casting="same_kind"):
    _concat_check(tup, dtype, out=None)
    result = _impl.vstack(tup, dtype=dtype, casting=casting)
    return result


row_stack = vstack


@normalizer
def hstack(tup: Sequence[ArrayLike], *, dtype: DTypeLike = None, casting="same_kind"):
    _concat_check(tup, dtype, out=None)
    result = _impl.hstack(tup, dtype=dtype, casting=casting)
    return result


@normalizer
def dstack(tup: Sequence[ArrayLike], *, dtype: DTypeLike = None, casting="same_kind"):
    # XXX: in numpy 1.24 dstack does not have dtype and casting keywords
    # but {h,v}stack do.  Hence add them here for consistency.
    result = _impl.dstack(tup, dtype=dtype, casting=casting)
    return result


@normalizer
def column_stack(
    tup: Sequence[ArrayLike], *, dtype: DTypeLike = None, casting="same_kind"
):
    # XXX: in numpy 1.24 column_stack does not have dtype and casting keywords
    # but row_stack does. (because row_stack is an alias for vstack, really).
    # Hence add these keywords here for consistency.
    _concat_check(tup, dtype, out=None)
    result = _impl.column_stack(tup, dtype=dtype, casting=casting)
    return result


@normalizer
def stack(
    arrays: Sequence[ArrayLike],
    axis=0,
    out: Optional[NDArray] = None,
    *,
    dtype: DTypeLike = None,
    casting="same_kind",
):
    _concat_check(arrays, dtype, out=out)
    result = _impl.stack(arrays, axis=axis, out=out, dtype=dtype, casting=casting)
    return _helpers.result_or_out(result, out)


@normalizer
def array_split(ary: ArrayLike, indices_or_sections, axis=0):
    result = _impl.split_helper(ary, indices_or_sections, axis)
    return result


@normalizer
def split(ary: ArrayLike, indices_or_sections, axis=0):
    result = _impl.split_helper(ary, indices_or_sections, axis, strict=True)
    return result


@normalizer
def hsplit(ary: ArrayLike, indices_or_sections):
    result = _impl.hsplit(ary, indices_or_sections)
    return result


@normalizer
def vsplit(ary: ArrayLike, indices_or_sections):
    result = _impl.vsplit(ary, indices_or_sections)
    return result


@normalizer
def dsplit(ary: ArrayLike, indices_or_sections):
    result = _impl.dsplit(ary, indices_or_sections)
    return result


@normalizer
def kron(a: ArrayLike, b: ArrayLike):
    result = torch.kron(a, b)
    return result


@normalizer
def vander(x: ArrayLike, N=None, increasing=False):
    result = torch.vander(x, N, increasing)
    return result


@normalizer
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num=50,
    endpoint=True,
    retstep=False,
    dtype: DTypeLike = None,
    axis=0,
):
    result = _impl.linspace(start, stop, num, endpoint, retstep, dtype, axis)
    return result


@normalizer
def geomspace(
    start: ArrayLike,
    stop: ArrayLike,
    num=50,
    endpoint=True,
    dtype: DTypeLike = None,
    axis=0,
):
    result = _impl.geomspace(start, stop, num, endpoint, dtype, axis)
    return result


@normalizer
def logspace(
    start, stop, num=50, endpoint=True, base=10.0, dtype: DTypeLike = None, axis=0
):
    result = _impl.logspace(start, stop, num, endpoint, base, dtype, axis)
    return result


@normalizer
def arange(
    start: Optional[ArrayLike] = None,
    stop: Optional[ArrayLike] = None,
    step: Optional[ArrayLike] = 1,
    dtype: DTypeLike = None,
    *,
    like: SubokLike = None,
):
    result = _impl.arange(start, stop, step, dtype=dtype)
    return result


@normalizer
def empty(shape, dtype: DTypeLike = float, order="C", *, like: SubokLike = None):
    result = _impl.empty(shape, dtype, order)
    return result


# NB: *_like functions deliberately deviate from numpy: it has subok=True
# as the default; we set subok=False and raise on anything else.
@normalizer
def empty_like(
    prototype: ArrayLike,
    dtype: DTypeLike = None,
    order="K",
    subok: SubokLike = False,
    shape=None,
):
    result = _impl.empty_like(prototype, dtype=dtype, shape=shape)
    return result


@normalizer
def full(
    shape,
    fill_value: ArrayLike,
    dtype: DTypeLike = None,
    order="C",
    *,
    like: SubokLike = None,
):
    result = _impl.full(shape, fill_value, dtype=dtype, order=order)
    return result


@normalizer
def full_like(
    a: ArrayLike,
    fill_value,
    dtype: DTypeLike = None,
    order="K",
    subok: SubokLike = False,
    shape=None,
):
    result = _impl.full_like(a, fill_value, dtype=dtype, shape=shape, order=order)
    return result


@normalizer
def ones(shape, dtype: DTypeLike = None, order="C", *, like: SubokLike = None):
    result = _impl.ones(shape, dtype, order)
    return result


@normalizer
def ones_like(
    a: ArrayLike,
    dtype: DTypeLike = None,
    order="K",
    subok: SubokLike = False,
    shape=None,
):
    result = _impl.ones_like(a, dtype=dtype, shape=shape, order=order)
    return result


@normalizer
def zeros(shape, dtype: DTypeLike = None, order="C", *, like: SubokLike = None):
    result = _impl.zeros(shape, dtype, order)
    return result


@normalizer
def zeros_like(
    a: ArrayLike,
    dtype: DTypeLike = None,
    order="K",
    subok: SubokLike = False,
    shape=None,
):
    result = _impl.zeros_like(a, dtype=dtype, shape=shape, order=order)
    return result


###### misc/unordered


def _xy_helper_corrcoef(x_tensor, y_tensor=None, rowvar=True):
    """Prepate inputs for cov and corrcoef."""

    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/function_base.py#L2636
    if y_tensor is not None:
        # make sure x and y are at least 2D
        ndim_extra = 2 - x_tensor.ndim
        if ndim_extra > 0:
            x_tensor = x_tensor.view((1,) * ndim_extra + x_tensor.shape)
        if not rowvar and x_tensor.shape[0] != 1:
            x_tensor = x_tensor.mT
        x_tensor = x_tensor.clone()

        ndim_extra = 2 - y_tensor.ndim
        if ndim_extra > 0:
            y_tensor = y_tensor.view((1,) * ndim_extra + y_tensor.shape)
        if not rowvar and y_tensor.shape[0] != 1:
            y_tensor = y_tensor.mT
        y_tensor = y_tensor.clone()

        x_tensor = _impl.concatenate((x_tensor, y_tensor), axis=0)

    return x_tensor


@normalizer
def corrcoef(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    rowvar=True,
    bias=NoValue,
    ddof=NoValue,
    *,
    dtype: DTypeLike = None,
):
    if bias is not None or ddof is not None:
        # deprecated in NumPy
        raise NotImplementedError
    tensor = _xy_helper_corrcoef(x, y, rowvar)
    result = _impl.corrcoef(tensor, dtype=dtype)
    return result


@normalizer
def cov(
    m: ArrayLike,
    y: Optional[ArrayLike] = None,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights: Optional[ArrayLike] = None,
    aweights: Optional[ArrayLike] = None,
    *,
    dtype: DTypeLike = None,
):
    m = _xy_helper_corrcoef(m, y, rowvar)
    result = _impl.cov(m, bias, ddof, fweights, aweights, dtype=dtype)
    return result


@normalizer
def bincount(x: ArrayLike, /, weights: Optional[ArrayLike] = None, minlength=0):
    result = _impl.bincount(x, weights, minlength)
    return result


@normalizer
def where(
    condition: ArrayLike,
    x: Optional[ArrayLike] = None,
    y: Optional[ArrayLike] = None,
    /,
):
    result = _impl.where(condition, x, y)
    return result


###### module-level queries of object properties


@normalizer
def ndim(a: ArrayLike):
    return a.ndim


@normalizer
def shape(a: ArrayLike):
    return tuple(a.shape)


@normalizer
def size(a: ArrayLike, axis=None):
    if axis is None:
        return a.numel()
    else:
        return a.shape[axis]


###### shape manipulations and indexing


@normalizer
def expand_dims(a: ArrayLike, axis):
    shape = _util.expand_shape(a.shape, axis)
    tensor = a.view(shape)  # never copies
    return tensor


@normalizer
def flip(m: ArrayLike, axis=None):
    result = _impl.flip(m, axis)
    return result


@normalizer
def flipud(m: ArrayLike):
    result = _impl.flipud(m)
    return result


@normalizer
def fliplr(m: ArrayLike):
    result = _impl.fliplr(m)
    return result


@normalizer
def rot90(m: ArrayLike, k=1, axes=(0, 1)):
    result = _impl.rot90(m, k, axes)
    return result


@normalizer
def broadcast_to(array: ArrayLike, shape, subok: SubokLike = False):
    result = torch.broadcast_to(array, size=shape)
    return result


from torch import broadcast_shapes


# YYY: pattern: tuple of arrays as input, tuple of arrays as output; cf nonzero
@normalizer
def broadcast_arrays(*args: ArrayLike, subok: SubokLike = False):
    res = torch.broadcast_tensors(*args)
    return res


def unravel_index(indices, shape, order="C"):
    # cf https://github.com/pytorch/pytorch/pull/66687
    # this version is from
    # https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/3
    if order != "C":
        raise NotImplementedError
    result = []
    for index in indices:
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        result.append(tuple(reversed(out)))
    return result


def ravel_multi_index(multi_index, dims, mode="raise", order="C"):
    # XXX: not available in pytorch, implement
    return sum(idx * dim for idx, dim in zip(multi_index, dims))


@normalizer
def meshgrid(*xi: ArrayLike, copy=True, sparse=False, indexing="xy"):
    output = _impl.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)
    return list(output)  # match numpy, return a list


@normalizer
def indices(dimensions, dtype: DTypeLike = int, sparse=False):
    result = _impl.indices(dimensions, dtype=dtype, sparse=sparse)
    return result


@normalizer
def roll(a: ArrayLike, shift, axis=None):
    result = _impl.roll(a, shift, axis)
    return result


###### tri{l, u} and related
@normalizer
def tril(m: ArrayLike, k=0):
    result = m.tril(k)
    return result


@normalizer
def triu(m: ArrayLike, k=0):
    result = m.triu(k)
    return result


def tril_indices(n, k=0, m=None):
    result = _impl.tril_indices(n, k, m)
    return result


def triu_indices(n, k=0, m=None):
    result = _impl.triu_indices(n, k, m)
    return result


@normalizer
def tril_indices_from(arr: ArrayLike, k=0):
    result = _impl.tril_indices_from(arr, k)
    return result


@normalizer
def triu_indices_from(arr: ArrayLike, k=0):
    result = _impl.triu_indices_from(arr, k)
    return result


@normalizer
def tri(N, M=None, k=0, dtype: DTypeLike = float, *, like: SubokLike = None):
    result = _impl.tri(N, M, k, dtype)
    return result


###### reductions


@normalizer
def average(
    a: ArrayLike,
    axis=None,
    weights: ArrayLike = None,
    returned=False,
    *,
    keepdims=NoValue,
):
    result, wsum = _impl.average(a, axis, weights, returned=returned, keepdims=keepdims)
    if returned:
        return result, wsum
    else:
        return result


@normalizer
def inner(a: ArrayLike, b: ArrayLike, /):
    result = _impl.inner(a, b)
    return result


@normalizer
def outer(a: ArrayLike, b: ArrayLike, out: Optional[NDArray] = None):
    result = torch.outer(a, b)
    return _helpers.result_or_out(result, out)


# ### FIXME: this is a stub


@normalizer
def nanmean(
    a: ArrayLike,
    axis=None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    *,
    where=NoValue,
):
    # XXX: this needs to be rewritten
    if where is not NoValue:
        raise NotImplementedError
    if dtype is None:
        dtype = a.dtype
    if axis is None:
        result = a.nanmean(dtype=dtype)
        if keepdims:
            result = torch.full(a.shape, result, dtype=result.dtype)
    else:
        result = a.nanmean(dtype=dtype, dim=axis, keepdim=bool(keepdims))
    if out is not None:
        out.copy_(result)
    return result


def nanmin():
    raise NotImplementedError


def nanmax():
    raise NotImplementedError


def nanvar():
    raise NotImplementedError


def nanstd():
    raise NotImplementedError


def nanargmin():
    raise NotImplementedError


def nanargmax():
    raise NotImplementedError


def nansum():
    raise NotImplementedError


def nanprod():
    raise NotImplementedError


def nancumsum():
    raise NotImplementedError


def nancumprod():
    raise NotImplementedError


def nanmedian():
    raise NotImplementedError


def nanquantile():
    raise NotImplementedError


def nanpercentile():
    raise NotImplementedError


@normalizer
def diff(
    a: ArrayLike,
    n=1,
    axis=-1,
    prepend: Optional[ArrayLike] = NoValue,
    append: Optional[ArrayLike] = NoValue,
):
    result = _impl.diff(
        a,
        n=n,
        axis=axis,
        prepend_tensor=prepend,
        append_tensor=append,
    )
    return result


##### math functions


@normalizer
def angle(z: ArrayLike, deg=False):
    result = _impl.angle(z, deg)
    return result


@normalizer
def sinc(x: ArrayLike):
    result = torch.sinc(x)
    return result


@normalizer
def real_if_close(a: ArrayLike, tol=100):
    result = _impl.real_if_close(a, tol=tol)
    return result


@normalizer
def iscomplex(x: ArrayLike):
    result = _impl.iscomplex(x)
    # XXX: missing .item on a zero-dim value; a case for array_or_scalar(value) ?
    return result


@normalizer
def isreal(x: ArrayLike):
    result = _impl.isreal(x)
    return result


@normalizer
def iscomplexobj(x: ArrayLike):
    result = torch.is_complex(x)
    return result


@normalizer
def isrealobj(x: ArrayLike):
    result = not torch.is_complex(x)
    return result


@normalizer
def isneginf(x: ArrayLike, out: Optional[NDArray] = None):
    result = torch.isneginf(x, out=out)
    return result


@normalizer
def isposinf(x: ArrayLike, out: Optional[NDArray] = None):
    result = torch.isposinf(x, out=out)
    return result


@normalizer
def i0(x: ArrayLike):
    result = torch.special.i0(x)
    return result


@normalizer(return_on_failure=False)
def isscalar(a: ArrayLike):
    # XXX: this is a stub
    if a is False:
        return a
    return a.numel() == 1


@normalizer
def isclose(a: ArrayLike, b: ArrayLike, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    result = _impl.isclose(a, b, rtol, atol, equal_nan=equal_nan)
    return result


@normalizer
def allclose(a: ArrayLike, b: ArrayLike, rtol=1e-05, atol=1e-08, equal_nan=False):
    result = _impl.isclose(a, b, rtol, atol, equal_nan=equal_nan)
    return result.all()


@normalizer
def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan=False):
    result = _impl.tensor_equal(a1, a2, equal_nan)
    return result


@normalizer
def array_equiv(a1: ArrayLike, a2: ArrayLike):
    result = _impl.tensor_equiv(a1, a2)
    return result


def common_type():
    raise NotImplementedError


def mintypecode():
    raise NotImplementedError


def nan_to_num():
    raise NotImplementedError


def asfarray():
    raise NotImplementedError


# ### put/take et al ###


@normalizer
def take_along_axis(arr: ArrayLike, indices: ArrayLike, axis):
    result = _impl.take_along_dim(arr, indices, axis)
    return result


@normalizer
def take(a: ArrayLike, indices: ArrayLike, axis=None, out=None, mode="raise"):
    if out is not None:
        raise NotImplementedError(f"{out=}")
    if mode != "raise":
        raise NotImplementedError(f"{mode=}")
    result = _impl.take(a, indices, axis)
    return result


@normalizer
def put_along_axis(arr: ArrayLike, indices: ArrayLike, values: ArrayLike, axis):
    # modify the argument in-place : here `arr` is `arr._tensor` of the orignal `arr` argument
    result = _impl.put_along_dim(arr, indices, values, axis)
    arr.copy_(result.reshape(arr.shape))
    return None


# ### unqiue et al ###


@normalizer
def unique(
    ar: ArrayLike,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    *,
    equal_nan=True,
):
    result = _impl.unique(
        ar,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
        equal_nan=equal_nan,
    )
    return result
