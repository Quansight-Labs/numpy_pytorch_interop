"""A thin pytorch / numpy compat layer.

Things imported from here have numpy-compatible signatures but operate on
pytorch tensors.
"""

from typing import Optional, Sequence

import torch

from . import _helpers
from ._detail import _dtypes_impl
from ._detail import _reductions as _impl
from ._detail import _util
from ._normalizations import (
    ArrayLike,
    AxisLike,
    DTypeLike,
    NDArray,
    SubokLike,
    normalize_array_like,
    normalizer,
)

NoValue = _util.NoValue


###### array creation routines


@normalizer
def copy(a: ArrayLike, order="K", subok: SubokLike = False):
    if order != "K":
        raise NotImplementedError
    return a.clone()


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
        # XXX:RuntimeError in torch, ValueError in numpy
        raise ValueError("need at least one array to concatenate")

    if out is not None:
        if dtype is not None:
            # mimic numpy
            raise TypeError(
                "concatenate() only takes `out` or `dtype` as an "
                "argument, but both were provided."
            )


def _concat_cast_helper(tensors, out=None, dtype=None, casting="same_kind"):
    """Figure out dtypes, cast if necessary."""

    if out is not None or dtype is not None:
        # figure out the type of the inputs and outputs
        out_dtype = out.dtype.torch_dtype if dtype is None else dtype
    else:
        out_dtype = _dtypes_impl.result_type_impl([t.dtype for t in tensors])

    # cast input arrays if necessary; do not broadcast them agains `out`
    tensors = _util.typecast_tensors(tensors, out_dtype, casting)

    return tensors


def _concatenate(tensors, axis=0, out=None, dtype=None, casting="same_kind"):
    # pure torch implementation, used below and in cov/corrcoef below
    tensors, axis = _util.axis_none_ravel(*tensors, axis=axis)
    tensors = _concat_cast_helper(tensors, out, dtype, casting)

    try:
        result = torch.cat(tensors, axis)
    except (IndexError, RuntimeError) as e:
        raise _util.AxisError(*e.args)
    return result


@normalizer
def concatenate(
    ar_tuple: Sequence[ArrayLike],
    axis=0,
    out: Optional[NDArray] = None,
    dtype: DTypeLike = None,
    casting="same_kind",
):
    _concat_check(ar_tuple, dtype, out=out)
    result = _concatenate(ar_tuple, axis=axis, out=out, dtype=dtype, casting=casting)
    return result


@normalizer
def vstack(tup: Sequence[ArrayLike], *, dtype: DTypeLike = None, casting="same_kind"):
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.vstack(tensors)


row_stack = vstack


@normalizer
def hstack(tup: Sequence[ArrayLike], *, dtype: DTypeLike = None, casting="same_kind"):
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.hstack(tensors)


@normalizer
def dstack(tup: Sequence[ArrayLike], *, dtype: DTypeLike = None, casting="same_kind"):
    # XXX: in numpy 1.24 dstack does not have dtype and casting keywords
    # but {h,v}stack do.  Hence add them here for consistency.
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.dstack(tensors)


@normalizer
def column_stack(
    tup: Sequence[ArrayLike], *, dtype: DTypeLike = None, casting="same_kind"
):
    # XXX: in numpy 1.24 column_stack does not have dtype and casting keywords
    # but row_stack does. (because row_stack is an alias for vstack, really).
    # Hence add these keywords here for consistency.
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.column_stack(tensors)


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

    tensors = _concat_cast_helper(arrays, dtype=dtype, casting=casting)
    result_ndim = tensors[0].ndim + 1
    axis = _util.normalize_axis_index(axis, result_ndim)
    try:
        result = torch.stack(tensors, axis=axis)
    except RuntimeError as e:
        raise ValueError(*e.args)
    return result


# ### split ###


def _split_helper(tensor, indices_or_sections, axis, strict=False):
    if isinstance(indices_or_sections, int):
        return _split_helper_int(tensor, indices_or_sections, axis, strict)
    elif isinstance(indices_or_sections, (list, tuple)):
        # NB: drop split=..., it only applies to split_helper_int
        return _split_helper_list(tensor, list(indices_or_sections), axis)
    else:
        raise TypeError("split_helper: ", type(indices_or_sections))


def _split_helper_int(tensor, indices_or_sections, axis, strict=False):
    if not isinstance(indices_or_sections, int):
        raise NotImplementedError("split: indices_or_sections")

    axis = _util.normalize_axis_index(axis, tensor.ndim)

    # numpy: l%n chunks of size (l//n + 1), the rest are sized l//n
    l, n = tensor.shape[axis], indices_or_sections

    if n <= 0:
        raise ValueError()

    if l % n == 0:
        num, sz = n, l // n
        lst = [sz] * num
    else:
        if strict:
            raise ValueError("array split does not result in an equal division")

        num, sz = l % n, l // n + 1
        lst = [sz] * num

    lst += [sz - 1] * (n - num)

    return torch.split(tensor, lst, axis)


def _split_helper_list(tensor, indices_or_sections, axis):
    if not isinstance(indices_or_sections, list):
        raise NotImplementedError("split: indices_or_sections: list")
    # numpy expectes indices, while torch expects lengths of sections
    # also, numpy appends zero-size arrays for indices above the shape[axis]
    lst = [x for x in indices_or_sections if x <= tensor.shape[axis]]
    num_extra = len(indices_or_sections) - len(lst)

    lst.append(tensor.shape[axis])
    lst = [
        lst[0],
    ] + [a - b for a, b in zip(lst[1:], lst[:-1])]
    lst += [0] * num_extra

    return torch.split(tensor, lst, axis)


@normalizer
def array_split(ary: ArrayLike, indices_or_sections, axis=0):
    return _split_helper(ary, indices_or_sections, axis)


@normalizer
def split(ary: ArrayLike, indices_or_sections, axis=0):
    return _split_helper(ary, indices_or_sections, axis, strict=True)


@normalizer
def hsplit(ary: ArrayLike, indices_or_sections):
    if ary.ndim == 0:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    axis = 1 if ary.ndim > 1 else 0
    return _split_helper(ary, indices_or_sections, axis, strict=True)


@normalizer
def vsplit(ary: ArrayLike, indices_or_sections):
    if ary.ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return _split_helper(ary, indices_or_sections, 0, strict=True)


@normalizer
def dsplit(ary: ArrayLike, indices_or_sections):
    if ary.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return _split_helper(ary, indices_or_sections, 2, strict=True)


@normalizer
def kron(a: ArrayLike, b: ArrayLike):
    return torch.kron(a, b)


@normalizer
def vander(x: ArrayLike, N=None, increasing=False):
    return torch.vander(x, N, increasing)


# ### linspace, geomspace, logspace and arange ###


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
    if axis != 0 or retstep or not endpoint:
        raise NotImplementedError
    # XXX: raises TypeError if start or stop are not scalars
    return torch.linspace(start, stop, num, dtype=dtype)


@normalizer
def geomspace(
    start: ArrayLike,
    stop: ArrayLike,
    num=50,
    endpoint=True,
    dtype: DTypeLike = None,
    axis=0,
):
    if axis != 0 or not endpoint:
        raise NotImplementedError
    base = torch.pow(stop / start, 1.0 / (num - 1))
    logbase = torch.log(base)
    return torch.logspace(
        torch.log(start) / logbase,
        torch.log(stop) / logbase,
        num,
        base=base,
    )


@normalizer
def logspace(
    start, stop, num=50, endpoint=True, base=10.0, dtype: DTypeLike = None, axis=0
):
    if axis != 0 or not endpoint:
        raise NotImplementedError
    return torch.logspace(start, stop, num, base=base, dtype=dtype)


@normalizer
def arange(
    start: Optional[ArrayLike] = None,
    stop: Optional[ArrayLike] = None,
    step: Optional[ArrayLike] = 1,
    dtype: DTypeLike = None,
    *,
    like: SubokLike = None,
):
    if step == 0:
        raise ZeroDivisionError
    if stop is None and start is None:
        raise TypeError
    if stop is None:
        # XXX: this breaks if start is passed as a kwarg:
        # arange(start=4) should raise (no stop) but doesn't
        start, stop = 0, start
    if start is None:
        start = 0

    # the dtype of the result
    if dtype is None:
        dtype = _dtypes_impl.default_int_dtype
    dt_list = [_util._coerce_to_tensor(x).dtype for x in (start, stop, step)]
    dt_list.append(dtype)
    dtype = _dtypes_impl.result_type_impl(dt_list)

    # work around RuntimeError: "arange_cpu" not implemented for 'ComplexFloat'
    if dtype.is_complex:
        work_dtype, target_dtype = torch.float64, dtype
    else:
        work_dtype, target_dtype = dtype, dtype

    if (step > 0 and start > stop) or (step < 0 and start < stop):
        # empty range
        return torch.empty(0, dtype=target_dtype)

    try:
        result = torch.arange(start, stop, step, dtype=work_dtype)
        result = _util.cast_if_needed(result, target_dtype)
    except RuntimeError:
        raise ValueError("Maximum allowed size exceeded")

    return result


# ### zeros/ones/empty/full ###


@normalizer
def empty(shape, dtype: DTypeLike = float, order="C", *, like: SubokLike = None):
    if order != "C":
        raise NotImplementedError
    if dtype is None:
        dtype = _dtypes_impl.default_float_dtype
    return torch.empty(shape, dtype=dtype)


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
    if order != "K":
        raise NotImplementedError
    result = torch.empty_like(prototype, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
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
    if isinstance(shape, int):
        shape = (shape,)
    if order != "C":
        raise NotImplementedError
    if dtype is None:
        dtype = fill_value.dtype
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    return torch.full(shape, fill_value, dtype=dtype)


@normalizer
def full_like(
    a: ArrayLike,
    fill_value,
    dtype: DTypeLike = None,
    order="K",
    subok: SubokLike = False,
    shape=None,
):
    if order != "K":
        raise NotImplementedError
    # XXX: fill_value broadcasts
    result = torch.full_like(a, fill_value, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


@normalizer
def ones(shape, dtype: DTypeLike = None, order="C", *, like: SubokLike = None):
    if order != "C":
        raise NotImplementedError
    if dtype is None:
        dtype = _dtypes_impl.default_float_dtype
    return torch.ones(shape, dtype=dtype)


@normalizer
def ones_like(
    a: ArrayLike,
    dtype: DTypeLike = None,
    order="K",
    subok: SubokLike = False,
    shape=None,
):
    if order != "K":
        raise NotImplementedError
    result = torch.ones_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


@normalizer
def zeros(shape, dtype: DTypeLike = None, order="C", *, like: SubokLike = None):
    if order != "C":
        raise NotImplementedError
    if dtype is None:
        dtype = _dtypes_impl.default_float_dtype
    return torch.zeros(shape, dtype=dtype)


@normalizer
def zeros_like(
    a: ArrayLike,
    dtype: DTypeLike = None,
    order="K",
    subok: SubokLike = False,
    shape=None,
):
    if order != "K":
        raise NotImplementedError
    result = torch.zeros_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


# ### cov & corrcoef ###


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

        x_tensor = _concatenate((x_tensor, y_tensor), axis=0)

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
    xy_tensor = _xy_helper_corrcoef(x, y, rowvar)

    is_half = dtype == torch.float16
    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32

    xy_tensor = _util.cast_if_needed(xy_tensor, dtype)
    result = torch.corrcoef(xy_tensor)

    if is_half:
        result = result.to(torch.float16)

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

    if ddof is None:
        ddof = 1 if bias == 0 else 0

    is_half = dtype == torch.float16
    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32

    m = _util.cast_if_needed(m, dtype)
    result = torch.cov(m, correction=ddof, aweights=aweights, fweights=fweights)

    if is_half:
        result = result.to(torch.float16)

    return result


# ### logic & element selection ###


@normalizer
def bincount(x: ArrayLike, /, weights: Optional[ArrayLike] = None, minlength=0):
    if x.numel() == 0:
        # edge case allowed by numpy
        x = x.new_empty(0, dtype=int)

    int_dtype = _dtypes_impl.default_int_dtype
    (x,) = _util.typecast_tensors((x,), int_dtype, casting="safe")

    return torch.bincount(x, weights, minlength)


@normalizer
def where(
    condition: ArrayLike,
    x: Optional[ArrayLike] = None,
    y: Optional[ArrayLike] = None,
    /,
):
    selector = (x is None) == (y is None)
    if not selector:
        raise ValueError("either both or neither of x and y should be given")

    if condition.dtype != torch.bool:
        condition = condition.to(torch.bool)

    if x is None and y is None:
        result = torch.where(condition)
    else:
        try:
            result = torch.where(condition, x, y)
        except RuntimeError as e:
            raise ValueError(*e.args)
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
    return a.view(shape)  # never copies


@normalizer
def flip(m: ArrayLike, axis=None):
    # XXX: semantic difference: np.flip returns a view, torch.flip copies
    if axis is None:
        axis = tuple(range(m.ndim))
    else:
        axis = _util.normalize_axis_tuple(axis, m.ndim)
    return torch.flip(m, axis)


@normalizer
def flipud(m: ArrayLike):
    return torch.flipud(m)


@normalizer
def fliplr(m: ArrayLike):
    return torch.fliplr(m)


@normalizer
def rot90(m: ArrayLike, k=1, axes=(0, 1)):
    axes = _util.normalize_axis_tuple(axes, m.ndim)
    return torch.rot90(m, k, axes)


# ### broadcasting and indices ###


@normalizer
def broadcast_to(array: ArrayLike, shape, subok: SubokLike = False):
    return torch.broadcast_to(array, size=shape)


from torch import broadcast_shapes


@normalizer
def broadcast_arrays(*args: ArrayLike, subok: SubokLike = False):
    return torch.broadcast_tensors(*args)


@normalizer
def meshgrid(*xi: ArrayLike, copy=True, sparse=False, indexing="xy"):
    ndim = len(xi)

    if indexing not in ["xy", "ij"]:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1 :]) for i, x in enumerate(xi)]

    if indexing == "xy" and ndim > 1:
        # switch first and second axis
        output[0] = output[0].reshape((1, -1) + s0[2:])
        output[1] = output[1].reshape((-1, 1) + s0[2:])

    if not sparse:
        # Return the full N-D matrix (not only the 1-D vector)
        output = torch.broadcast_tensors(*output)

    if copy:
        output = [x.clone() for x in output]

    return list(output)  # match numpy, return a list


@normalizer
def indices(dimensions, dtype: DTypeLike = int, sparse=False):
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1691-L1791
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    if sparse:
        res = tuple()
    else:
        res = torch.empty((N,) + dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        idx = torch.arange(dim, dtype=dtype).reshape(
            shape[:i] + (dim,) + shape[i + 1 :]
        )
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    return res


# ### tri*-something ###


@normalizer
def tril(m: ArrayLike, k=0):
    return torch.tril(m, k)


@normalizer
def triu(m: ArrayLike, k=0):
    return torch.triu(m, k)


def tril_indices(n, k=0, m=None):
    if m is None:
        m = n
    return torch.tril_indices(n, m, offset=k)


def triu_indices(n, k=0, m=None):
    if m is None:
        m = n
    return torch.triu_indices(n, m, offset=k)


@normalizer
def tril_indices_from(arr: ArrayLike, k=0):
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    result = torch.tril_indices(arr.shape[0], arr.shape[1], offset=k)
    return tuple(result)


@normalizer
def triu_indices_from(arr: ArrayLike, k=0):
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    result = torch.triu_indices(arr.shape[0], arr.shape[1], offset=k)
    # unpack: numpy returns a 2-tuple of index arrays; torch returns a 2-row tensor
    return tuple(result)


@normalizer
def tri(N, M=None, k=0, dtype: DTypeLike = float, *, like: SubokLike = None):
    if M is None:
        M = N
    tensor = torch.ones((N, M), dtype=dtype)
    tensor = torch.tril(tensor, diagonal=k)
    return tensor


# ### nanfunctions ###  # FIXME: this is a stub


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


# ### equality, equivalence, allclose ###


@normalizer
def isclose(a: ArrayLike, b: ArrayLike, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    dtype = _dtypes_impl.result_type_impl((a.dtype, b.dtype))
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    result = torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return result


@normalizer
def allclose(a: ArrayLike, b: ArrayLike, rtol=1e-05, atol=1e-08, equal_nan=False):
    dtype = _dtypes_impl.result_type_impl((a.dtype, b.dtype))
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def _tensor_equal(a1, a2, equal_nan=False):
    # Implementation of array_equal/array_equiv.
    if equal_nan:
        return (a1.shape == a2.shape) and (
            (a1 == a2) | (torch.isnan(a1) & torch.isnan(a2))
        ).all().item()
    else:
        return torch.equal(a1, a2)


@normalizer
def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan=False):
    return _tensor_equal(a1, a2, equal_nan=equal_nan)


@normalizer
def array_equiv(a1: ArrayLike, a2: ArrayLike):
    # *almost* the same as array_equal: _equiv tries to broadcast, _equal does not
    try:
        a1_t, a2_t = torch.broadcast_tensors(a1, a2)
    except RuntimeError:
        # failed to broadcast => not equivalent
        return False
    return _tensor_equal(a1_t, a2_t)


def mintypecode():
    raise NotImplementedError


def nan_to_num():
    raise NotImplementedError


def asfarray():
    raise NotImplementedError


def block(*args, **kwds):
    raise NotImplementedError


# ### put/take_along_axis ###


@normalizer
def take(
    a: ArrayLike,
    indices: ArrayLike,
    axis=None,
    out: Optional[NDArray] = None,
    mode="raise",
):
    if mode != "raise":
        raise NotImplementedError(f"{mode=}")

    (a,), axis = _util.axis_none_ravel(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)
    idx = (slice(None),) * axis + (indices, ...)
    result = a[idx]
    return result


@normalizer
def take_along_axis(arr: ArrayLike, indices: ArrayLike, axis):
    (arr,), axis = _util.axis_none_ravel(arr, axis=axis)
    axis = _util.normalize_axis_index(axis, arr.ndim)
    return torch.take_along_dim(arr, indices, axis)


@normalizer
def put_along_axis(arr: ArrayLike, indices: ArrayLike, values: ArrayLike, axis):
    (arr,), axis = _util.axis_none_ravel(arr, axis=axis)
    axis = _util.normalize_axis_index(axis, arr.ndim)

    indices, values = torch.broadcast_tensors(indices, values)
    values = _util.cast_if_needed(values, arr.dtype)
    result = torch.scatter(arr, axis, indices, values)
    arr.copy_(result.reshape(arr.shape))
    return None


# ### unique et al ###


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
    if return_index or not equal_nan:
        raise NotImplementedError

    if axis is None:
        ar = ar.ravel()
        axis = 0
    axis = _util.normalize_axis_index(axis, ar.ndim)

    is_half = ar.dtype == torch.float16
    if is_half:
        ar = ar.to(torch.float32)

    result = torch.unique(
        ar, return_inverse=return_inverse, return_counts=return_counts, dim=axis
    )

    if is_half:
        if isinstance(result, tuple):
            result = (result[0].to(torch.float16),) + result[1:]
        else:
            result = result.to(torch.float16)

    return result


@normalizer
def nonzero(a: ArrayLike):
    return torch.nonzero(a, as_tuple=True)


@normalizer
def argwhere(a: ArrayLike):
    return torch.argwhere(a)


@normalizer
def flatnonzero(a: ArrayLike):
    return torch.ravel(a).nonzero(as_tuple=True)[0]


@normalizer
def clip(
    a: ArrayLike,
    min: Optional[ArrayLike] = None,
    max: Optional[ArrayLike] = None,
    out: Optional[NDArray] = None,
):
    # np.clip requires both a_min and a_max not None, while ndarray.clip allows
    # one of them to be None. Follow the more lax version.
    if min is None and max is None:
        raise ValueError("One of max or min must be given")
    result = torch.clamp(a, min, max)
    return result


@normalizer
def repeat(a: ArrayLike, repeats: ArrayLike, axis=None):
    # XXX: scalar repeats; ArrayLikeOrScalar ?
    return torch.repeat_interleave(a, repeats, axis)


@normalizer
def tile(A: ArrayLike, reps):
    if isinstance(reps, int):
        reps = (reps,)
    return torch.tile(A, reps)


# ### diag et al ###


@normalizer
def diagonal(a: ArrayLike, offset=0, axis1=0, axis2=1):
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    return torch.diagonal(a, offset, axis1, axis2)


@normalizer
def trace(
    a: ArrayLike,
    offset=0,
    axis1=0,
    axis2=1,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
):
    result = torch.diagonal(a, offset, dim1=axis1, dim2=axis2).sum(-1, dtype=dtype)
    return result


@normalizer
def eye(N, M=None, k=0, dtype: DTypeLike = float, order="C", *, like: SubokLike = None):
    if order != "C":
        raise NotImplementedError
    if M is None:
        M = N
    z = torch.zeros(N, M, dtype=dtype)
    z.diagonal(k).fill_(1)
    return z


@normalizer
def identity(n, dtype: DTypeLike = None, *, like: SubokLike = None):
    return torch.eye(n, dtype=dtype)


@normalizer
def diag(v: ArrayLike, k=0):
    return torch.diag(v, k)


@normalizer
def diagflat(v: ArrayLike, k=0):
    return torch.diagflat(v, k)


def diag_indices(n, ndim=2):
    idx = torch.arange(n)
    return (idx,) * ndim


@normalizer
def diag_indices_from(arr: ArrayLike):
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    s = arr.shape
    if s[1:] != s[:-1]:
        raise ValueError("All dimensions of input must be of equal length")
    return diag_indices(s[0], arr.ndim)


@normalizer
def fill_diagonal(a: ArrayLike, val: ArrayLike, wrap=False):
    # torch.Tensor.fill_diagonal_ only accepts scalars. Thus vendor the numpy source,
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/index_tricks.py#L786-L917

    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    end = None
    if a.ndim == 2:
        # Explicit, fast formula for the common case.  For 2-d arrays, we
        # accept rectangular ones.
        step = a.shape[1] + 1
        # This is needed to don't have tall matrix have the diagonal wrap.
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        # For more than d=2, the strided formula is only valid for arrays with
        # all dimensions equal, so we check first.
        s = a.shape
        if s[1:] != s[:-1]:
            raise ValueError("All dimensions of input must be of equal length")
        sz = torch.as_tensor(a.shape[:-1])
        step = 1 + (torch.cumprod(sz, 0)).sum()

    # Write the value out into the diagonal.
    a.ravel()[:end:step] = val
    return a


@normalizer
def vdot(a: ArrayLike, b: ArrayLike, /):
    # 1. torch only accepts 1D arrays, numpy ravels
    # 2. torch requires matching dtype, while numpy casts (?)
    t_a, t_b = torch.atleast_1d(a, b)
    if t_a.ndim > 1:
        t_a = t_a.ravel()
    if t_b.ndim > 1:
        t_b = t_b.ravel()

    dtype = _dtypes_impl.result_type_impl((t_a.dtype, t_b.dtype))
    is_half = dtype == torch.float16
    is_bool = dtype == torch.bool

    # work around torch's "dot" not implemented for 'Half', 'Bool'
    if is_half:
        dtype = torch.float32
    elif is_bool:
        dtype = torch.uint8

    t_a = _util.cast_if_needed(t_a, dtype)
    t_b = _util.cast_if_needed(t_b, dtype)

    result = torch.vdot(t_a, t_b)

    if is_half:
        result = result.to(torch.float16)
    elif is_bool:
        result = result.to(torch.bool)

    return result.item()


@normalizer
def tensordot(a: ArrayLike, b: ArrayLike, axes=2):
    if isinstance(axes, (list, tuple)):
        axes = [[ax] if isinstance(ax, int) else ax for ax in axes]
    return torch.tensordot(a, b, dims=axes)


@normalizer
def dot(a: ArrayLike, b: ArrayLike, out: Optional[NDArray] = None):
    dtype = _dtypes_impl.result_type_impl((a.dtype, b.dtype))
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    if a.ndim == 0 or b.ndim == 0:
        result = a * b
    else:
        result = torch.matmul(a, b)
    return result


@normalizer
def inner(a: ArrayLike, b: ArrayLike, /):
    dtype = _dtypes_impl.result_type_impl((a.dtype, b.dtype))
    is_half = dtype == torch.float16
    is_bool = dtype == torch.bool

    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32
    elif is_bool:
        dtype = torch.uint8

    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    result = torch.inner(a, b)

    if is_half:
        result = result.to(torch.float16)
    elif is_bool:
        result = result.to(torch.bool)
    return result


@normalizer
def outer(a: ArrayLike, b: ArrayLike, out: Optional[NDArray] = None):
    return torch.outer(a, b)


# ### sort and partition ###


def _sort_helper(tensor, axis, kind, order):
    if order is not None:
        # only relevant for structured dtypes; not supported
        raise NotImplementedError(
            "'order' keyword is only relevant for structured dtypes"
        )

    (tensor,), axis = _util.axis_none_ravel(tensor, axis=axis)
    axis = _util.normalize_axis_index(axis, tensor.ndim)

    stable = kind == "stable"

    return tensor, axis, stable


def _sort(tensor, axis, kind, order):
    # pure torch implementation, used below and in ndarray.sort
    tensor, axis, stable = _sort_helper(tensor, axis, kind, order)
    result = torch.sort(tensor, dim=axis, stable=stable)
    return result.values


@normalizer
def sort(a: ArrayLike, axis=-1, kind=None, order=None):
    return _sort(a, axis, kind, order)


@normalizer
def argsort(a: ArrayLike, axis=-1, kind=None, order=None):
    a, axis, stable = _sort_helper(a, axis, kind, order)
    return torch.argsort(a, dim=axis, stable=stable)


@normalizer
def searchsorted(
    a: ArrayLike, v: ArrayLike, side="left", sorter: Optional[ArrayLike] = None
):
    return torch.searchsorted(a, v, side=side, sorter=sorter)


# ### swap/move/roll axis ###


@normalizer
def moveaxis(a: ArrayLike, source, destination):
    source = _util.normalize_axis_tuple(source, a.ndim, "source")
    destination = _util.normalize_axis_tuple(destination, a.ndim, "destination")
    return torch.moveaxis(a, source, destination)


@normalizer
def swapaxes(a: ArrayLike, axis1, axis2):
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    return torch.swapaxes(a, axis1, axis2)


@normalizer
def rollaxis(a: ArrayLike, axis, start=0):
    # Straight vendor from:
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1259
    #
    # Also note this function in NumPy is mostly retained for backwards compat
    # (https://stackoverflow.com/questions/29891583/reason-why-numpy-rollaxis-is-so-confusing)
    # so let's not touch it unless hard pressed.
    n = a.ndim
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
        return a
    axes = list(range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.view(axes)


@normalizer
def roll(a: ArrayLike, shift, axis=None):
    if axis is not None:
        axis = _util.normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        if not isinstance(shift, tuple):
            shift = (shift,) * len(axis)
    return torch.roll(a, shift, axis)


# ### shape manipulations ###


@normalizer
def squeeze(a: ArrayLike, axis=None):
    if axis == ():
        result = a
    elif axis is None:
        result = a.squeeze()
    else:
        if isinstance(axis, tuple):
            result = a
            for ax in axis:
                result = a.squeeze(ax)
        else:
            result = a.squeeze(axis)
    return result


@normalizer
def reshape(a: ArrayLike, newshape, order="C"):
    if order != "C":
        raise NotImplementedError
    # if sh = (1, 2, 3), numpy allows both .reshape(sh) and .reshape(*sh)
    newshape = newshape[0] if len(newshape) == 1 else newshape
    return a.reshape(newshape)


# NB: cannot use torch.reshape(a, newshape) above, because of
# (Pdb) torch.reshape(torch.as_tensor([1]), 1)
# *** TypeError: reshape(): argument 'shape' (position 2) must be tuple of SymInts, not int


@normalizer
def transpose(a: ArrayLike, axes=None):
    # numpy allows both .tranpose(sh) and .transpose(*sh)
    axes = axes[0] if len(axes) == 1 else axes
    if axes in [(), None, (None,)]:
        axes = tuple(range(a.ndim))[::-1]
    try:
        result = a.permute(axes)
    except RuntimeError:
        raise ValueError("axes don't match array")
    return result


@normalizer
def ravel(a: ArrayLike, order="C"):
    if order != "C":
        raise NotImplementedError
    return torch.ravel(a)


# leading underscore since arr.flatten exists but np.flatten does not
@normalizer
def _flatten(a: ArrayLike, order="C"):
    if order != "C":
        raise NotImplementedError
    # may return a copy
    return torch.flatten(a)


# ### Type/shape etc queries ###


@normalizer
def real(a: ArrayLike):
    result = torch.real(a)
    return result


@normalizer
def imag(a: ArrayLike):
    if a.is_complex():
        result = a.imag
    else:
        result = torch.zeros_like(a)
    return result


@normalizer
def round_(a: ArrayLike, decimals=0, out: Optional[NDArray] = None):
    if a.is_floating_point():
        result = torch.round(a, decimals=decimals)
    elif a.is_complex():
        # RuntimeError: "round_cpu" not implemented for 'ComplexFloat'
        result = (
            torch.round(a.real, decimals=decimals)
            + torch.round(a.imag, decimals=decimals) * 1j
        )
    else:
        # RuntimeError: "round_cpu" not implemented for 'int'
        result = a
    return result


around = round_
round = round_


# ### reductions ###


@normalizer
def sum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    initial=NoValue,
    where=NoValue,
):
    result = _impl.sum(
        a, axis=axis, dtype=dtype, initial=initial, where=where, keepdims=keepdims
    )
    return result


@normalizer
def prod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    initial=NoValue,
    where=NoValue,
):
    result = _impl.prod(
        a, axis=axis, dtype=dtype, initial=initial, where=where, keepdims=keepdims
    )
    return result


product = prod


@normalizer
def mean(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    *,
    where=NoValue,
):
    result = _impl.mean(a, axis=axis, dtype=dtype, where=NoValue, keepdims=keepdims)
    return result


@normalizer
def var(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
    ddof=0,
    keepdims=NoValue,
    *,
    where=NoValue,
):
    result = _impl.var(
        a, axis=axis, dtype=dtype, ddof=ddof, where=where, keepdims=keepdims
    )
    return result


@normalizer
def std(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
    ddof=0,
    keepdims=NoValue,
    *,
    where=NoValue,
):
    result = _impl.std(
        a, axis=axis, dtype=dtype, ddof=ddof, where=where, keepdims=keepdims
    )
    return result


@normalizer
def argmin(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    *,
    keepdims=NoValue,
):
    result = _impl.argmin(a, axis=axis, keepdims=keepdims)
    return result


@normalizer
def argmax(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    *,
    keepdims=NoValue,
):
    result = _impl.argmax(a, axis=axis, keepdims=keepdims)
    return result


@normalizer
def amax(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    initial=NoValue,
    where=NoValue,
):
    result = _impl.max(a, axis=axis, initial=initial, where=where, keepdims=keepdims)
    return result


max = amax


@normalizer
def amin(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    initial=NoValue,
    where=NoValue,
):
    result = _impl.min(a, axis=axis, initial=initial, where=where, keepdims=keepdims)
    return result


min = amin


@normalizer
def ptp(
    a: ArrayLike, axis: AxisLike = None, out: Optional[NDArray] = None, keepdims=NoValue
):
    result = _impl.ptp(a, axis=axis, keepdims=keepdims)
    return result


@normalizer
def all(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    *,
    where=NoValue,
):
    result = _impl.all(a, axis=axis, where=where, keepdims=keepdims)
    return result


@normalizer
def any(
    a: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    keepdims=NoValue,
    *,
    where=NoValue,
):
    result = _impl.any(a, axis=axis, where=where, keepdims=keepdims)
    return result


@normalizer
def count_nonzero(a: ArrayLike, axis: AxisLike = None, *, keepdims=False):
    result = _impl.count_nonzero(a, axis=axis, keepdims=keepdims)
    return result


@normalizer
def cumsum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
):
    result = _impl.cumsum(a, axis=axis, dtype=dtype)
    return result


@normalizer
def cumprod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike = None,
    out: Optional[NDArray] = None,
):
    result = _impl.cumprod(a, axis=axis, dtype=dtype)
    return result


cumproduct = cumprod


@normalizer(promote_scalar_result=True)
def quantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None,
):
    result = _impl.quantile(
        a,
        q,
        axis,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        interpolation=interpolation,
    )
    return result


@normalizer(promote_scalar_result=True)
def percentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = None,
    out: Optional[NDArray] = None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None,
):
    result = _impl.percentile(
        a,
        q,
        axis,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        interpolation=interpolation,
    )
    return result


def median(
    a, axis=None, out: Optional[NDArray] = None, overwrite_input=False, keepdims=False
):
    return quantile(
        a, 0.5, axis=axis, overwrite_input=overwrite_input, out=out, keepdims=keepdims
    )


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
def diff(
    a: ArrayLike,
    n=1,
    axis=-1,
    prepend: Optional[ArrayLike] = NoValue,
    append: Optional[ArrayLike] = NoValue,
):
    axis = _util.normalize_axis_index(axis, a.ndim)

    if n < 0:
        raise ValueError(f"order must be non-negative but got {n}")

    if n == 0:
        # match numpy and return the input immediately
        return a

    if prepend is not None:
        shape = list(a.shape)
        shape[axis] = prepend.shape[axis] if prepend.ndim > 0 else 1
        prepend = torch.broadcast_to(prepend, shape)

    if append is not None:
        shape = list(a.shape)
        shape[axis] = append.shape[axis] if append.ndim > 0 else 1
        append = torch.broadcast_to(append, shape)

    return torch.diff(a, n, axis=axis, prepend=prepend, append=append)


# ### math functions ###


@normalizer
def angle(z: ArrayLike, deg=False):
    result = torch.angle(z)
    if deg:
        result = result * 180 / torch.pi
    return result


@normalizer
def sinc(x: ArrayLike):
    return torch.sinc(x)


@normalizer
def real(a: ArrayLike):
    return torch.real(a)


@normalizer
def imag(a: ArrayLike):
    if a.is_complex():
        result = a.imag
    else:
        result = torch.zeros_like(a)
    return result


@normalizer
def round_(a: ArrayLike, decimals=0, out: Optional[NDArray] = None):
    if a.is_floating_point():
        result = torch.round(a, decimals=decimals)
    elif a.is_complex():
        # RuntimeError: "round_cpu" not implemented for 'ComplexFloat'
        result = (
            torch.round(a.real, decimals=decimals)
            + torch.round(a.imag, decimals=decimals) * 1j
        )
    else:
        # RuntimeError: "round_cpu" not implemented for 'int'
        result = a
    return result


around = round_
round = round_


@normalizer
def real_if_close(a: ArrayLike, tol=100):
    # XXX: copies vs views; numpy seems to return a copy?
    if not torch.is_complex(a):
        return a
    if tol > 1:
        # Undocumented in numpy: if tol < 1, it's an absolute tolerance!
        # Otherwise, tol > 1 is relative tolerance, in units of the dtype epsilon
        # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/type_check.py#L577
        tol = tol * torch.finfo(a.dtype).eps

    mask = torch.abs(a.imag) < tol
    return a.real if mask.all() else a


@normalizer
def iscomplex(x: ArrayLike):
    if torch.is_complex(x):
        return x.imag != 0
    result = torch.zeros_like(x, dtype=torch.bool)
    if result.ndim == 0:
        result = result.item()
    return result


@normalizer
def isreal(x: ArrayLike):
    if torch.is_complex(x):
        return x.imag == 0
    result = torch.ones_like(x, dtype=torch.bool)
    if result.ndim == 0:
        result = result.item()
    return result


@normalizer
def iscomplexobj(x: ArrayLike):
    result = torch.is_complex(x)
    return result


@normalizer
def isrealobj(x: ArrayLike):
    return not torch.is_complex(x)


@normalizer
def isneginf(x: ArrayLike, out: Optional[NDArray] = None):
    return torch.isneginf(x, out=out)


@normalizer
def isposinf(x: ArrayLike, out: Optional[NDArray] = None):
    return torch.isposinf(x, out=out)


@normalizer
def i0(x: ArrayLike):
    return torch.special.i0(x)


def isscalar(a):
    # XXX: this is a stub
    try:
        t = normalize_array_like(a)
        return t.numel() == 1
    except Exception:
        return False


"""
Vendored objects from numpy.lib.index_tricks
"""


class IndexExpression:
    """
    Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
    last revision: 1999-7-23

    Cosmetic changes by T. Oliphant 2001
    """

    def __init__(self, maketuple):
        self.maketuple = maketuple

    def __getitem__(self, item):
        if self.maketuple and not isinstance(item, tuple):
            return (item,)
        else:
            return item


index_exp = IndexExpression(maketuple=True)
s_ = IndexExpression(maketuple=False)


# ### Filter windows ###


@normalizer
def hamming(M):
    dtype = _dtypes_impl.default_float_dtype
    return torch.hamming_window(M, periodic=False, dtype=dtype)


@normalizer
def hanning(M):
    dtype = _dtypes_impl.default_float_dtype
    return torch.hann_window(M, periodic=False, dtype=dtype)


@normalizer
def kaiser(M, beta):
    dtype = _dtypes_impl.default_float_dtype
    return torch.kaiser_window(M, beta=beta, periodic=False, dtype=dtype)


@normalizer
def blackman(M):
    dtype = _dtypes_impl.default_float_dtype
    return torch.blackman_window(M, periodic=False, dtype=dtype)


@normalizer
def bartlett(M):
    dtype = _dtypes_impl.default_float_dtype
    return torch.bartlett_window(M, periodic=False, dtype=dtype)



# ### Dtype routines ###

# vendored from https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/type_check.py#L666


array_type = [[torch.float16, torch.float32, torch.float64],
              [None, torch.complex64, torch.complex128]]
array_precision = {torch.float16: 0,
                   torch.float32: 1,
                   torch.float64: 2,
                   torch.complex64: 1,
                   torch.complex128: 2,}

@normalizer
def common_type(*tensors: ArrayLike):

    import builtins

    is_complex = False
    precision = 0
    for a in tensors:
        t = a.dtype
        if iscomplexobj(a):
            is_complex = True
        if not(t.is_floating_point or t.is_complex):
            p = 2  # array_precision[_nx.double]
        else:
            p = array_precision.get(t, None)
            if p is None:
                raise TypeError("can't get common type for non-numeric array")
        precision = builtins.max(precision, p)
    if is_complex:
        return array_type[1][precision]
    else:
        return array_type[0][precision]


