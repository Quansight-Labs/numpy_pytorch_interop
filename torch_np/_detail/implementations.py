import torch

from . import _dtypes_impl, _util

# ### equality, equivalence, allclose ###


def tensor_equal(a1_t, a2_t, equal_nan=False):
    """Implementation of array_equal/array_equiv."""
    if a1_t.shape != a2_t.shape:
        return False
    if equal_nan:
        nan_loc = (torch.isnan(a1_t) == torch.isnan(a2_t)).all()
        if nan_loc:
            # check the values
            result = a1_t[~torch.isnan(a1_t)] == a2_t[~torch.isnan(a2_t)]
        else:
            return False
    else:
        result = a1_t == a2_t
    return bool(result.all())


def tensor_equiv(a1_t, a2_t):
    # *almost* the same as tensor_equal: _equiv tries to broadcast, _equal does not
    try:
        a1_t, a2_t = torch.broadcast_tensors(a1_t, a2_t)
    except RuntimeError:
        # failed to broadcast => not equivalent
        return False
    return tensor_equal(a1_t, a2_t)


def isclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    dtype = _dtypes_impl.result_type_impl((a.dtype, b.dtype))
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    result = torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return result


# ### is arg real or complex valued ###


def iscomplex(x):
    if torch.is_complex(x):
        return torch.as_tensor(x).imag != 0
    result = torch.zeros_like(x, dtype=torch.bool)
    if result.ndim == 0:
        result = result.item()
    return result


def isreal(x):
    if torch.is_complex(x):
        return torch.as_tensor(x).imag == 0
    result = torch.ones_like(x, dtype=torch.bool)
    if result.ndim == 0:
        result = result.item()
    return result


def real_if_close(x, tol=100):
    # XXX: copies vs views; numpy seems to return a copy?
    if not torch.is_complex(x):
        return x
    if tol > 1:
        # Undocumented in numpy: if tol < 1, it's an absolute tolerance!
        # Otherwise, tol > 1 is relative tolerance, in units of the dtype epsilon
        # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/type_check.py#L577
        tol = tol * torch.finfo(x.dtype).eps

    mask = torch.abs(x.imag) < tol
    if mask.all():
        return x.real
    else:
        return x


# ### math functions ###


def angle(z, deg=False):
    result = torch.angle(z)
    if deg:
        result = result * 180 / torch.pi
    return result


# ### sorting ###


def argsort(tensor, axis=-1, kind=None, order=None):
    if order is not None:
        raise NotImplementedError
    stable = kind == "stable"
    if axis is None:
        axis = -1
    return torch.argsort(tensor, stable=stable, dim=axis, descending=False)


# ### tri*-something ###


def tri(N, M, k, dtype):
    if M is None:
        M = N
    tensor = torch.ones((N, M), dtype=dtype)
    tensor = torch.tril(tensor, diagonal=k)
    return tensor


def triu_indices_from(tensor, k):
    if tensor.ndim != 2:
        raise ValueError("input array must be 2-d")
    result = torch.triu_indices(tensor.shape[0], tensor.shape[1], offset=k)
    return result


def tril_indices_from(tensor, k=0):
    if tensor.ndim != 2:
        raise ValueError("input array must be 2-d")
    result = torch.tril_indices(tensor.shape[0], tensor.shape[1], offset=k)
    return result


def tril_indices(n, k=0, m=None):
    if m is None:
        m = n
    result = torch.tril_indices(n, m, offset=k)
    return result


def triu_indices(n, k=0, m=None):
    if m is None:
        m = n
    result = torch.triu_indices(n, m, offset=k)
    return result


# ### splits ###


def split_helper(tensor, indices_or_sections, axis, strict=False):
    if isinstance(indices_or_sections, int):
        return split_helper_int(tensor, indices_or_sections, axis, strict)
    elif isinstance(indices_or_sections, (list, tuple)):
        # NB: drop split=..., it only applies to split_helper_int
        return split_helper_list(tensor, list(indices_or_sections), axis)
    else:
        raise TypeError("split_helper: ", type(indices_or_sections))


def split_helper_int(tensor, indices_or_sections, axis, strict=False):
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

    result = torch.split(tensor, lst, axis)

    return result


def split_helper_list(tensor, indices_or_sections, axis):
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


def hsplit(tensor, indices_or_sections):
    if tensor.ndim == 0:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    axis = 1 if tensor.ndim > 1 else 0
    return split_helper(tensor, indices_or_sections, axis, strict=True)


def vsplit(tensor, indices_or_sections):
    if tensor.ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return split_helper(tensor, indices_or_sections, 0, strict=True)


def dsplit(tensor, indices_or_sections):
    if tensor.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return split_helper(tensor, indices_or_sections, 2, strict=True)


def clip(tensor, t_min, t_max):
    if t_min is not None:
        t_min = torch.broadcast_to(t_min, tensor.shape)

    if t_max is not None:
        t_max = torch.broadcast_to(t_max, tensor.shape)

    if t_min is None and t_max is None:
        raise ValueError("One of max or min must be given")

    result = tensor.clamp(t_min, t_max)
    return result


def diff(a_tensor, n=1, axis=-1, prepend_tensor=None, append_tensor=None):
    axis = _util.normalize_axis_index(axis, a_tensor.ndim)

    if n < 0:
        raise ValueError(f"order must be non-negative but got {n}")

    if prepend_tensor is not None:
        shape = list(a_tensor.shape)
        shape[axis] = prepend_tensor.shape[axis] if prepend_tensor.ndim > 0 else 1
        prepend_tensor = torch.broadcast_to(prepend_tensor, shape)

    if append_tensor is not None:
        shape = list(a_tensor.shape)
        shape[axis] = append_tensor.shape[axis] if append_tensor.ndim > 0 else 1
        append_tensor = torch.broadcast_to(append_tensor, shape)

    result = torch.diff(
        a_tensor, n, axis=axis, prepend=prepend_tensor, append=append_tensor
    )

    return result


# #### concatenate and relatives


def _concat_cast_helper(tensors, out=None, dtype=None, casting="same_kind"):
    """Figure out dtypes, cast if necessary."""

    if out is not None or dtype is not None:
        # figure out the type of the inputs and outputs
        out_dtype = out.dtype.torch_dtype if dtype is None else dtype
    else:
        out_dtype = _dtypes_impl.result_type_impl([t.dtype for t in tensors])

    # cast input arrays if necessary; do not broadcast them agains `out`
    tensors = _util.cast_dont_broadcast(tensors, out_dtype, casting)

    return tensors


def concatenate(tensors, axis=0, out=None, dtype=None, casting="same_kind"):
    # np.concatenate ravels if axis=None
    tensors, axis = _util.axis_none_ravel(*tensors, axis=axis)
    tensors = _concat_cast_helper(tensors, out, dtype, casting)

    try:
        result = torch.cat(tensors, axis)
    except (IndexError, RuntimeError) as e:
        raise _util.AxisError(*e.args)

    return result


def stack(tensors, axis=0, out=None, *, dtype=None, casting="same_kind"):
    tensors = _concat_cast_helper(tensors, dtype=dtype, casting=casting)
    result_ndim = tensors[0].ndim + 1
    axis = _util.normalize_axis_index(axis, result_ndim)
    try:
        result = torch.stack(tensors, axis=axis)
    except RuntimeError as e:
        raise ValueError(*e.args)
    return result


def column_stack(tensors, *, dtype=None, casting="same_kind"):
    tensors = _concat_cast_helper(tensors, dtype=dtype, casting=casting)
    result = torch.column_stack(tensors)
    return result


def dstack(tensors, *, dtype=None, casting="same_kind"):
    tensors = _concat_cast_helper(tensors, dtype=dtype, casting=casting)
    result = torch.dstack(tensors)
    return result


def hstack(tensors, *, dtype=None, casting="same_kind"):
    tensors = _concat_cast_helper(tensors, dtype=dtype, casting=casting)
    result = torch.hstack(tensors)
    return result


def vstack(tensors, *, dtype=None, casting="same_kind"):
    tensors = _concat_cast_helper(tensors, dtype=dtype, casting=casting)
    result = torch.vstack(tensors)
    return result


# #### cov & corrcoef


def corrcoef(xy_tensor, *, dtype=None):
    is_half = dtype == torch.float16
    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32

    xy_tensor = _util.cast_if_needed(xy_tensor, dtype)
    result = torch.corrcoef(xy_tensor)

    if is_half:
        result = result.to(torch.float16)

    return result


def cov(
    m_tensor,
    bias=False,
    ddof=None,
    fweights_tensor=None,
    aweights_tensor=None,
    *,
    dtype=None,
):
    if ddof is None:
        ddof = 1 if bias == 0 else 0

    is_half = dtype == torch.float16
    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32

    m_tensor = _util.cast_if_needed(m_tensor, dtype)

    result = torch.cov(
        m_tensor, correction=ddof, aweights=aweights_tensor, fweights=fweights_tensor
    )

    if is_half:
        result = result.to(torch.float16)

    return result


def meshgrid(*xi_tensors, copy=True, sparse=False, indexing="xy"):
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/function_base.py#L4892-L5047
    ndim = len(xi_tensors)

    if indexing not in ["xy", "ij"]:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1 :]) for i, x in enumerate(xi_tensors)]

    if indexing == "xy" and ndim > 1:
        # switch first and second axis
        output[0] = output[0].reshape((1, -1) + s0[2:])
        output[1] = output[1].reshape((-1, 1) + s0[2:])

    if not sparse:
        # Return the full N-D matrix (not only the 1-D vector)
        output = torch.broadcast_tensors(*output)

    if copy:
        output = [x.clone() for x in output]

    return output


def bincount(x_tensor, /, weights_tensor=None, minlength=0):
    int_dtype = _dtypes_impl.default_int_dtype
    (x_tensor,) = _util.cast_dont_broadcast((x_tensor,), int_dtype, casting="safe")

    result = torch.bincount(x_tensor, weights_tensor, minlength)
    return result


# ### linspace, geomspace, logspace and arange ###


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    if axis != 0 or not endpoint:
        raise NotImplementedError
    base = torch.pow(stop / start, 1.0 / (num - 1))
    logbase = torch.log(base)
    result = torch.logspace(
        torch.log(start) / logbase,
        torch.log(stop) / logbase,
        num,
        base=base,
    )
    return result


def arange(start=None, stop=None, step=1, dtype=None):
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

    if dtype is None:
        dt_list = [_util._coerce_to_tensor(x).dtype for x in (start, stop, step)]
        dtype = _dtypes_impl.default_int_dtype
        dt_list.append(dtype)
        dtype = _dtypes_impl.result_type_impl(dt_list)

    try:
        return torch.arange(start, stop, step, dtype=dtype)
    except RuntimeError:
        raise ValueError("Maximum allowed size exceeded")


# ### empty/full et al ###


def eye(N, M=None, k=0, dtype=float):
    if M is None:
        M = N
    z = torch.zeros(N, M, dtype=dtype)
    z.diagonal(k).fill_(1)
    return z


def zeros_like(a, dtype=None, shape=None):
    result = torch.zeros_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def ones_like(a, dtype=None, shape=None):
    result = torch.ones_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def full_like(a, fill_value, dtype=None, shape=None):
    # XXX: fill_value broadcasts
    result = torch.full_like(a, fill_value, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def empty_like(prototype, dtype=None, shape=None):
    result = torch.empty_like(prototype, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def full(shape, fill_value, dtype=None):
    if dtype is None:
        dtype = fill_value.dtype
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    result = torch.full(shape, fill_value, dtype=dtype)
    return result


# ### shape manipulations ###


def roll(tensor, shift, axis=None):
    if axis is not None:
        axis = _util.normalize_axis_tuple(axis, tensor.ndim, allow_duplicate=True)
        if not isinstance(shift, tuple):
            shift = (shift,) * len(axis)
    result = tensor.roll(shift, axis)
    return result


def squeeze(tensor, axis=None):
    if axis == ():
        result = tensor
    elif axis is None:
        result = tensor.squeeze()
    else:
        result = tensor.squeeze(axis)
    return result


def reshape(tensor, *shape, order="C"):
    if order != "C":
        raise NotImplementedError
    newshape = shape[0] if len(shape) == 1 else shape
    # if sh = (1, 2, 3), numpy allows both .reshape(sh) and .reshape(*sh)
    result = tensor.reshape(newshape)
    return result


def transpose(tensor, *axes):
    # numpy allows both .reshape(sh) and .reshape(*sh)
    axes = axes[0] if len(axes) == 1 else axes
    if axes == () or axes is None:
        axes = tuple(range(tensor.ndim))[::-1]
    try:
        result = tensor.permute(axes)
    except RuntimeError:
        raise ValueError("axes don't match array")
    return result


# ### Numeric ###


def round(tensor, decimals=0):
    if tensor.is_floating_point():
        result = torch.round(tensor, decimals=decimals)
    elif tensor.is_complex():
        # RuntimeError: "round_cpu" not implemented for 'ComplexFloat'
        result = (
            torch.round(tensor.real, decimals=decimals)
            + torch.round(tensor.imag, decimals=decimals) * 1j
        )
    else:
        # RuntimeError: "round_cpu" not implemented for 'int'
        result = tensor
    return result
