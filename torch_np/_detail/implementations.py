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


def tensor_isclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    dtype = _dtypes_impl.result_type_impl((a.dtype, b.dtype))
    a = a.to(dtype)
    b = b.to(dtype)
    result = torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return result


# ### is arg real or complex valued ###


def tensor_iscomplex(x):
    if torch.is_complex(x):
        return torch.as_tensor(x).imag != 0
    result = torch.zeros_like(x, dtype=torch.bool)
    if result.ndim == 0:
        result = result.item()
    return result


def tensor_isreal(x):
    if torch.is_complex(x):
        return torch.as_tensor(x).imag == 0
    result = torch.zeros_like(x, dtype=torch.bool)
    if result.ndim == 0:
        result = result.item()
    return result


def tensor_real_if_close(x, tol=100):
    if not torch.is_complex(x):
        return x
    mask = torch.abs(x.imag) < tol * torch.finfo(x.dtype).eps
    if mask.all():
        return x.real
    else:
        return x


# ### math functions ###


def tensor_angle(z, deg=False):
    result = torch.angle(z)
    if deg:
        result *= 180 / torch.pi
    return result


# ### splits ###


def split_helper(tensor, indices_or_sections, axis, strict=False):
    if isinstance(indices_or_sections, int):
        return split_helper_int(tensor, indices_or_sections, axis, strict)
    elif isinstance(indices_or_sections, (list, tuple)):
        return split_helper_list(tensor, list(indices_or_sections), axis, strict)
    else:
        raise TypeError("split_helper: ", type(indices_or_sections))


def split_helper_int(tensor, indices_or_sections, axis, strict=False):
    if not isinstance(indices_or_sections, int):
        raise NotImplementedError("split: indices_or_sections")

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


def split_helper_list(tensor, indices_or_sections, axis, strict=False):
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


def concatenate(tensors, axis=0, out=None, dtype=None, casting="same_kind"):
    # np.concatenate ravels if axis=None
    tensors, axis = _util.axis_none_ravel(*tensors, axis=axis)

    if out is not None or dtype is not None:
        # figure out the type of the inputs and outputs
        out_dtype = out.dtype.torch_dtype if dtype is None else dtype

        # cast input arrays if necessary; do not broadcast them agains `out`
        tensors = _util.cast_dont_broadcast(tensors, out_dtype, casting)

    try:
        result = torch.cat(tensors, axis)
    except (IndexError, RuntimeError):
        raise _util.AxisError

    return result


# #### cov & corrcoef


def corrcoef(xy_tensor, rowvar=True, *, dtype=None):
    if rowvar is False:
        # xy_tensor is at least 2D, so using .T is safe
        xy_tensor = x_tensor.T

    is_half = dtype == torch.float16
    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32

    if dtype is not None:
        xy_tensor = xy_tensor.to(dtype)

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

    if dtype is not None:
        m_tensor = m_tensor.to(dtype)

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
