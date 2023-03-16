import torch

from ._detail import _dtypes_impl, _util


def ufunc_preprocess(
    tensors, out, where, casting, order, dtype, subok, signature, extobj
):
    """
    Notes
    -----
    The `out` array broadcasts `tensors`, but not vice versa.
    """
    # internal preprocessing or args in ufuncs (cf _unary_ufuncs, _binary_ufuncs)
    if order != "K" or not where or signature or extobj:
        raise NotImplementedError

    # dtype of the result: depends on both dtype=... and out=... arguments
    if dtype is None:
        out_dtype = None if out is None else out.dtype.torch_dtype
    else:
        out_dtype = (
            dtype
            if out is None
            else _dtypes_impl.result_type_impl([dtype, out.dtype.torch_dtype])
        )

    if out_dtype:
        tensors = _util.typecast_tensors(tensors, out_dtype, casting)
    return tensors


# ### Return helpers: wrap a single tensor, a tuple of tensors, out= etc ###


def result_or_out(result_tensor, out_array=None, promote_scalar=False):
    """A helper for returns with out= argument.

    If `promote_scalar is True`, then:
        if result_tensor.numel() == 1 and out is zero-dimensional,
            result_tensor is placed into the out array.
    This weirdness is used e.g. in `np.percentile`
    """
    if out_array is not None:
        if result_tensor.shape != out_array.shape:
            can_fit = result_tensor.numel() == 1 and out_array.ndim == 0
            if promote_scalar and can_fit:
                result_tensor = result_tensor.squeeze()
            else:
                raise ValueError(
                    f"Bad size of the out array: out.shape = {out_array.shape}"
                    f" while result.shape = {result_tensor.shape}."
                )
        out_tensor = out_array.get()
        out_tensor.copy_(result_tensor)
        return out_array
    else:
        return array_from(result_tensor)


def array_from(tensor, base=None):
    from ._ndarray import ndarray

    return ndarray._from_tensor(tensor)


def tuple_arrays_from(result):
    from ._ndarray import asarray

    return tuple(asarray(x) for x in result)


# ### Various ways of converting array-likes to tensors ###


def ndarrays_to_tensors(*inputs):
    """Convert all ndarrays from `inputs` to tensors. (other things are intact)"""
    from ._ndarray import asarray, ndarray

    if len(inputs) == 0:
        return ValueError()
    elif len(inputs) == 1:
        input_ = inputs[0]
        if isinstance(input_, ndarray):
            return input_.get()
        elif isinstance(input_, tuple):
            result = []
            for sub_input in input_:
                sub_result = ndarrays_to_tensors(sub_input)
                result.append(sub_result)
            return tuple(result)
        else:
            return input_
    else:
        assert isinstance(inputs, tuple)  # sanity check
        return ndarrays_to_tensors(inputs)


def to_tensors(*inputs):
    """Convert all array_likes from `inputs` to tensors."""
    from ._ndarray import asarray, ndarray

    return tuple(asarray(value).get() for value in inputs)
