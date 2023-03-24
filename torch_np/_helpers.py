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


def ndarrays_to_tensors(*inputs):
    """Convert all ndarrays from `inputs` to tensors. (other things are intact)"""
    from ._ndarray import asarray, ndarray

    if len(inputs) == 0:
        return ValueError()
    elif len(inputs) == 1:
        input_ = inputs[0]
        if isinstance(input_, ndarray):
            return input_.tensor
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
