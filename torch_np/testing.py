import torch

from ._ndarray import asarray_replacer
import torch_np as np

@asarray_replacer("two")
def assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=True,
                    err_msg='', verbose=True, check_dtype=True):
    result = torch.testing.assert_close(actual, desired, atol=atol, rtol=rtol,
                                        check_dtype=check_dtype)
    return True


@asarray_replacer("two")
def assert_equal(actual, desired):
    eq = np.all(actual == desired)
    if not eq:
        raise AssertionError('not equal')
    return eq

assert_array_equal = assert_equal
