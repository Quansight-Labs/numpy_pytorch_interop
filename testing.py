import torch

from _ndarray import asarray, asarray_replacer_2

@asarray_replacer_2
def assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=True,
                    err_msg='', verbose=True, check_dtype=True):
    result = torch.testing.assert_close(actual, desired, atol=atol, rtol=rtol,
                                        check_dtype=check_dtype)
    return True
