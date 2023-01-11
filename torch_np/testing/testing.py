import torch

from .._ndarray import asarray
import torch_np as np

def assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=True,
                    err_msg='', verbose=True, check_dtype=True):
    actual = asarray(actual).get()
    desired = asarray(desired).get()
    result = torch.testing.assert_close(actual, desired, atol=atol, rtol=rtol,
                                        check_dtype=check_dtype)
    return result


def assert_equal(actual, desired):
    """Check `actual == desired`, broadcast if needed """
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    eq = np.all(actual == desired)
    if not eq:
        raise AssertionError('not equal')
    return eq



def assert_array_equal(actual, desired):
    """Check that actual == desired, both shapes and values."""
    a_actual = asarray(actual)
    a_desired = asarray(desired)

    assert a_actual.shape == a_desired.shape
    assert (a_actual == a_desired).all()

