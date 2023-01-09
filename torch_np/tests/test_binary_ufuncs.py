import numpy as np
import torch

from ..testing import assert_allclose


def test_add():
    assert_allclose(np.add(0.5, 0.6),
                               np.add(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_arctan2():
    assert_allclose(np.arctan2(0.5, 0.6),
                               np.arctan2(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_bitwise_and():
    assert_allclose(np.bitwise_and(5, 6),
                               np.bitwise_and(5, 6), atol=1e-7, check_dtype=False)



def test_bitwise_or():
    assert_allclose(np.bitwise_or(5, 6),
                               np.bitwise_or(5, 6), atol=1e-7, check_dtype=False)



def test_bitwise_xor():
    assert_allclose(np.bitwise_xor(5, 6),
                               np.bitwise_xor(5, 6), atol=1e-7, check_dtype=False)



def test_copysign():
    assert_allclose(np.copysign(0.5, 0.6),
                               np.copysign(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_divide():
    assert_allclose(np.divide(0.5, 0.6),
                               np.divide(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_equal():
    assert_allclose(np.equal(0.5, 0.6),
                               np.equal(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_float_power():
    assert_allclose(np.float_power(0.5, 0.6),
                               np.float_power(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_floor_divide():
    assert_allclose(np.floor_divide(0.5, 0.6),
                               np.floor_divide(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_fmax():
    assert_allclose(np.fmax(0.5, 0.6),
                               np.fmax(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_fmin():
    assert_allclose(np.fmin(0.5, 0.6),
                               np.fmin(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_fmod():
    assert_allclose(np.fmod(0.5, 0.6),
                               np.fmod(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_gcd():
    assert_allclose(np.gcd(5, 6),
                               np.gcd(5, 6), atol=1e-7, check_dtype=False)



def test_greater():
    assert_allclose(np.greater(0.5, 0.6),
                               np.greater(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_greater_equal():
    assert_allclose(np.greater_equal(0.5, 0.6),
                               np.greater_equal(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_heaviside():
    assert_allclose(np.heaviside(0.5, 0.6),
                               np.heaviside(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_hypot():
    assert_allclose(np.hypot(0.5, 0.6),
                               np.hypot(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_lcm():
    assert_allclose(np.lcm(5, 6),
                               np.lcm(5, 6), atol=1e-7, check_dtype=False)



def test_left_shift():
    assert_allclose(np.left_shift(5, 6),
                               np.left_shift(5, 6), atol=1e-7, check_dtype=False)



def test_less():
    assert_allclose(np.less(0.5, 0.6),
                               np.less(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_less_equal():
    assert_allclose(np.less_equal(0.5, 0.6),
                               np.less_equal(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_logaddexp():
    assert_allclose(np.logaddexp(0.5, 0.6),
                               np.logaddexp(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_logaddexp2():
    assert_allclose(np.logaddexp2(0.5, 0.6),
                               np.logaddexp2(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_logical_and():
    assert_allclose(np.logical_and(0.5, 0.6),
                               np.logical_and(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_logical_or():
    assert_allclose(np.logical_or(0.5, 0.6),
                               np.logical_or(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_logical_xor():
    assert_allclose(np.logical_xor(0.5, 0.6),
                               np.logical_xor(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_maximum():
    assert_allclose(np.maximum(0.5, 0.6),
                               np.maximum(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_minimum():
    assert_allclose(np.minimum(0.5, 0.6),
                               np.minimum(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_remainder():
    assert_allclose(np.remainder(0.5, 0.6),
                               np.remainder(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_multiply():
    assert_allclose(np.multiply(0.5, 0.6),
                               np.multiply(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_nextafter():
    assert_allclose(np.nextafter(0.5, 0.6),
                               np.nextafter(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_not_equal():
    assert_allclose(np.not_equal(0.5, 0.6),
                               np.not_equal(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_power():
    assert_allclose(np.power(0.5, 0.6),
                               np.power(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_remainder():
    assert_allclose(np.remainder(0.5, 0.6),
                               np.remainder(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_right_shift():
    assert_allclose(np.right_shift(5, 6),
                               np.right_shift(5, 6), atol=1e-7, check_dtype=False)



def test_subtract():
    assert_allclose(np.subtract(0.5, 0.6),
                               np.subtract(0.5, 0.6), atol=1e-7, check_dtype=False)



def test_divide():
    assert_allclose(np.divide(0.5, 0.6),
                               np.divide(0.5, 0.6), atol=1e-7, check_dtype=False)

