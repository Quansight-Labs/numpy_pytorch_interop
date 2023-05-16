"""Test examples for NEP 50."""

import torch_np as tnp
from torch_np import (
    array,
    bool_,
    complex64,
    complex128,
    float32,
    float64,
    inf,
    int16,
    int64,
    uint8,
)
from torch_np.testing import assert_allclose

uint16 = uint8  # can be anything here, see below


# from numpy import array, uint8, uint16, int64, float32, float64, inf
# from numpy.testing import assert_allclose
# import numpy as np
# np._set_promotion_state('weak')

import pytest
from pytest import raises as assert_raises

unchanged = None

# expression    old result   new_result
examples = {
    "uint8(1) + 2": (int64(3), uint8(3)),
    "array([1], uint8) + int64(1)": (array([2], uint8), array([2], int64)),
    "array([1], uint8) + array(1, int64)": (array([2], uint8), array([2], int64)),
    "array([1.], float32) + float64(1.)": (
        array([2.0], float32),
        array([2.0], float64),
    ),
    "array([1.], float32) + array(1., float64)": (
        array([2.0], float32),
        array([2.0], float64),
    ),
    "array([1], uint8) + 1": (array([2], uint8), unchanged),
    "array([1], uint8) + 200": (array([201], uint8), unchanged),
    "array([100], uint8) + 200": (array([44], uint8), unchanged),
    "array([1], uint8) + 300": (array([301], uint16), Exception),
    "uint8(1) + 300": (int64(301), Exception),
    "uint8(100) + 200": (int64(301), uint8(44)),  # and RuntimeWarning
    "float32(1) + 3e100": (float64(3e100), float32(inf)),  # and RuntimeWarning [T7]
    "array([0.1], float32) == 0.1": (
        array([False]),
        unchanged,
    ),  # XXX: a typo in NEP50?
    "array([0.1], float32) == float64(0.1)": (array([True]), array([False])),
    "array([1.], float32) + 3": (array([4.0], float32), unchanged),
    "array([1.], float32) + int64(3)": (array([4.0], float32), array([4.0], float64)),
    # additional examples from the NEP text
    "int16(2) + 2": (int64(4), int16(4)),
    "int16(4) + 4j": (complex128(4 + 4j), unchanged),
    "float32(5) + 5j": (complex128(5 + 5j), complex64(5 + 5j)),
    "bool_(True) + 1": (int64(2), unchanged),
    "True + uint8(2)": (uint8(3), unchanged),
}


fails = [
    "array([1], uint8) + 300",
    "uint8(1) + 300",
    "array([0.1], float32) == 0.1",  # TODO: fix the example
    "float32(5) + 5j",  # TODO: implement
]


@pytest.mark.parametrize("example", examples)
def test_nep50_exceptions(example):

    if example in fails:
        pytest.xfail(reason="scalars")

    old, new = examples[example]

    if new == Exception:
        with assert_raises(OverflowError):
            eval(example)

    else:
        result = eval(example)

        if new is unchanged:
            new = old

        assert_allclose(result, new, atol=1e-16)
        assert result.dtype == new.dtype


class TestScalarsWeakTyping:
    def test_asarray_scalars(self):
        assert tnp.asarray(3).tensor.is_weakly_typed is False

    def test_asarray_asarray_scalars(self):
        a = tnp.asarray(3)
        assert tnp.asarray(a).tensor.is_weakly_typed is False

    def test_scalar_scalar(self):
        a = tnp.uint8(3)
        is_weakly_typed = getattr(a.tensor, "is_weakly_typed", False)
        assert is_weakly_typed is False
