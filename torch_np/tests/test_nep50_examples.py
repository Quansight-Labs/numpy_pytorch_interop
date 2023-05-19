"""Test examples for NEP 50."""

import itertools

try:
    import numpy as _np

    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

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
    "array([0.1], float32) == 0.1",  # TODO: fix the example
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


# ### Directly compare to numpy ###

weaks = (True, 1, 2.0, 3j)
non_weaks = (
    tnp.asarray(True),
    tnp.uint8(1),
    tnp.int8(1),
    tnp.int32(1),
    tnp.int64(1),
    tnp.float32(1),
    tnp.float64(1),
    tnp.complex64(1),
    tnp.complex128(1),
)
if HAVE_NUMPY:
    dtypes = (
        None,
        _np.bool_,
        _np.uint8,
        _np.int8,
        _np.int32,
        _np.int64,
        _np.float32,
        _np.float64,
        _np.complex64,
        _np.complex128,
    )
else:
    dtypes = (None,)


@pytest.mark.skipif(not HAVE_NUMPY, reason="NumPy not found")
@pytest.mark.parametrize(
    "scalar, array, dtype", itertools.product(weaks, non_weaks, dtypes)
)
def test_direct_compare(scalar, array, dtype):
    # compare to NumPy w/ NEP 50.
    try:
        state = _np._get_promotion_state()
        _np._set_promotion_state("weak")

        if dtype is not None:
            kwargs = {"dtype": dtype}
        try:
            result_numpy = _np.add(scalar, array.tensor.numpy(), **kwargs)
        except Exception:
            return

        kwargs = {}
        if dtype is not None:
            kwargs = {"dtype": getattr(tnp, dtype.__name__)}
        result = tnp.add(scalar, array, **kwargs).tensor.numpy()
        assert result.dtype == result_numpy.dtype
        assert result == result_numpy

    finally:
        _np._set_promotion_state(state)


# ufunc name: [array.dtype]
corners = {
    "true_divide": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "divide": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "arctan2": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "copysign": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "heaviside": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "ldexp": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "power": ["uint8"],
    "nextafter": ["float32"],
}


@pytest.mark.skipif(not HAVE_NUMPY, reason="NumPy not found")
@pytest.mark.parametrize("name", tnp._ufuncs._binary)
@pytest.mark.parametrize("scalar, array", itertools.product(weaks, non_weaks))
def test_compare_ufuncs(name, scalar, array):

    if name in corners and (
        array.dtype.name in corners[name]
        or tnp.asarray(scalar).dtype.name in corners[name]
    ):
        return pytest.skip(f"{name}(..., dtype=array.dtype)")

    try:
        state = _np._get_promotion_state()
        _np._set_promotion_state("weak")

        if name in ["matmul", "modf", "divmod"]:
            return
        ufunc = getattr(tnp, name)
        ufunc_numpy = getattr(_np, name)

        try:
            result = ufunc(scalar, array)
        except RuntimeError:
            # RuntimeError: "bitwise_xor_cpu" not implemented for 'ComplexDouble' etc
            result = None

        try:
            result_numpy = ufunc_numpy(scalar, array.tensor.numpy())
        except TypeError:
            # TypeError: ufunc 'hypot' not supported for the input types
            result_numpy = None

        if result is not None and result_numpy is not None:
            assert result.tensor.numpy().dtype == result_numpy.dtype

    finally:
        _np._set_promotion_state(state)
