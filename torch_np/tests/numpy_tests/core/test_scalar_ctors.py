"""
Test the scalar constructors, which also do type-coercion
"""
import pytest

import torch_np as np
from torch_np.testing import (
    assert_equal, assert_almost_equal, assert_warns,
    )

class TestFromString:
    def test_floating(self):
        # Ticket #640, floats from string
        fsingle = np.single('1.234')
        fdouble = np.double('1.234')
        assert_almost_equal(fsingle, 1.234)
        assert_almost_equal(fdouble, 1.234)

    def test_floating_overflow(self):
        """ Strings containing an unrepresentable float overflow """
        fhalf = np.half('1e10000')
        assert_equal(fhalf, np.inf)
        fsingle = np.single('1e10000')
        assert_equal(fsingle, np.inf)
        fdouble = np.double('1e10000')
        assert_equal(fdouble, np.inf)

        fhalf = np.half('-1e10000')
        assert_equal(fhalf, -np.inf)
        fsingle = np.single('-1e10000')
        assert_equal(fsingle, -np.inf)
        fdouble = np.double('-1e10000')
        assert_equal(fdouble, -np.inf)


    def test_bool(self):
        with pytest.raises(TypeError):
            np.bool_(False, garbage=True)


class TestFromInt:
    def test_intp(self):
        # Ticket #99
        assert_equal(1024, np.intp(1024))

    def test_uint64_from_negative(self):
        with pytest.warns(DeprecationWarning):
            assert_equal(np.uint64(-2), np.uint64(18446744073709551614))


int_types = [np.byte, np.short, np.intc, np.int_, np.longlong]
uint_types = [np.ubyte]
float_types = [np.half, np.single, np.double]
cfloat_types = [np.csingle, np.cdouble]


class TestArrayFromScalar:
    """ gh-15467 """

    def _do_test(self, t1, t2):
        x = t1(2)
        arr = np.array(x, dtype=t2)
        # type should be preserved exactly
        if t2 is None:
            assert arr.dtype.type is t1
        else:
            assert arr.dtype.type is t2

    @pytest.mark.parametrize('t1', int_types + uint_types)
    @pytest.mark.parametrize('t2', int_types + uint_types + [None])
    def test_integers(self, t1, t2):
        return self._do_test(t1, t2)

    @pytest.mark.parametrize('t1', float_types)
    @pytest.mark.parametrize('t2', float_types + [None])
    def test_reals(self, t1, t2):
        return self._do_test(t1, t2)

    @pytest.mark.parametrize('t1', cfloat_types)
    @pytest.mark.parametrize('t2', cfloat_types + [None])
    def test_complex(self, t1, t2):
        return self._do_test(t1, t2)

