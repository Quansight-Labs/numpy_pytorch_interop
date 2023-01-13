"""
Poking around ufunc casting/broadcasting/dtype/out behavior.

The goal is to validate on numpy, and tests should work when replacing
>>> import numpy as no

by 
>>> import torch_np as np
"""
import operator

import pytest
from pytest import raises as assert_raises

import torch_np as np
from torch_np.testing import assert_equal

#import numpy as np
#from numpy.testing import assert_equal


parametrize_unary_ufuncs = pytest.mark.parametrize('ufunc', [np.sin])
parametrize_casting = pytest.mark.parametrize("casting",
            ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])

class TestUnaryUfuncs:

    def get_x(self, ufunc):
        return np.arange(5, dtype='float64')

    @parametrize_unary_ufuncs
    def test_scalar(self, ufunc):
        # check that ufunc accepts a scalar and the result is convertible to scalar
        x = self.get_x(ufunc)[0]
        float(ufunc(x))

    @pytest.mark.skip(reason='XXX: unary ufuncs ignore the dtype=... parameter')
    @parametrize_unary_ufuncs
    def test_x_and_dtype(self, ufunc):
        x = self.get_x(ufunc)
        res = ufunc(x, dtype='float')
        assert res.dtype == np.dtype('float')

    @pytest.mark.skip(reason='XXX: unary ufuncs ignore the dtype=... parameter')
    @parametrize_casting
    @parametrize_unary_ufuncs
    @pytest.mark.parametrize('dtype', ['float64', 'complex128', 'float32'])
    def test_x_and_dtype_casting(self, ufunc, casting, dtype):
        x = self.get_x(ufunc)
        if not np.can_cast(x, dtype, casting=casting):
            with assert_raises(TypeError):
                ufunc(x, dtype=dtype, casting=casting)
        else:
            assert ufunc(x, dtype=dtype, casting=casting).dtype == dtype

    @parametrize_casting
    @parametrize_unary_ufuncs
    @pytest.mark.parametrize('out_dtype', ['float64', 'complex128', 'float32'])
    def test_x_and_out_casting(self, ufunc, casting, out_dtype):
        x = self.get_x(ufunc)
        out = np.empty_like(x, dtype=out_dtype)
        if not np.can_cast(x, out_dtype, casting=casting):
            with assert_raises(TypeError):
                ufunc(x, out=out, casting=casting)
        else:
            result = ufunc(x, out=out, casting=casting)
            assert result.dtype == out_dtype
            assert result is out

    @parametrize_unary_ufuncs
    def test_x_and_out_broadcast(self, ufunc):
        x = self.get_x(ufunc)
        out = np.empty((x.shape[0], x.shape[0]))

        x_b = np.broadcast_to(x, out.shape)

        res_out = ufunc(x, out=out)
        res_bcast = ufunc(x_b)
        assert_equal(res_out, res_bcast)
        assert res_out is out

        out = np.empty((1, x.shape[0]))
        x_b = np.broadcast_to(x, out.shape)

        res_out = ufunc(x, out=out)
        res_bcast = ufunc(x_b)
        assert_equal(res_out, res_bcast)
        assert res_out is out



ufunc_op_iop_numeric = [
    (np.add, operator.__add__, operator.__iadd__),
    (np.subtract, operator.__sub__, operator.__isub__),
    (np.multiply, operator.__mul__, operator.__imul__),
    (np.divide, operator.__truediv__, operator.__itruediv__),
    (np.floor_divide, operator.__floordiv__, operator.__ifloordiv__),
    (np.float_power, operator.__pow__, operator.__ipow__),
 ##   (np.remainder, operator.__mod__, operator.__imod__),   # does not handle complex


# remainder vs fmod?
# pow vs power vs float_power
]

ufuncs_with_dunders = [ufunc for ufunc, _, _ in ufunc_op_iop_numeric]
numeric_binary_ufuncs = [np.float_power, np.power,]

# these are not implemented for complex inputs
no_complex = [np.floor_divide, np.hypot, np.arctan2, np.copysign, np.fmax,
        np.fmin, np.fmod, np.heaviside, np.logaddexp, np.logaddexp2,
        np.maximum, np.minimum,
]

parametrize_binary_ufuncs = pytest.mark.parametrize(
        'ufunc', ufuncs_with_dunders + numeric_binary_ufuncs + no_complex)



# TODO: these snowflakes need special handling
"""
 'bitwise_and',
 'bitwise_or',
 'bitwise_xor',
 'equal',
 'lcm',
 'ldexp',
 'left_shift',
 'less',
 'less_equal',
 'gcd',
 'greater',
 'greater_equal',
 'logical_and',
 'logical_or',
 'logical_xor',
 'matmul',
 'not_equal',
"""



class TestBinaryUfuncs:

    def get_xy(self, ufunc):
        return np.arange(5, dtype='float64'), np.arange(8, 13, dtype='float64')

    @parametrize_binary_ufuncs
    def test_scalar(self, ufunc):
        # check that ufunc accepts a scalar and the result is convertible to scalar
        xy = self.get_xy(ufunc)
        x, y = xy[0][0], xy[1][0]
        float(ufunc(x, y))

    @parametrize_binary_ufuncs
    def test_vector_vs_scalar(self, ufunc):
        x, y = self.get_xy(ufunc)
        assert_equal(ufunc(x, y), [ufunc(a, b) for a, b in zip(x, y)])

    @parametrize_casting
    @parametrize_binary_ufuncs
    @pytest.mark.parametrize('out_dtype', ['float64', 'complex128', 'float32'])
    def test_xy_and_out_casting(self, ufunc, casting, out_dtype):
        x, y = self.get_xy(ufunc)
        out = np.empty_like(x, dtype=out_dtype)

        if ufunc in no_complex and np.issubdtype(out_dtype, np.complexfloating):
            pytest.skip(f'{ufunc} does not accept complex.')

        can_cast_x = np.can_cast(x, out_dtype, casting=casting)
        can_cast_y = np.can_cast(y, out_dtype, casting=casting)

        if not(can_cast_x and can_cast_y):
            with assert_raises(TypeError):
                ufunc(x, out=out, casting=casting)
        else:
            result = ufunc(x, y, out=out, casting=casting)
            assert result.dtype == out_dtype
            assert result is out

    @parametrize_binary_ufuncs
    def test_xy_and_out_broadcast(self, ufunc):
        x, y = self.get_xy(ufunc)
        y = y[:, None]
        out = np.empty((2, y.shape[0], x.shape[0]))

        x_b = np.broadcast_to(x, out.shape)
        y_b = np.broadcast_to(y, out.shape)

        res_out = ufunc(x, y, out=out)
        res_bcast = ufunc(x_b, y_b)

        assert_equal(res_out, res_bcast)
        assert res_out is out


dtypes_numeric = [np.int32, np.float32, np.float64, np.complex128]


class TestNdarrayDunderVsUfunc:
    """Test ndarray dunders which delegate to ufuncs, vs ufuncs."""

    @pytest.mark.parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    def test_basic(self, ufunc, op, iop):
        """basic op/rop/iop, no dtypes, no broadcasting"""

        # __add__
        a = np.array([1, 2, 3])
        assert_equal(op(a, 1), ufunc(a, 1))
        assert_equal(op(a, a.tolist()), ufunc(a, a.tolist()))
        assert_equal(op(a, a), ufunc(a, a))

        # __radd__
        a = np.array([1, 2, 3])
        assert_equal(op(1, a), ufunc(a, 1))
        assert_equal(op(a.tolist(), a), ufunc(a, a.tolist()))

        # __iadd__
        a0 = np.array([2, 4, 6])
        a = a0.copy()

        iop(a, 2)     # modifies a in-place
        assert_equal(a, op(a0, 2))

        a0 = np.array([2, 4, 6])
        a = a0.copy()
        iop(a, a)
        assert_equal(a, op(a0, a0))

    @pytest.mark.parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    @pytest.mark.parametrize("other_dtype", dtypes_numeric)
    def test_other_scalar(self, ufunc, op, iop, other_dtype):
        """Test op/iop/rop when the other argument is a scalar of a different dtype."""
        a = np.array([1, 2, 3])
        b = other_dtype(3)

        if ufunc in no_complex and issubclass(other_dtype, np.complexfloating):
            pytest.skip(f'{ufunc} does not accept complex.')

        # __op__
        result = op(a, b)
        assert_equal(result, ufunc(a, b))
        assert result.dtype == np.result_type(a, b)

        # __rop__
        result = op(b, a)
        assert_equal(result, ufunc(b, a))
        assert result.dtype == np.result_type(a, b)

        # __iop__ : casts the result to self.dtype, raises if cannot
        can_cast = np.can_cast(np.result_type(a.dtype, other_dtype),
                               a.dtype,
                               casting="same_kind")
        if can_cast:
            a0 = a.copy()
            result = iop(a, b)
            assert_equal(result, ufunc(a0, b))
            assert result.dtype == np.result_type(a0, b)
        else:
            with assert_raises((TypeError, RuntimeError)):    # XXX np.UFuncTypeError
                iop(a, b)


    @pytest.mark.parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    @pytest.mark.parametrize("other_dtype", dtypes_numeric)
    def test_other_array(self, ufunc, op, iop, other_dtype):
        """Test op/iop/rop when the other argument is an array of a different dtype."""
        a = np.array([1, 2, 3])
        b = np.array([5, 6, 7], dtype=other_dtype)

        if ufunc in no_complex and issubclass(other_dtype, np.complexfloating):
            pytest.skip(f'{ufunc} does not accept complex.')

        # __op__
        result = op(a, b)
        assert_equal(result, ufunc(a, b))
        assert result.dtype == np.result_type(a, b)

        # __rop__(other array)
        result = op(b, a)
        assert_equal(result, ufunc(b, a))
        assert result.dtype == np.result_type(a, b)

        # __iop__
        can_cast = np.can_cast(np.result_type(a.dtype, other_dtype),
                               a.dtype,
                               casting="same_kind")
        if can_cast:
            a0 = a.copy()
            result = iop(a, b)
            assert_equal(result, ufunc(a0, b))
            assert result.dtype == np.result_type(a0, b)
        else:
            with assert_raises((TypeError, RuntimeError)):    # XXX np.UFuncTypeError
                iop(a, b)


    @pytest.mark.parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    def test_other_array_bcast(self, ufunc, op, iop):
        """Test op/rop/iop with broadcasting """
        # __op__
        a = np.array([1, 2, 3])
        result_op = op(a, a[:, None])
        result_ufunc = ufunc(a, a[:, None])
        assert result_op.shape == result_ufunc.shape
        assert result_op.dtype == result_ufunc.dtype
        assert_equal(result_op, result_ufunc)

        # __rop__
        a = np.array([1, 2, 3])
        result_op = op(a[:, None], a)
        result_ufunc = ufunc(a[:, None], a)
        assert result_op.shape == result_ufunc.shape
        assert result_op.dtype == result_ufunc.dtype
        assert_equal(result_op, result_ufunc)

        # __iop__ : in-place ops (`self += other` etc) do not broadcast self
        b = a[:, None].copy()
        with assert_raises((ValueError, RuntimeError)):    # XXX ValueError in numpy
            iop(a, b)

        # however, `self += other` broadcasts other
        aa = np.broadcast_to(a, (3, 3)).copy()
        aa0 = aa.copy()

        result = iop(aa, a)
        result_ufunc = ufunc(aa0, a)

        assert result.shape == result_ufunc.shape
        assert result.dtype == result_ufunc.dtype
        assert_equal(result, result_ufunc)

