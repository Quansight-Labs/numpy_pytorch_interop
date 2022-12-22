"""
Poking around ufunc casting/broadcasting/dtype/out behavior.

The goal is to validate on numpy, and tests should work when replacing
>>> import numpy as no

by 
>>> import torch_np as np
"""
import pytest
from pytest import raises as assert_raises

#import numpy as np
import torch_np as np

assert_equal = np.testing.assert_equal


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



parametrize_binary_ufuncs = pytest.mark.parametrize('ufunc', [np.add]) #, np.logaddexp, np.hypot])

class TestBinaryUfuncs:

    def get_xy(self, ufunc):
        return np.arange(5, dtype='float64'), np.arange(8, 13, dtype='float64')

    @parametrize_binary_ufuncs
    def test_scalar(self, ufunc):
        # check that ufunc accepts a scalar and the result is convertible to scalar
        xy = self.get_xy(ufunc)
        x, y = xy[0][0], xy[1][0]
        float(ufunc(x, y))

    @parametrize_casting
    @parametrize_binary_ufuncs
    @pytest.mark.parametrize('out_dtype', ['float64', 'complex128', 'float32'])
    def test_xy_and_out_casting(self, ufunc, casting, out_dtype):
        x, y = self.get_xy(ufunc)
        out = np.empty_like(x, dtype=out_dtype)

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


#### This is a template for a unary ufunc to pass the above sorts of test

'''
def ufunc(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or not where:
        raise NotImplementedError
    # XXX: dtype=... parameter is silently ignored

    x_array = asarray(x)
    x_tensor = x_array.get()

    if out is not None:
       if not isinstance(out, np.ndarray):
            raise TypeError("Return arrays must be of ArrayType")

        out_tensor = out.get()
        # check dtypes of x and out
        if not np.can_cast(x_array.dtype, out.dtype, casting=casting):
            raise TypeError(f"Cannot cast array data from {ar.dtype} to"
                             " {out_dtype} according to the rule '{casting}'")            

        # `out` broadcasts `x`
        if x_array.shape != out.shape:
            x_tensor = torch.broadcast_to(x_tensor, out.shape)

        # cast x if needed
        if x_array.dtype != out.dtype:
            x_tensor = x_tensor.to(_dtype.torch_dtype_from(out.dtype))

    # result = {torch_stanza}
    if out is not None:
        torch.ufunc(x_tensor, out=out_tensor)
        return out
    else:
        result = torch.ufunc(x_tensor)
        return asarray(result)
'''
