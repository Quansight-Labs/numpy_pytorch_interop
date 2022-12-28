import pytest
from pytest import raises as assert_raises

import torch_np as np
from torch_np.testing import assert_equal, assert_array_equal

import torch_np._util as _util

class TestNonzeroAndCountNonzero:
    def test_nonzero_trivial(self):
        assert_equal(np.count_nonzero(np.array([])), 0)
        assert_equal(np.count_nonzero(np.array([], dtype='?')), 0)
        assert_equal(np.nonzero(np.array([])), ([],))

        assert_equal(np.count_nonzero(np.array([0])), 0)
        assert_equal(np.count_nonzero(np.array([0], dtype='?')), 0)
        assert_equal(np.nonzero(np.array([0])), ([],))

        assert_equal(np.count_nonzero(np.array([1])), 1)
        assert_equal(np.count_nonzero(np.array([1], dtype='?')), 1)
        assert_equal(np.nonzero(np.array([1])), ([0],))

        assert isinstance(np.count_nonzero([]), np.ndarray)

    def test_nonzero_zerod(self):
        assert_equal(np.count_nonzero(np.array(0)), 0)
        assert_equal(np.count_nonzero(np.array(0, dtype='?')), 0)

        assert_equal(np.count_nonzero(np.array(1)), 1)
        assert_equal(np.count_nonzero(np.array(1, dtype='?')), 1)

        assert isinstance(np.count_nonzero(np.array(1)), np.ndarray)

    def test_nonzero_onedim(self):
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))

        assert isinstance(np.count_nonzero(x), np.ndarray)

    def test_nonzero_twodim(self):
        x = np.array([[0, 1, 0], [2, 0, 3]])
        assert_equal(np.count_nonzero(x.astype('i1')), 3)
        assert_equal(np.count_nonzero(x.astype('i2')), 3)
        assert_equal(np.count_nonzero(x.astype('i4')), 3)
        assert_equal(np.count_nonzero(x.astype('i8')), 3)
        assert_equal(np.nonzero(x), ([0, 1, 1], [1, 0, 2]))

        x = np.eye(3)
        assert_equal(np.count_nonzero(x.astype('i1')), 3)
        assert_equal(np.count_nonzero(x.astype('i2')), 3)
        assert_equal(np.count_nonzero(x.astype('i4')), 3)
        assert_equal(np.count_nonzero(x.astype('i8')), 3)
        assert_equal(np.nonzero(x), ([0, 1, 2], [0, 1, 2]))

    def test_sparse(self):
        # test special sparse condition boolean code path
        for i in range(20):
            c = np.zeros(200, dtype=bool)
            c[i::20] = True
            assert_equal(np.nonzero(c)[0], np.arange(i, 200 + i, 20))

            c = np.zeros(400, dtype=bool)
            c[10 + i:20 + i] = True
            c[20 + i*2] = True
            assert_equal(np.nonzero(c)[0],
                         np.concatenate((np.arange(10 + i, 20 + i), [20 + i*2])))

    def test_count_nonzero_axis(self):
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        expected = np.array([1, 1, 1, 1, 1])
        assert_equal(np.count_nonzero(m, axis=0), expected)

        expected = np.array([2, 3])
        assert_equal(np.count_nonzero(m, axis=1), expected)

        assert isinstance(np.count_nonzero(m, axis=1), np.ndarray)

        assert_raises(ValueError, np.count_nonzero, m, axis=(1, 1))
        assert_raises(TypeError, np.count_nonzero, m, axis='foo')
        assert_raises(np.AxisError, np.count_nonzero, m, axis=3)
        assert_raises(TypeError, np.count_nonzero,
                      m, axis=np.array([[1], [2]]))

    @pytest.mark.parametrize('typecode', np.typecodes['All'])
    def test_count_nonzero_axis_all_dtypes(self, typecode):
        # More thorough test that the axis argument is respected
        # for all dtypes and responds correctly when presented with
        # either integer or tuple arguments for axis

        m = np.zeros((3, 3), dtype=typecode)
        n = np.ones(1, dtype=typecode)

        m[0, 0] = n[0]
        m[1, 0] = n[0]

        expected = np.array([2, 0, 0], dtype=np.intp)
        result = np.count_nonzero(m, axis=0)
        assert_equal(result, expected)
        assert expected.dtype == result.dtype

        expected = np.array([1, 1, 0], dtype=np.intp)
        result = np.count_nonzero(m, axis=1)
        assert_equal(result, expected)
        assert expected.dtype == result.dtype

        expected = np.array(2)
        assert_equal(np.count_nonzero(m, axis=(0, 1)),
                     expected)
        assert_equal(np.count_nonzero(m, axis=None),
                     expected)
        assert_equal(np.count_nonzero(m),
                     expected)

    def test_countnonzero_axis_empty(self):
        a = np.array([[0, 0, 1], [1, 0, 1]])
        assert_equal(np.count_nonzero(a, axis=()), a.astype(bool))

    def test_countnonzero_keepdims(self):
        a = np.array([[0, 0, 1, 0],
                      [0, 3, 5, 0],
                      [7, 9, 2, 0]])
        assert_equal(np.count_nonzero(a, axis=0, keepdims=True),
                     [[1, 2, 3, 0]])
        assert_equal(np.count_nonzero(a, axis=1, keepdims=True),
                     [[1], [2], [3]])
        assert_equal(np.count_nonzero(a, keepdims=True),
                     [[6]])
        assert isinstance(np.count_nonzero(a, axis=1, keepdims=True), np.ndarray)


class TestFlatnonzero:
    def test_basic(self):
        x = np.arange(-2, 3)
        assert_equal(np.flatnonzero(x),
                     [0, 1, 3, 4])


class TestArgwhere:

    @pytest.mark.parametrize('nd', [0, 1, 2])
    def test_nd(self, nd):
        # get an nd array with multiple elements in every dimension
        x = np.empty((2,)*nd, bool)

        # none
        x[...] = False
        assert_equal(np.argwhere(x).shape, (0, nd))

        # only one
        x[...] = False
        x.ravel()[0] = True
        assert_equal(np.argwhere(x).shape, (1, nd))

        # all but one
        x[...] = True
        x.ravel()[0] = False
        assert_equal(np.argwhere(x).shape, (x.size - 1, nd))

        # all
        x[...] = True
        assert_equal(np.argwhere(x).shape, (x.size, nd))

    def test_2D(self):
        x = np.arange(6).reshape((2, 3))
        assert_array_equal(np.argwhere(x > 1),
                           [[0, 2],
                            [1, 0],
                            [1, 1],
                            [1, 2]])

    def test_list(self):
        assert_equal(np.argwhere([4, 0, 2, 1, 3]), [[0], [2], [3], [4]])


class TestAny:
    def test_basic(self):
        y1 = [0, 0, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 0, 1, 0]
        assert np.any(y1)
        assert np.any(y3)
        assert not np.any(y2)

    def test_nd(self):
        y1 = [[0, 0, 0], [0, 1, 0], [1, 1, 0]]
        assert np.any(y1)
        assert_equal(np.any(y1, axis=0), [1, 1, 0])
        assert_equal(np.any(y1, axis=1), [0, 1, 1])
        assert_equal(np.any(y1), True)
        assert isinstance(np.any(y1, axis=1), np.ndarray)

    # YYY: deduplicate 
    def test_method_vs_function(self):
        y = np.array([[0, 1, 0, 3], [1, 0, 2, 0]])
        assert_equal(np.any(y), y.any())


class TestAll:
    def test_basic(self):
        y1 = [0, 1, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 1, 1, 1]
        assert not np.all(y1)
        assert np.all(y3)
        assert not np.all(y2)
        assert np.all(~np.array(y2))

    def test_nd(self):
        y1 = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        assert not np.all(y1)
        assert_equal(np.all(y1, axis=0), [0, 0, 1])
        assert_equal(np.all(y1, axis=1), [0, 0, 1])
        assert_equal(np.all(y1), False)

    def test_method_vs_function(self):
        y = np.array([[0, 1, 0, 3], [1, 0, 2, 0]])
        assert_equal(np.all(y), y.all())


class _GenericReductionsTestMixin:
    """Run a set of generic tests to verify that self.func acts like a
    reduction operation.

    Specifically, this class checks axis=... and keepdims=... parameters.
    To check the out=... parameter, see the _GenericHasOutTestMixin class below.

    To use: subclass, define self.func and self.allowed_axes.
    """
    def test_bad_axis(self):
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        assert_raises(TypeError, self.func, m, axis='foo')
        assert_raises(np.AxisError, self.func, m, axis=3)
        assert_raises(TypeError, self.func,
                      m, axis=np.array([[1], [2]]))

    def test_axis_empty_generic(self):
        a = np.array([[0, 0, 1], [1, 0, 1]])
        assert_equal(self.func(a, axis=()),
                     self.func(np.expand_dims(a, axis=0), axis=0))

    def test_axis_bad_tuple(self):
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        with assert_raises(ValueError):
             self.func(m, axis=(1, 1))

    def _check_keepdims_generic(self, axis):
        a = np.arange(2*3*4).reshape((2, 3, 4))
        with_keepdims = self.func(a, axis, keepdims=True)
        expanded = np.expand_dims( self.func(a, axis=axis), axis=axis)
        assert_equal(with_keepdims, expanded)

    def test_keepdims_generic(self):
        for axis in self.allowed_axes:
            self._check_keepdims_generic(axis)


class _GenericHasOutTestMixin:
    """Tests for reduction functions which support out=... parameter.
    """
    def test_keepdims_generic_axis_none(self):
        a = np.arange(2*3*4).reshape((2, 3, 4))
        with_keepdims = self.func(a, axis=None, keepdims=True)
        scalar = self.func(a, axis=None)
        expanded = np.full(a.shape, fill_value=scalar)
        assert_equal(with_keepdims, expanded)

    def test_out_scalar(self):
        # out no axis: scalar
        a = np.arange(2*3*4).reshape((2, 3, 4))

        result = self.func(a)
        out = np.empty_like(result)
        result_with_out = self.func(a, out=out)

        assert result_with_out is out
        assert_equal(result, result_with_out)

    def _check_out_axis(self, axis, dtype, keepdims):
        # out with axis
        a = np.arange(2*3*4).reshape((2, 3, 4))
        result = self.func(a, axis=axis, keepdims=keepdims).astype(dtype)

        out = np.empty_like(result, dtype=dtype)
        result_with_out = self.func(a, axis=axis, keepdims=keepdims, out=out)

        assert result_with_out is out
        assert result_with_out.dtype == dtype
        assert_equal(result, result_with_out)

        # out of wrong shape (any/out does not broadcast)
        # np.any(m, out=np.empty_like(m)) raises a ValueError (wrong number
        # of dimensions.)
        # pytorch.any emits a warning and resizes the out array.
        # Here we follow pytorch, since the result is a superset
        # of the numpy functionality

    @pytest.mark.parametrize('keepdims', [True, False, None])
    @pytest.mark.parametrize('dtype', [bool, 'int32', 'float64'])
    def test_out_axis(self, dtype, keepdims):
        for axis in self.allowed_axes + [None,]:
            self._check_out_axis(axis, dtype, keepdims)

    def _check_keepdims_out(self, axis):
        """Check the expicit shape of out w/keepdims=True"""
        d = np.ones((3, 5, 7, 11))
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = _util.normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim))
        out = np.empty(shape_out)
        result = self.func(d, axis=axis, keepdims=True, out=out)
        assert result is out
        assert_equal(result.shape, shape_out)

    def test_keepdims_out(self):
        allowed_axes = [ax for ax in self.allowed_axes if ax != ()]
        for axis in allowed_axes + [None,]:
            self._check_keepdims_out(axis)




class TestAnyGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.any
        self.allowed_axes =  [0, 1, 2, -1, -2, (),]

    @pytest.mark.xfail(reason='XXX: pytorch does not support any(..., axis=tuple)')
    def test_axis_bad_tuple(self):
        super().test_axis_bad_tuple()


class TestAllGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.all
        self.allowed_axes =  [0, 1, 2, -1, -2, (),]

    @pytest.mark.xfail(reason='XXX: pytorch does not support all(..., axis=tuple)')
    def test_axis_bad_tuple(self):
        super().test_axis_bad_tuple()


class TestCountNonzeroGeneric(_GenericReductionsTestMixin):
    # count_nonzero does not have the out=... argument
    def setup_method(self):
        self.func = np.count_nonzero
        self.allowed_axes =  [0, 1, 2, -1, -2, (),
                              (0, 1), (1, 0), (0, 1, 2), (1, -1, 0)]


class TestArgminGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.argmin
        self.allowed_axes =  [0, 1, 2, -1, -2, ]

    @pytest.mark.xfail(reason='XXX: argmin does not allow axis=tuple')
    def test_axis_bad_tuple(self):
        super().test_axis_bad_tuple()

    @pytest.mark.xfail(reason='XXX: argmin does not allow axis=tuple)')
    def test_axis_empty_generic(self):
        super().test_axis_empty_generic()


class TestArgmaxGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.argmax
        self.allowed_axes =  [0, 1, 2, -1, -2, ]

    @pytest.mark.xfail(reason='XXX: argmax does not allow axis=tuple')
    def test_axis_bad_tuple(self):
        super().test_axis_bad_tuple()

    @pytest.mark.xfail(reason='XXX: argmax does not allow axis=tuple)')
    def test_axis_empty_generic(self):
        super().test_axis_empty_generic()


class TestAmaxGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.amax
        self.allowed_axes =  [0, 1, 2, -1, -2, (),
                              (0, 1), (1, 0), (0, 1, 2), (1, -1, 0)]


class TestAminGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.amin
        self.allowed_axes =  [0, 1, 2, -1, -2, (),
                              (0, 1), (1, 0), (0, 1, 2), (1, -1, 0)]

