import pytest

import numpy as np
import torch

import torch_np as w


class TestConcatenate:
    def test_out_and_dtype(self):
        # numpy raises TypeError on both out=... and dtype=...
        a, b, out = w.ones(3), w.ones(4), w.ones(3+4)

        with pytest.raises(TypeError):
            w.concatenate((a, b), out=out, dtype=float)

    def test_returns_copy(self):
        a = np.eye(3)
        b = np.concatenate([a])
        b[0, 0] = 2
        assert b[0, 0] != a[0, 0]

    def test_exceptions(self):
        # test axis must be in bounds
        for ndim in [1, 2, 3]:
            a = np.ones((1,)*ndim)
            np.concatenate((a, a), axis=0)  # OK
            assert_raises(np.AxisError, np.concatenate, (a, a), axis=ndim)
            assert_raises(np.AxisError, np.concatenate, (a, a), axis=-(ndim + 1))

        # Scalars cannot be concatenated
        assert_raises(ValueError, concatenate, (0,))
        assert_raises(ValueError, concatenate, (np.array(0),))

        # dimensionality must match
        assert_raises_regex(
            ValueError,
            r"all the input arrays must have same number of dimensions, but "
            r"the array at index 0 has 1 dimension\(s\) and the array at "
            r"index 1 has 2 dimension\(s\)",
            np.concatenate, (np.zeros(1), np.zeros((1, 1))))

        # test shapes must match except for concatenation axis
        a = np.ones((1, 2, 3))
        b = np.ones((2, 2, 3))
        axis = list(range(3))
        for i in range(3):
            np.concatenate((a, b), axis=axis[0])  # OK
            assert_raises_regex(
                ValueError,
                "all the input array dimensions except for the concatenation axis "
                "must match exactly, but along dimension {}, the array at "
                "index 0 has size 1 and the array at index 1 has size 2"
                .format(i),
                np.concatenate, (a, b), axis=axis[1])
            assert_raises(ValueError, np.concatenate, (a, b), axis=axis[2])
            a = np.moveaxis(a, -1, 0)
            b = np.moveaxis(b, -1, 0)
            axis.append(axis.pop(0))

        # No arrays to concatenate raises ValueError
        assert_raises(ValueError, concatenate, ())

    def test_concatenate_axis_None(self):
        a = np.arange(4, dtype=np.float64).reshape((2, 2))
        b = list(range(3))
        c = ['x']
        r = np.concatenate((a, a), axis=None)
        assert_equal(r.dtype, a.dtype)
        assert_equal(r.ndim, 1)
        r = np.concatenate((a, b), axis=None)
        assert_equal(r.size, a.size + len(b))
        assert_equal(r.dtype, a.dtype)
        r = np.concatenate((a, b, c), axis=None, dtype="U")
        d = array(['0.0', '1.0', '2.0', '3.0',
                   '0', '1', '2', 'x'])
        assert_array_equal(r, d)

        out = np.zeros(a.size + len(b))
        r = np.concatenate((a, b), axis=None)
        rout = np.concatenate((a, b), axis=None, out=out)
        assert_(out is rout)
        assert_equal(r, rout)

    def test_large_concatenate_axis_None(self):
        # When no axis is given, concatenate uses flattened versions.
        # This also had a bug with many arrays (see gh-5979).
        x = np.arange(1, 100)
        r = np.concatenate(x, None)
        assert_array_equal(x, r)

        # This should probably be deprecated:
        r = np.concatenate(x, 100)  # axis is >= MAXDIMS
        assert_array_equal(x, r)

    def test_concatenate(self):
        # Test concatenate function
        # One sequence returns unmodified (but as array)
        r4 = list(range(4))
        assert_array_equal(concatenate((r4,)), r4)
        # Any sequence
        assert_array_equal(concatenate((tuple(r4),)), r4)
        assert_array_equal(concatenate((array(r4),)), r4)
        # 1D default concatenation
        r3 = list(range(3))
        assert_array_equal(concatenate((r4, r3)), r4 + r3)
        # Mixed sequence types
        assert_array_equal(concatenate((tuple(r4), r3)), r4 + r3)
        assert_array_equal(concatenate((array(r4), r3)), r4 + r3)
        # Explicit axis specification
        assert_array_equal(concatenate((r4, r3), 0), r4 + r3)
        # Including negative
        assert_array_equal(concatenate((r4, r3), -1), r4 + r3)
        # 2D
        a23 = array([[10, 11, 12], [13, 14, 15]])
        a13 = array([[0, 1, 2]])
        res = array([[10, 11, 12], [13, 14, 15], [0, 1, 2]])
        assert_array_equal(concatenate((a23, a13)), res)
        assert_array_equal(concatenate((a23, a13), 0), res)
        assert_array_equal(concatenate((a23.T, a13.T), 1), res.T)
        assert_array_equal(concatenate((a23.T, a13.T), -1), res.T)
        # Arrays much match shape
        assert_raises(ValueError, concatenate, (a23.T, a13.T), 0)
        # 3D
        res = arange(2 * 3 * 7).reshape((2, 3, 7))
        a0 = res[..., :4]
        a1 = res[..., 4:6]
        a2 = res[..., 6:]
        assert_array_equal(concatenate((a0, a1, a2), 2), res)
        assert_array_equal(concatenate((a0, a1, a2), -1), res)
        assert_array_equal(concatenate((a0.T, a1.T, a2.T), 0), res.T)

        out = res.copy()
        rout = concatenate((a0, a1, a2), 2, out=out)
        assert_(out is rout)
        assert_equal(res, rout)

    @pytest.mark.skipif(IS_PYPY, reason="PYPY handles sq_concat, nb_add differently than cpython")
    def test_operator_concat(self):
        import operator
        a = array([1, 2])
        b = array([3, 4])
        n = [1,2]
        res = array([1, 2, 3, 4])
        assert_raises(TypeError, operator.concat, a, b)
        assert_raises(TypeError, operator.concat, a, n)
        assert_raises(TypeError, operator.concat, n, a)
        assert_raises(TypeError, operator.concat, a, 1)
        assert_raises(TypeError, operator.concat, 1, a)

    def test_bad_out_shape(self):
        a = array([1, 2])
        b = array([3, 4])

        assert_raises(ValueError, concatenate, (a, b), out=np.empty(5))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((4,1)))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((1,4)))
        concatenate((a, b), out=np.empty(4))

    @pytest.mark.parametrize("axis", [None, 0])
    @pytest.mark.parametrize("out_dtype", ["c8", "f4", "f8", ">f8", "i8", "S4"])
    @pytest.mark.parametrize("casting",
            ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
    def test_out_and_dtype(self, axis, out_dtype, casting):
        # Compare usage of `out=out` with `dtype=out.dtype`
        out = np.empty(4, dtype=out_dtype)
        to_concat = (array([1.1, 2.2]), array([3.3, 4.4]))

        if not np.can_cast(to_concat[0], out_dtype, casting=casting):
            with assert_raises(TypeError):
                concatenate(to_concat, out=out, axis=axis, casting=casting)
            with assert_raises(TypeError):
                concatenate(to_concat, dtype=out.dtype,
                            axis=axis, casting=casting)
        else:
            res_out = concatenate(to_concat, out=out,
                                  axis=axis, casting=casting)
            res_dtype = concatenate(to_concat, dtype=out.dtype,
                                    axis=axis, casting=casting)
            assert res_out is out
            assert_array_equal(out, res_dtype)
            assert res_dtype.dtype == out_dtype

        with assert_raises(TypeError):
            concatenate(to_concat, out=out, dtype=out_dtype, axis=axis)

    @pytest.mark.parametrize("axis", [None, 0])
    @pytest.mark.parametrize("string_dt", ["S", "U", "S0", "U0"])
    @pytest.mark.parametrize("arrs",
            [([0.],), ([0.], [1]), ([0], ["string"], [1.])])
    def test_dtype_with_promotion(self, arrs, string_dt, axis):
        # Note that U0 and S0 should be deprecated eventually and changed to
        # actually give the empty string result (together with `np.array`)
        res = np.concatenate(arrs, axis=axis, dtype=string_dt, casting="unsafe")
        # The actual dtype should be identical to a cast (of a double array):
        assert res.dtype == np.array(1.).astype(string_dt).dtype

    @pytest.mark.parametrize("axis", [None, 0])
    def test_string_dtype_does_not_inspect(self, axis):
        with pytest.raises(TypeError):
            np.concatenate(([None], [1]), dtype="S", axis=axis)
        with pytest.raises(TypeError):
            np.concatenate(([None], [1]), dtype="U", axis=axis)

    @pytest.mark.parametrize("axis", [None, 0])
    def test_subarray_error(self, axis):
        with pytest.raises(TypeError, match=".*subarray dtype"):
            np.concatenate(([1], [1]), dtype="(2,)i", axis=axis)

