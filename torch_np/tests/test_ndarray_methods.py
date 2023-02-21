import itertools

import pytest
from pytest import raises as assert_raises

# import numpy as np
import torch_np as np
from torch_np.testing import assert_equal


class TestIndexing:
    def test_indexing_simple(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])

        assert isinstance(a[0, 0], np.ndarray)
        assert isinstance(a[0, :], np.ndarray)
        assert a[0, :].base is a

    def test_setitem(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        a[0, 0] = 8
        assert isinstance(a, np.ndarray)
        assert_equal(a, [[8, 2, 3], [4, 5, 6]])


class TestReshape:
    def test_reshape_function(self):
        arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        assert np.all(np.reshape(arr, (2, 6)) == tgt)

        arr = np.asarray(arr)
        assert np.transpose(arr, (1, 0)).base is arr

    def test_reshape_method(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        arr_shape = arr.shape

        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]

        # reshape(*shape_tuple)
        assert np.all(arr.reshape(2, 6) == tgt)
        assert arr.reshape(2, 6).base is arr  # reshape keeps the base
        assert arr.shape == arr_shape  # arr is intact

        # XXX: move out to dedicated test(s)
        assert arr.reshape(2, 6)._tensor._base is arr._tensor

        # reshape(shape_tuple)
        assert np.all(arr.reshape((2, 6)) == tgt)
        assert arr.reshape((2, 6)).base is arr
        assert arr.shape == arr_shape

        tgt = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        assert np.all(arr.reshape(3, 4) == tgt)
        assert arr.reshape(3, 4).base is arr
        assert arr.shape == arr_shape

        assert np.all(arr.reshape((3, 4)) == tgt)
        assert arr.reshape((3, 4)).base is arr
        assert arr.shape == arr_shape


# XXX : order='C' / 'F'
##        tgt = [[1, 4, 7, 10],
##               [2, 5, 8, 11],
##               [3, 6, 9, 12]]
##        assert np.all(arr.T.reshape((3, 4), order='C') == tgt)
##
##        tgt = [[1, 10, 8, 6], [4, 2, 11, 9], [7, 5, 3, 12]]
##        assert_equal(arr.reshape((3, 4), order='F'), tgt)
##


class TestTranspose:
    def test_transpose_function(self):
        arr = [[1, 2], [3, 4], [5, 6]]
        tgt = [[1, 3, 5], [2, 4, 6]]
        assert_equal(np.transpose(arr, (1, 0)), tgt)

        arr = np.asarray(arr)
        assert np.transpose(arr, (1, 0)).base is arr

    def test_transpose_method(self):
        a = np.array([[1, 2], [3, 4]])
        assert_equal(a.transpose(), [[1, 3], [2, 4]])
        assert_equal(a.transpose(None), [[1, 3], [2, 4]])
        assert_raises(ValueError, lambda: a.transpose(0))
        assert_raises(ValueError, lambda: a.transpose(0, 0))
        assert_raises(ValueError, lambda: a.transpose(0, 1, 2))

        assert a.transpose().base is a


class TestRavel:
    def test_ravel_function(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert_equal(np.ravel(a), tgt)

        arr = np.asarray(a)
        assert np.ravel(arr).base is arr

    def test_ravel_method(self):
        a = np.array([[0, 1], [2, 3]])
        assert_equal(a.ravel(), [0, 1, 2, 3])

        assert a.ravel().base is a


class TestNonzero:
    def test_nonzero_trivial(self):
        assert_equal(np.nonzero(np.array([])), ([],))
        assert_equal(np.array([]).nonzero(), ([],))

        assert_equal(np.nonzero(np.array([0])), ([],))
        assert_equal(np.array([0]).nonzero(), ([],))

        assert_equal(np.nonzero(np.array([1])), ([0],))
        assert_equal(np.array([1]).nonzero(), ([0],))

    def test_nonzero_onedim(self):
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))
        assert_equal(x.nonzero(), ([0, 2, 3, 6],))

    def test_nonzero_twodim(self):
        x = np.array([[0, 1, 0], [2, 0, 3]])
        assert_equal(np.nonzero(x), ([0, 1, 1], [1, 0, 2]))
        assert_equal(x.nonzero(), ([0, 1, 1], [1, 0, 2]))

        x = np.eye(3)
        assert_equal(np.nonzero(x), ([0, 1, 2], [0, 1, 2]))
        assert_equal(x.nonzero(), ([0, 1, 2], [0, 1, 2]))

    def test_sparse(self):
        # test special sparse condition boolean code path
        for i in range(20):
            c = np.zeros(200, dtype=bool)
            c[i::20] = True
            assert_equal(np.nonzero(c)[0], np.arange(i, 200 + i, 20))
            assert_equal(c.nonzero()[0], np.arange(i, 200 + i, 20))

            c = np.zeros(400, dtype=bool)
            c[10 + i : 20 + i] = True
            c[20 + i * 2] = True
            assert_equal(
                np.nonzero(c)[0],
                np.concatenate((np.arange(10 + i, 20 + i), [20 + i * 2])),
            )

    def test_array_method(self):
        # Tests that the array method
        # call to nonzero works
        m = np.array([[1, 0, 0], [4, 0, 6]])
        tgt = [[0, 1, 1], [0, 0, 2]]

        assert_equal(m.nonzero(), tgt)


class TestArgmaxArgminCommon:

    sizes = [
        (),
        (3,),
        (3, 2),
        (2, 3),
        (3, 3),
        (2, 3, 4),
        (4, 3, 2),
        (1, 2, 3, 4),
        (2, 3, 4, 1),
        (3, 4, 1, 2),
        (4, 1, 2, 3),
        (64,),
        (128,),
        (256,),
    ]

    @pytest.mark.parametrize(
        "size, axis",
        itertools.chain(
            *[
                [(size, axis) for axis in list(range(-len(size), len(size))) + [None]]
                for size in sizes
            ]
        ),
    )
    @pytest.mark.parametrize("method", [np.argmax, np.argmin])
    def test_np_argmin_argmax_keepdims(self, size, axis, method):

        # arr = np.random.normal(size=size)
        arr = np.empty(shape=size)

        # contiguous arrays
        if axis is None:
            new_shape = [1 for _ in range(len(size))]
        else:
            new_shape = list(size)
            new_shape[axis] = 1
        new_shape = tuple(new_shape)

        _res_orig = method(arr, axis=axis)
        res_orig = _res_orig.reshape(new_shape)
        res = method(arr, axis=axis, keepdims=True)
        assert_equal(res, res_orig)
        assert res.shape == new_shape

        outarray = np.empty(res.shape, dtype=res.dtype)
        res1 = method(arr, axis=axis, out=outarray, keepdims=True)
        assert res1 is outarray
        assert_equal(res, outarray)

        if len(size) > 0:
            wrong_shape = list(new_shape)
            if axis is not None:
                wrong_shape[axis] = 2
            else:
                wrong_shape[0] = 2
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            with pytest.raises(ValueError):
                method(arr.T, axis=axis, out=wrong_outarray, keepdims=True)

        # non-contiguous arrays
        if axis is None:
            new_shape = [1 for _ in range(len(size))]
        else:
            new_shape = list(size)[::-1]
            new_shape[axis] = 1
        new_shape = tuple(new_shape)

        _res_orig = method(arr.T, axis=axis)
        res_orig = _res_orig.reshape(new_shape)
        res = method(arr.T, axis=axis, keepdims=True)
        assert_equal(res, res_orig)
        assert res.shape == new_shape
        outarray = np.empty(new_shape[::-1], dtype=res.dtype)
        outarray = outarray.T
        res1 = method(arr.T, axis=axis, out=outarray, keepdims=True)
        assert res1 is outarray
        assert_equal(res, outarray)

        if len(size) > 0:
            # one dimension lesser for non-zero sized
            # array should raise an error
            with pytest.raises(ValueError):
                method(arr[0], axis=axis, out=outarray, keepdims=True)

        if len(size) > 0:
            wrong_shape = list(new_shape)
            if axis is not None:
                wrong_shape[axis] = 2
            else:
                wrong_shape[0] = 2
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            with pytest.raises(ValueError):
                method(arr.T, axis=axis, out=wrong_outarray, keepdims=True)

    @pytest.mark.skipif(reason="XXX: need ndarray.chooses")
    @pytest.mark.parametrize("method", ["max", "min"])
    def test_all(self, method):
        # a = np.random.normal(0, 1, (4, 5, 6, 7, 8))
        a = np.arange((4 * 5 * 6 * 7 * 8)).reshape((4, 5, 6, 7, 8))
        arg_method = getattr(a, "arg" + method)
        val_method = getattr(a, method)
        for i in range(a.ndim):
            a_maxmin = val_method(i)
            aarg_maxmin = arg_method(i)
            axes = list(range(a.ndim))
            axes.remove(i)
            assert np.all(a_maxmin == aarg_maxmin.choose(*a.transpose(i, *axes)))

    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_output_shape(self, method):
        # see also gh-616
        a = np.ones((10, 5))
        arg_method = getattr(a, method)

        # Check some simple shape mismatches
        out = np.ones(11, dtype=np.int_)

        with assert_raises(ValueError):
            arg_method(-1, out=out)

        out = np.ones((2, 5), dtype=np.int_)
        with assert_raises(ValueError):
            arg_method(-1, out=out)

        # these could be relaxed possibly (used to allow even the previous)
        out = np.ones((1, 10), dtype=np.int_)
        with assert_raises(ValueError):
            arg_method(-1, out=out)

        out = np.ones(10, dtype=np.int_)
        arg_method(-1, out=out)
        assert_equal(out, arg_method(-1))

    @pytest.mark.parametrize("ndim", [0, 1])
    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_ret_is_out(self, ndim, method):
        a = np.ones((4,) + (256,) * ndim)
        arg_method = getattr(a, method)
        out = np.empty((256,) * ndim, dtype=np.intp)
        ret = arg_method(axis=0, out=out)
        assert ret is out

    @pytest.mark.parametrize(
        "arr_method, np_method", [("argmax", np.argmax), ("argmin", np.argmin)]
    )
    def test_np_vs_ndarray(self, arr_method, np_method):
        # make sure both ndarray.argmax/argmin and
        # numpy.argmax/argmin support out/axis args
        # a = np.random.normal(size=(2, 3))
        a = np.arange(6).reshape((2, 3))
        arg_method = getattr(a, arr_method)

        # check keyword args
        out1 = np.zeros(3, dtype=int)
        out2 = np.zeros(3, dtype=int)
        assert_equal(arg_method(out=out1, axis=0), np_method(a, out=out2, axis=0))
        assert_equal(out1, out2)

    @pytest.mark.xfail(reason="out=... as a positional arg")
    @pytest.mark.parametrize(
        "arr_method, np_method", [("argmax", np.argmax), ("argmin", np.argmin)]
    )
    def test_np_vs_ndarray_positional(self, arr_method, np_method):
        a = np.arange(6).reshape((2, 3))
        arg_method = getattr(a, arr_method)

        # check positional args
        out1 = np.zeros(2, dtype=int)
        out2 = np.zeros(2, dtype=int)
        assert_equal(arg_method(1, out1), np_method(a, 1, out2))
        assert_equal(out1, out2)


class TestArgmax:
    usg_data = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 0),
        ([3, 3, 3, 3, 2, 2, 2, 2], 0),
        ([0, 1, 2, 3, 4, 5, 6, 7], 7),
        ([7, 6, 5, 4, 3, 2, 1, 0], 0),
    ]
    sg_data = usg_data + [
        ([1, 2, 3, 4, -4, -3, -2, -1], 3),
        ([1, 2, 3, 4, -1, -2, -3, -4], 3),
    ]
    darr = [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (itertools.product(usg_data, (np.uint8,)))
    ]
    darr = darr + [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                sg_data, (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64)
            )
        )
    ]
    darr = darr + [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                (
                    ([0, 1, 2, 3, np.nan], 4),
                    ([0, 1, 2, np.nan, 3], 3),
                    ([np.nan, 0, 1, 2, 3], 0),
                    ([np.nan, 0, np.nan, 2, 3], 0),
                    # To hit the tail of SIMD multi-level(x4, x1) inner loops
                    # on variant SIMD widthes
                    ([1] * (2 * 5 - 1) + [np.nan], 2 * 5 - 1),
                    ([1] * (4 * 5 - 1) + [np.nan], 4 * 5 - 1),
                    ([1] * (8 * 5 - 1) + [np.nan], 8 * 5 - 1),
                    ([1] * (16 * 5 - 1) + [np.nan], 16 * 5 - 1),
                    ([1] * (32 * 5 - 1) + [np.nan], 32 * 5 - 1),
                ),
                (np.float32, np.float64),
            )
        )
    ]
    nan_arr = darr + [
        ([0, 1, 2, 3, complex(0, np.nan)], 4),
        ([0, 1, 2, 3, complex(np.nan, 0)], 4),
        ([0, 1, 2, complex(np.nan, 0), 3], 3),
        ([0, 1, 2, complex(0, np.nan), 3], 3),
        ([complex(0, np.nan), 0, 1, 2, 3], 0),
        ([complex(np.nan, np.nan), 0, 1, 2, 3], 0),
        ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, 1)], 0),
        ([complex(np.nan, np.nan), complex(np.nan, 2), complex(np.nan, 1)], 0),
        ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, np.nan)], 0),
        ([complex(0, 0), complex(0, 2), complex(0, 1)], 1),
        ([complex(1, 0), complex(0, 2), complex(0, 1)], 0),
        ([complex(1, 0), complex(0, 2), complex(1, 1)], 2),
        ([False, False, False, False, True], 4),
        ([False, False, False, True, False], 3),
        ([True, False, False, False, False], 0),
        ([True, False, True, False, False], 0),
    ]

    @pytest.mark.parametrize("data", nan_arr)
    def test_combinations(self, data):
        arr, pos = data
        #      with suppress_warnings() as sup:
        #          sup.filter(RuntimeWarning,
        #                      "invalid value encountered in reduce")
        if np.asarray(arr).dtype.kind in "cb":
            pytest.xfail(
                reason="'max_values_cpu' not implemented for 'ComplexDouble', 'Bool'"
            )

        val = np.max(arr)

        assert_equal(np.argmax(arr), pos)  # , err_msg="%r" % arr)
        assert_equal(arr[np.argmax(arr)], val)  # , err_msg="%r" % arr)

        # add padding to test SIMD loops
        rarr = np.repeat(arr, 129)
        rpos = pos * 129
        assert_equal(np.argmax(rarr), rpos, err_msg="%r" % rarr)
        assert_equal(rarr[np.argmax(rarr)], val, err_msg="%r" % rarr)

        padd = np.repeat(np.min(arr), 513)
        rarr = np.concatenate((arr, padd))
        rpos = pos
        assert_equal(np.argmax(rarr), rpos, err_msg="%r" % rarr)
        assert_equal(rarr[np.argmax(rarr)], val, err_msg="%r" % rarr)

    def test_maximum_signed_integers(self):

        a = np.array([1, 2**7 - 1, -(2**7)], dtype=np.int8)
        assert_equal(np.argmax(a), 1)

        a = np.array([1, 2**15 - 1, -(2**15)], dtype=np.int16)
        assert_equal(np.argmax(a), 1)

        a = np.array([1, 2**31 - 1, -(2**31)], dtype=np.int32)
        assert_equal(np.argmax(a), 1)

        a = np.array([1, 2**63 - 1, -(2**63)], dtype=np.int64)
        assert_equal(np.argmax(a), 1)


class TestArgmin:
    usg_data = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 8),
        ([3, 3, 3, 3, 2, 2, 2, 2], 4),
        ([0, 1, 2, 3, 4, 5, 6, 7], 0),
        ([7, 6, 5, 4, 3, 2, 1, 0], 7),
    ]
    sg_data = usg_data + [
        ([1, 2, 3, 4, -4, -3, -2, -1], 4),
        ([1, 2, 3, 4, -1, -2, -3, -4], 7),
    ]
    darr = [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (itertools.product(usg_data, (np.uint8,)))
    ]
    darr = darr + [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                sg_data, (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64)
            )
        )
    ]
    darr = darr + [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                (
                    ([0, 1, 2, 3, np.nan], 4),
                    ([0, 1, 2, np.nan, 3], 3),
                    ([np.nan, 0, 1, 2, 3], 0),
                    ([np.nan, 0, np.nan, 2, 3], 0),
                    # To hit the tail of SIMD multi-level(x4, x1) inner loops
                    # on variant SIMD widthes
                    ([1] * (2 * 5 - 1) + [np.nan], 2 * 5 - 1),
                    ([1] * (4 * 5 - 1) + [np.nan], 4 * 5 - 1),
                    ([1] * (8 * 5 - 1) + [np.nan], 8 * 5 - 1),
                    ([1] * (16 * 5 - 1) + [np.nan], 16 * 5 - 1),
                    ([1] * (32 * 5 - 1) + [np.nan], 32 * 5 - 1),
                ),
                (np.float32, np.float64),
            )
        )
    ]
    nan_arr = darr + [
        ([0, 1, 2, 3, complex(0, np.nan)], 4),
        ([0, 1, 2, 3, complex(np.nan, 0)], 4),
        ([0, 1, 2, complex(np.nan, 0), 3], 3),
        ([0, 1, 2, complex(0, np.nan), 3], 3),
        ([complex(0, np.nan), 0, 1, 2, 3], 0),
        ([complex(np.nan, np.nan), 0, 1, 2, 3], 0),
        ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, 1)], 0),
        ([complex(np.nan, np.nan), complex(np.nan, 2), complex(np.nan, 1)], 0),
        ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, np.nan)], 0),
        ([complex(0, 0), complex(0, 2), complex(0, 1)], 0),
        ([complex(1, 0), complex(0, 2), complex(0, 1)], 2),
        ([complex(1, 0), complex(0, 2), complex(1, 1)], 1),
        ([True, True, True, True, False], 4),
        ([True, True, True, False, True], 3),
        ([False, True, True, True, True], 0),
        ([False, True, False, True, True], 0),
    ]

    @pytest.mark.parametrize("data", nan_arr)
    def test_combinations(self, data):
        arr, pos = data

        if np.asarray(arr).dtype.kind in "cb":
            pytest.xfail(
                reason="'min_values_cpu' not implemented for 'ComplexDouble', 'Bool'"
            )

        #        with suppress_warnings() as sup:
        #            sup.filter(RuntimeWarning, "invalid value encountered in reduce")
        min_val = np.min(arr)

        assert_equal(np.argmin(arr), pos, err_msg="%r" % arr)
        assert_equal(arr[np.argmin(arr)], min_val, err_msg="%r" % arr)

        # add padding to test SIMD loops
        rarr = np.repeat(arr, 129)
        rpos = pos * 129
        assert_equal(np.argmin(rarr), rpos, err_msg="%r" % rarr)
        assert_equal(rarr[np.argmin(rarr)], min_val, err_msg="%r" % rarr)

        padd = np.repeat(np.max(arr), 513)
        rarr = np.concatenate((arr, padd))
        rpos = pos
        assert_equal(np.argmin(rarr), rpos, err_msg="%r" % rarr)
        assert_equal(rarr[np.argmin(rarr)], min_val, err_msg="%r" % rarr)

    def test_minimum_signed_integers(self):

        a = np.array([1, -(2**7), -(2**7) + 1, 2**7 - 1], dtype=np.int8)
        assert_equal(np.argmin(a), 1)

        a = np.array([1, -(2**15), -(2**15) + 1, 2**15 - 1], dtype=np.int16)
        assert_equal(np.argmin(a), 1)

        a = np.array([1, -(2**31), -(2**31) + 1, 2**31 - 1], dtype=np.int32)
        assert_equal(np.argmin(a), 1)

        a = np.array([1, -(2**63), -(2**63) + 1, 2**63 - 1], dtype=np.int64)
        assert_equal(np.argmin(a), 1)


class TestAmax:
    def test_basic(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.amax(a), 10.0)
        b = [[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]]
        assert_equal(np.amax(b, axis=0), [8.0, 10.0, 9.0])
        assert_equal(np.amax(b, axis=1), [9.0, 10.0, 8.0])

        arr = np.asarray(a)
        assert_equal(np.amax(arr), arr.max())


class TestAmin:
    def test_basic(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.amin(a), -5.0)
        b = [[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]]
        assert_equal(np.amin(b, axis=0), [3.0, 3.0, 2.0])
        assert_equal(np.amin(b, axis=1), [3.0, 4.0, 2.0])

        arr = np.asarray(a)
        assert_equal(np.amin(arr), arr.min())
