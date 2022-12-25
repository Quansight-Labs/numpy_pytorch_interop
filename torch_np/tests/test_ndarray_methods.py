import pytest
from pytest import raises as assert_raises

#import numpy as np
import torch_np as np

assert_equal = np.testing.assert_equal


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
        arr = [[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9],
               [10, 11, 12]]
        tgt = [[1, 2, 3, 4, 5, 6],
               [7, 8, 9, 10, 11, 12]]
        assert np.all(np.reshape(arr, (2, 6)) == tgt)

        arr= np.asarray(arr)
        assert np.transpose(arr, (1, 0)).base is arr

    def test_reshape_method(self):
        arr = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 12]])
        arr_shape = arr.shape

        tgt = [[1, 2, 3, 4, 5, 6],
               [7, 8, 9, 10, 11, 12]]

        # reshape(*shape_tuple)
        assert np.all(arr.reshape(2, 6) == tgt)
        assert arr.reshape(2, 6).base is arr   # reshape keeps the base 
        assert arr.shape == arr_shape          # arr is intact

        # XXX: move out to dedicated test(s)
        assert arr.reshape(2, 6)._tensor._base is arr._tensor

        # reshape(shape_tuple)
        assert np.all(arr.reshape((2, 6)) == tgt)
        assert arr.reshape((2, 6)).base is arr 
        assert arr.shape == arr_shape
        
        tgt = [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12]]
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
            c[10 + i:20 + i] = True
            c[20 + i*2] = True
            assert_equal(np.nonzero(c)[0],
                         np.concatenate((np.arange(10 + i, 20 + i), [20 + i*2])))

    def test_array_method(self):
        # Tests that the array method
        # call to nonzero works
        m = np.array([[1, 0, 0], [4, 0, 6]])
        tgt = [[0, 1, 1], [0, 0, 2]]

        assert_equal(m.nonzero(), tgt)

