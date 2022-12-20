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