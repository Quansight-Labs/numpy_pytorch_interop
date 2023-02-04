import numpy as _np
import pytest
from pytest import raises as assert_raises

import torch_np as np
from torch_np import (
    array,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    column_stack,
    concatenate,
    dstack,
    expand_dims,
    hstack,
    stack,
    vstack,
)
from torch_np.testing import assert_array_equal, assert_equal, assert_


class TestDstack:
    def test_non_iterable(self):
        assert_raises(TypeError, dstack, 1)

    def test_0D_array(self):
        a = np.array(1)
        b = np.array(2)
        res = dstack([a, b])
        desired = np.array([[[1, 2]]])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = np.array([1])
        b = np.array([2])
        res = dstack([a, b])
        desired = np.array([[[1, 2]]])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = np.array([[1], [2]])
        b = np.array([[1], [2]])
        res = dstack([a, b])
        desired = np.array(
            [
                [[1, 1]],
                [
                    [
                        2,
                        2,
                    ]
                ],
            ]
        )
        assert_array_equal(res, desired)

    def test_2D_array2(self):
        a = np.array([1, 2])
        b = np.array([1, 2])
        res = dstack([a, b])
        desired = np.array([[[1, 1], [2, 2]]])
        assert_array_equal(res, desired)

    def test_generator(self):
        # numpy 1.24 emits a warning but we don't
        # with assert_warns(FutureWarning):
        dstack((np.arange(3) for _ in range(2)))


class TestColumnStack:
    def test_non_iterable(self):
        assert_raises(TypeError, column_stack, 1)

    def test_1D_arrays(self):
        # example from docstring
        a = np.array((1, 2, 3))
        b = np.array((2, 3, 4))
        expected = np.array([[1, 2], [2, 3], [3, 4]])
        actual = np.column_stack((a, b))
        assert_equal(actual, expected)

    def test_2D_arrays(self):
        # same as hstack 2D docstring example
        a = np.array([[1], [2], [3]])
        b = np.array([[2], [3], [4]])
        expected = np.array([[1, 2], [2, 3], [3, 4]])
        actual = np.column_stack((a, b))
        assert_equal(actual, expected)

    def test_generator(self):
        # numpy 1.24 emits a warning but we don't
        # with assert_warns(FutureWarning):
        column_stack((np.arange(3) for _ in range(2)))




class TestSqueeze:
    def test_basic(self):
        a = np.arange(20 * 10 * 10 * 1 * 1).reshape(20, 10, 10, 1, 1)
        b = np.arange(20 * 1 * 10 * 1 * 20).reshape(20, 1, 10, 1, 20)
        c = np.arange(1 * 1 * 20 * 10).reshape(1, 1, 20, 10)
        assert_array_equal(np.squeeze(a), np.reshape(a, (20, 10, 10)))
        assert_array_equal(np.squeeze(b), np.reshape(b, (20, 10, 20)))
        assert_array_equal(np.squeeze(c), np.reshape(c, (20, 10)))

        # Squeezing to 0-dim should still give an ndarray
        a = [[[1.5]]]
        res = np.squeeze(a)
        assert_equal(res, 1.5)
        assert_equal(res.ndim, 0)
        assert type(res) is np.ndarray

        aa = np.ones((3, 1, 4, 1, 1))
        assert aa.squeeze().base is aa

    def test_squeeze_axis(self):
        A = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        assert_equal(np.squeeze(A).shape, (3, 3))
        assert_equal(np.squeeze(A, axis=()), A)

        assert_equal(np.squeeze(np.zeros((1, 3, 1))).shape, (3,))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=0).shape, (3, 1))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=-1).shape, (1, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))]).shape, (3,))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=0).shape, (3, 1))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=-1).shape, (1, 3))

    def test_squeeze_type(self):
        # Ticket #133
        a = np.array([3])
        b = np.array(3)
        assert type(a.squeeze()) is np.ndarray
        assert type(b.squeeze()) is np.ndarray

    @pytest.mark.skip(reason="XXX: order='F' not implemented")
    def test_squeeze_contiguous(self):
        # Similar to GitHub issue #387
        a = np.zeros((1, 2)).squeeze()
        b = np.zeros((2, 2, 2), order="F")[:, :, ::2].squeeze()
        assert_(a.flags.c_contiguous)
        assert_(a.flags.f_contiguous)
        assert_(b.flags.f_contiguous)

    @pytest.mark.xfail(reason="XXX: noop in torch, while numpy raises")
    def test_squeeze_axis_handling(self):
        with assert_raises(ValueError):
            np.squeeze(np.array([[1], [2], [3]]), axis=0)


class TestExpandDims:
    def test_functionality(self):
        s = (2, 3, 4, 5)
        a = np.empty(s)
        for axis in range(-5, 4):
            b = expand_dims(a, axis)
            assert b.shape[axis] == 1
            assert np.squeeze(b).shape == s
            assert b.base is a
            assert isinstance(b, np.ndarray)

    def test_axis_tuple(self):
        a = np.empty((3, 3, 3))
        assert np.expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
        assert np.expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
        assert np.expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
        assert np.expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)

    def test_axis_out_of_range(self):
        s = (2, 3, 4, 5)
        a = np.empty(s)
        assert_raises(np.AxisError, expand_dims, a, -6)
        assert_raises(np.AxisError, expand_dims, a, 5)

        a = np.empty((3, 3, 3))
        assert_raises(np.AxisError, expand_dims, a, (0, -6))
        assert_raises(np.AxisError, expand_dims, a, (0, 5))

    def test_repeated_axis(self):
        a = np.empty((3, 3, 3))
        assert_raises(ValueError, expand_dims, a, axis=(1, 1))
