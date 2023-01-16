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
from torch_np.testing import assert_array_equal, assert_equal


class TestAtleast1d:
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [array([1]), array([2])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [array([1, 2]), array([2, 3])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_r1array(self):
        """Test to make sure equivalent Travis O's r1array function"""
        assert atleast_1d(3).shape == (1,)
        assert atleast_1d(3j).shape == (1,)
        assert atleast_1d(3.0).shape == (1,)
        assert atleast_1d([[2, 3], [4, 5]]).shape == (2, 2)


class TestAtleast2d:
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [array([[1]]), array([[2]])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [array([[1, 2]]), array([[2, 3]])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_r2array(self):
        """Test to make sure equivalent Travis O's r2array function"""
        assert atleast_2d(3).shape == (1, 1)
        assert atleast_2d([3j, 1]).shape == (1, 2)
        assert atleast_2d([[[3, 1], [4, 5]], [[3, 5], [1, 2]]]).shape == (2, 2, 2)


class TestAtleast3d:
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [array([[[1]]]), array([[[2]]])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [array([[[1], [2]]]), array([[[2], [3]]])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [a[:, :, np.newaxis], b[:, :, np.newaxis]]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)


class TestHstack:
    def test_non_iterable(self):
        assert_raises(TypeError, hstack, 1)

    def test_empty_input(self):
        assert_raises(ValueError, hstack, ())

    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1])
        b = array([2])
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = hstack([a, b])
        desired = array([[1, 1], [2, 2]])
        assert_array_equal(res, desired)

    def test_generator(self):
        # numpy 1.24 emits warnings but we don't
        # with assert_warns(FutureWarning):
        hstack((np.arange(3) for _ in range(2)))
        # with assert_warns(FutureWarning):
        hstack(map(lambda x: x, np.ones((3, 2))))

    def test_casting_and_dtype(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.hstack((a, b), casting="unsafe", dtype=np.int64)
        expected_res = np.array([1, 2, 3, 2, 3, 4])
        assert_array_equal(res, expected_res)

    def test_casting_and_dtype_type_error(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            hstack((a, b), casting="safe", dtype=np.int64)


class TestVstack:
    def test_non_iterable(self):
        assert_raises(TypeError, vstack, 1)

    def test_empty_input(self):
        assert_raises(ValueError, vstack, ())

    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = vstack([a, b])
        desired = array([[1], [2]])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1])
        b = array([2])
        res = vstack([a, b])
        desired = array([[1], [2]])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = vstack([a, b])
        desired = array([[1], [2], [1], [2]])
        assert_array_equal(res, desired)

    def test_2D_array2(self):
        a = array([1, 2])
        b = array([1, 2])
        res = vstack([a, b])
        desired = array([[1, 2], [1, 2]])
        assert_array_equal(res, desired)

    def test_generator(self):
        # numpy 1.24 emits a warning but we don't
        vstack((np.arange(3) for _ in range(2)))

    def test_casting_and_dtype(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.vstack((a, b), casting="unsafe", dtype=np.int64)
        expected_res = np.array([[1, 2, 3], [2, 3, 4]])
        assert_array_equal(res, expected_res)

    def test_casting_and_dtype_type_error(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            vstack((a, b), casting="safe", dtype=np.int64)


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


class TestConcatenate:
    def test_out_and_dtype_simple(self):
        # numpy raises TypeError on both out=... and dtype=...
        a, b, out = np.ones(3), np.ones(4), np.ones(3 + 4)

        with pytest.raises(TypeError):
            concatenate((a, b), out=out, dtype=float)

    def test_returns_copy(self):
        a = np.eye(3)
        b = concatenate([a])
        b[0, 0] = 2
        assert b[0, 0] != a[0, 0]

    def test_exceptions(self):
        # test axis must be in bounds
        for ndim in [1, 2, 3]:
            a = np.ones((1,) * ndim)
            np.concatenate((a, a), axis=0)  # OK
            assert_raises(np.AxisError, np.concatenate, (a, a), axis=ndim)
            assert_raises(np.AxisError, np.concatenate, (a, a), axis=-(ndim + 1))

        # Scalars cannot be concatenated
        assert_raises(ValueError, concatenate, (0,))
        assert_raises(ValueError, concatenate, (np.array(0),))

        # dimensionality must match
        assert_raises(
            ValueError,
            #        assert_raises_regex(
            #            ValueError,
            #            r"all the input arrays must have same number of dimensions, but "
            #            r"the array at index 0 has 1 dimension\(s\) and the array at "
            #            r"index 1 has 2 dimension\(s\)",
            np.concatenate,
            (np.zeros(1), np.zeros((1, 1))),
        )

        # test shapes must match except for concatenation axis
        a = np.ones((1, 2, 3))
        b = np.ones((2, 2, 3))
        axis = list(range(3))
        for i in range(3):
            np.concatenate((a, b), axis=axis[0])  # OK
            #            assert_raises_regex(
            assert_raises(
                ValueError,
                #                "all the input array dimensions except for the concatenation axis "
                #                "must match exactly, but along dimension {}, the array at "
                #                "index 0 has size 1 and the array at index 1 has size 2"
                #                .format(i),
                np.concatenate,
                (a, b),
                axis=axis[1],
            )
            assert_raises(ValueError, np.concatenate, (a, b), axis=axis[2])
            a = np.moveaxis(a, -1, 0)
            b = np.moveaxis(b, -1, 0)
            axis.append(axis.pop(0))

        # No arrays to concatenate raises ValueError
        assert_raises(ValueError, concatenate, ())

    def test_concatenate_axis_None(self):
        a = np.arange(4, dtype=np.float64).reshape((2, 2))
        b = list(range(3))

        r = np.concatenate((a, a), axis=None)
        assert r.dtype == a.dtype
        assert r.ndim == 1

        r = np.concatenate((a, b), axis=None)
        assert r.size == a.size + len(b)
        assert r.dtype == a.dtype

        out = np.zeros(a.size + len(b))
        r = np.concatenate((a, b), axis=None)
        rout = np.concatenate((a, b), axis=None, out=out)
        assert out is rout
        assert np.all(r == rout)

    @pytest.mark.xfail(reason="concatenate(x, axis=None) relies on x being a sequence")
    def test_large_concatenate_axis_None(self):
        # When no axis is given, concatenate uses flattened versions.
        # This also had a bug with many arrays (see gh-5979).
        x = np.arange(1, 100)
        r = np.concatenate(x, None)
        assert np.all(x == r)

        # This should probably be deprecated:
        r = np.concatenate(x, 100)  # axis is >= MAXDIMS
        assert_array_equal(x, r)

    def test_concatenate(self):
        # Test concatenate function
        # One sequence returns unmodified (but as array)

        # XXX: a single argument; relies on an ndarray being a sequence
        r4 = list(range(4))
        ##        assert_array_equal(concatenate((r4,)), r4)
        ##        # Any sequence
        ##        assert_array_equal(concatenate((tuple(r4),)), r4)
        ##        assert_array_equal(concatenate((array(r4),)), r4)
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
        res = np.arange(2 * 3 * 7).reshape((2, 3, 7))
        a0 = res[..., :4]
        a1 = res[..., 4:6]
        a2 = res[..., 6:]
        assert_array_equal(concatenate((a0, a1, a2), 2), res)
        assert_array_equal(concatenate((a0, a1, a2), -1), res)
        c = concatenate((a0.T, a1.T, a2.T), 0)
        assert_array_equal(concatenate((a0.T, a1.T, a2.T), 0), res.T)

        out = res.copy()
        rout = concatenate((a0, a1, a2), 2, out=out)
        assert out is rout
        assert_equal(res, rout)

    def test_bad_out_shape(self):
        a = array([1, 2])
        b = array([3, 4])

        assert_raises(ValueError, concatenate, (a, b), out=np.empty(5))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((4, 1)))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((1, 4)))
        concatenate((a, b), out=np.empty(4))

    @pytest.mark.parametrize("axis", [None, 0])
    @pytest.mark.parametrize(
        "out_dtype", ["c8", "f4", "f8", "i8"]
    )  # torch does not have ">f8", "S4"
    @pytest.mark.parametrize("casting", ["no", "equiv", "safe", "same_kind", "unsafe"])
    def test_out_and_dtype(self, axis, out_dtype, casting):
        # Compare usage of `out=out` with `dtype=out.dtype`
        out = np.empty(4, dtype=out_dtype)
        to_concat = (array([1.1, 2.2]), array([3.3, 4.4]))

        if not np.can_cast(to_concat[0], out_dtype, casting=casting):
            with assert_raises(TypeError):
                concatenate(to_concat, out=out, axis=axis, casting=casting)
            with assert_raises(TypeError):
                concatenate(to_concat, dtype=out.dtype, axis=axis, casting=casting)
        else:
            res_out = concatenate(to_concat, out=out, axis=axis, casting=casting)
            res_dtype = concatenate(
                to_concat, dtype=out.dtype, axis=axis, casting=casting
            )
            assert res_out is out
            assert_array_equal(out, res_dtype)
            assert res_dtype.dtype == out_dtype

        with assert_raises(TypeError):
            concatenate(to_concat, out=out, dtype=out_dtype, axis=axis)


def test_stack():
    # non-iterable input
    assert_raises(TypeError, stack, 1)

    # 0d input
    for input_ in [
        (1, 2, 3),
        [np.int32(1), np.int32(2), np.int32(3)],
        [np.array(1), np.array(2), np.array(3)],
    ]:
        assert_array_equal(stack(input_), [1, 2, 3])
    # 1d input examples
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    r1 = array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(np.stack((a, b)), r1)
    assert_array_equal(np.stack((a, b), axis=1), r1.T)

    # all input types
    assert_array_equal(np.stack(list([a, b])), r1)
    assert_array_equal(np.stack(array([a, b])), r1)

    # all shapes for 1d input
    arrays = [np.ones(3) * j for j in range(10)]
    axes = [0, 1, -1, -2]
    expected_shapes = [(10, 3), (3, 10), (3, 10), (10, 3)]
    for axis, expected_shape in zip(axes, expected_shapes):
        assert_equal(np.stack(arrays, axis).shape, expected_shape)

    # XXX: numpy raises AxesError, an IndexError subclass
    assert_raises(IndexError, stack, arrays, axis=2)
    assert_raises(IndexError, stack, arrays, axis=-3)

    # all shapes for 2d input
    arrays = [np.ones((3, 4)) * j for j in range(10)]
    axes = [0, 1, 2, -1, -2, -3]
    expected_shapes = [
        (10, 3, 4),
        (3, 10, 4),
        (3, 4, 10),
        (3, 4, 10),
        (3, 10, 4),
        (10, 3, 4),
    ]
    for axis, expected_shape in zip(axes, expected_shapes):
        assert_equal(np.stack(arrays, axis).shape, expected_shape)

    # empty arrays
    assert stack([[], [], []]).shape == (3, 0)
    assert stack([[], [], []], axis=1).shape == (0, 3)

    # out
    out = np.zeros_like(r1)
    np.stack((a, b), out=out)
    assert_array_equal(out, r1)

    # edge cases
    assert_raises(ValueError, stack, [])
    assert_raises(ValueError, stack, [])
    assert_raises(ValueError, stack, [1, np.arange(3)])
    assert_raises(ValueError, stack, [np.arange(3), 1])
    assert_raises(ValueError, stack, [np.arange(3), 1], axis=1)
    assert_raises(ValueError, stack, [np.zeros((3, 3)), np.zeros(3)], axis=1)
    assert_raises(ValueError, stack, [np.arange(2), np.arange(3)])

    # generator is deprecated: numpy 1.24 emits a warning but we don't
    # with assert_warns(FutureWarning):
    result = stack((x for x in range(3)))

    assert_array_equal(result, np.array([0, 1, 2]))

    # casting and dtype test
    a = np.array([1, 2, 3])
    b = np.array([2.5, 3.5, 4.5])
    res = np.stack((a, b), axis=1, casting="unsafe", dtype=np.int64)
    expected_res = np.array([[1, 2], [2, 3], [3, 4]])
    assert_array_equal(res, expected_res)

    # casting and dtype with TypeError
    with assert_raises(TypeError):
        stack((a, b), dtype=np.int64, axis=1, casting="safe")


@pytest.mark.parametrize("axis", [0])
@pytest.mark.parametrize(
    "out_dtype", ["c8", "f4", "f8", "i8"]
)  # torch does not have ">f8",
@pytest.mark.parametrize("casting", ["no", "equiv", "safe", "same_kind", "unsafe"])
def test_stack_out_and_dtype(axis, out_dtype, casting):
    to_concat = (array([1, 2]), array([3, 4]))
    res = array([[1, 2], [3, 4]])
    out = np.zeros_like(res)

    if not np.can_cast(to_concat[0], out_dtype, casting=casting):
        with assert_raises(TypeError):
            stack(to_concat, dtype=out_dtype, axis=axis, casting=casting)
    else:
        res_out = stack(to_concat, out=out, axis=axis, casting=casting)
        res_dtype = stack(to_concat, dtype=out_dtype, axis=axis, casting=casting)
        assert res_out is out
        assert_array_equal(out, res_dtype)
        assert res_dtype.dtype == out_dtype

    with assert_raises(TypeError):
        stack(to_concat, out=out, dtype=out_dtype, axis=axis)


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
