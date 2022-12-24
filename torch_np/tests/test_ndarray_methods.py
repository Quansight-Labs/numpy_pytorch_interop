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
        assert_equal(np.count_nonzero(np.array([])), 0)
        assert_equal(np.count_nonzero(np.array([], dtype='?')), 0)
        assert_equal(np.nonzero(np.array([])), ([],))

        assert_equal(np.count_nonzero(np.array([0])), 0)
        assert_equal(np.count_nonzero(np.array([0], dtype='?')), 0)
        assert_equal(np.nonzero(np.array([0])), ([],))

        assert_equal(np.count_nonzero(np.array([1])), 1)
        assert_equal(np.count_nonzero(np.array([1], dtype='?')), 1)
        assert_equal(np.nonzero(np.array([1])), ([0],))

    def test_nonzero_zerod(self):
        assert_equal(np.count_nonzero(np.array(0)), 0)
        assert_equal(np.count_nonzero(np.array(0, dtype='?')), 0)
        with assert_warns(DeprecationWarning):
            assert_equal(np.nonzero(np.array(0)), ([],))

        assert_equal(np.count_nonzero(np.array(1)), 1)
        assert_equal(np.count_nonzero(np.array(1, dtype='?')), 1)
        with assert_warns(DeprecationWarning):
            assert_equal(np.nonzero(np.array(1)), ([0],))

    def test_nonzero_onedim(self):
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))

        # x = np.array([(1, 2), (0, 0), (1, 1), (-1, 3), (0, 7)],
        #              dtype=[('a', 'i4'), ('b', 'i2')])
        x = np.array([(1, 2, -5, -3), (0, 0, 2, 7), (1, 1, 0, 1), (-1, 3, 1, 0), (0, 7, 0, 4)],
                     dtype=[('a', 'i4'), ('b', 'i2'), ('c', 'i1'), ('d', 'i8')])
        assert_equal(np.count_nonzero(x['a']), 3)
        assert_equal(np.count_nonzero(x['b']), 4)
        assert_equal(np.count_nonzero(x['c']), 3)
        assert_equal(np.count_nonzero(x['d']), 4)
        assert_equal(np.nonzero(x['a']), ([0, 2, 3],))
        assert_equal(np.nonzero(x['b']), ([0, 2, 3, 4],))

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

        x = np.array([[(0, 1), (0, 0), (1, 11)],
                   [(1, 1), (1, 0), (0, 0)],
                   [(0, 0), (1, 5), (0, 1)]], dtype=[('a', 'f4'), ('b', 'u1')])
        assert_equal(np.count_nonzero(x['a']), 4)
        assert_equal(np.count_nonzero(x['b']), 5)
        assert_equal(np.nonzero(x['a']), ([0, 1, 1, 2], [2, 0, 1, 1]))
        assert_equal(np.nonzero(x['b']), ([0, 0, 1, 2, 2], [0, 2, 0, 1, 2]))

        assert_(not x['a'].T.flags.aligned)
        assert_equal(np.count_nonzero(x['a'].T), 4)
        assert_equal(np.count_nonzero(x['b'].T), 5)
        assert_equal(np.nonzero(x['a'].T), ([0, 1, 1, 2], [1, 1, 2, 0]))
        assert_equal(np.nonzero(x['b'].T), ([0, 0, 1, 2, 2], [0, 1, 2, 0, 2]))

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

    def test_return_type(self):
        class C(np.ndarray):
            pass

        for view in (C, np.ndarray):
            for nd in range(1, 4):
                shape = tuple(range(2, 2+nd))
                x = np.arange(np.prod(shape)).reshape(shape).view(view)
                for nzx in (np.nonzero(x), x.nonzero()):
                    for nzx_i in nzx:
                        assert_(type(nzx_i) is np.ndarray)
                        assert_(nzx_i.flags.writeable)

    def test_count_nonzero_axis(self):
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        expected = np.array([1, 1, 1, 1, 1])
        assert_equal(np.count_nonzero(m, axis=0), expected)

        expected = np.array([2, 3])
        assert_equal(np.count_nonzero(m, axis=1), expected)

        assert_raises(ValueError, np.count_nonzero, m, axis=(1, 1))
        assert_raises(TypeError, np.count_nonzero, m, axis='foo')
        assert_raises(np.AxisError, np.count_nonzero, m, axis=3)
        assert_raises(TypeError, np.count_nonzero,
                      m, axis=np.array([[1], [2]]))

    def test_count_nonzero_axis_all_dtypes(self):
        # More thorough test that the axis argument is respected
        # for all dtypes and responds correctly when presented with
        # either integer or tuple arguments for axis
        msg = "Mismatch for dtype: %s"

        def assert_equal_w_dt(a, b, err_msg):
            assert_equal(a.dtype, b.dtype, err_msg=err_msg)
            assert_equal(a, b, err_msg=err_msg)

        for dt in np.typecodes['All']:
            err_msg = msg % (np.dtype(dt).name,)

            if dt != 'V':
                if dt != 'M':
                    m = np.zeros((3, 3), dtype=dt)
                    n = np.ones(1, dtype=dt)

                    m[0, 0] = n[0]
                    m[1, 0] = n[0]

                else:  # np.zeros doesn't work for np.datetime64
                    m = np.array(['1970-01-01'] * 9)
                    m = m.reshape((3, 3))

                    m[0, 0] = '1970-01-12'
                    m[1, 0] = '1970-01-12'
                    m = m.astype(dt)

                expected = np.array([2, 0, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=0),
                                  expected, err_msg=err_msg)

                expected = np.array([1, 1, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=1),
                                  expected, err_msg=err_msg)

                expected = np.array(2)
                assert_equal(np.count_nonzero(m, axis=(0, 1)),
                             expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m, axis=None),
                             expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m),
                             expected, err_msg=err_msg)

            if dt == 'V':
                # There are no 'nonzero' objects for np.void, so the testing
                # setup is slightly different for this dtype
                m = np.array([np.void(1)] * 6).reshape((2, 3))

                expected = np.array([0, 0, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=0),
                                  expected, err_msg=err_msg)

                expected = np.array([0, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=1),
                                  expected, err_msg=err_msg)

                expected = np.array(0)
                assert_equal(np.count_nonzero(m, axis=(0, 1)),
                             expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m, axis=None),
                             expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m),
                             expected, err_msg=err_msg)

    def test_count_nonzero_axis_consistent(self):
        # Check that the axis behaviour for valid axes in
        # non-special cases is consistent (and therefore
        # correct) by checking it against an integer array
        # that is then casted to the generic object dtype
        from itertools import combinations, permutations

        axis = (0, 1, 2, 3)
        size = (5, 5, 5, 5)
        msg = "Mismatch for axis: %s"

        rng = np.random.RandomState(1234)
        m = rng.randint(-100, 100, size=size)
        n = m.astype(object)

        for length in range(len(axis)):
            for combo in combinations(axis, length):
                for perm in permutations(combo):
                    assert_equal(
                        np.count_nonzero(m, axis=perm),
                        np.count_nonzero(n, axis=perm),
                        err_msg=msg % (perm,))

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

    def test_array_method(self):
        # Tests that the array method
        # call to nonzero works
        m = np.array([[1, 0, 0], [4, 0, 6]])
        tgt = [[0, 1, 1], [0, 0, 2]]

        assert_equal(m.nonzero(), tgt)

    def test_nonzero_invalid_object(self):
        # gh-9295
        a = np.array([np.array([1, 2]), 3], dtype=object)
        assert_raises(ValueError, np.nonzero, a)

        class BoolErrors:
            def __bool__(self):
                raise ValueError("Not allowed")

        assert_raises(ValueError, np.nonzero, np.array([BoolErrors()]))

    def test_nonzero_sideeffect_safety(self):
        # gh-13631
        class FalseThenTrue:
            _val = False
            def __bool__(self):
                try:
                    return self._val
                finally:
                    self._val = True

        class TrueThenFalse:
            _val = True
            def __bool__(self):
                try:
                    return self._val
                finally:
                    self._val = False

        # result grows on the second pass
        a = np.array([True, FalseThenTrue()])
        assert_raises(RuntimeError, np.nonzero, a)

        a = np.array([[True], [FalseThenTrue()]])
        assert_raises(RuntimeError, np.nonzero, a)

        # result shrinks on the second pass
        a = np.array([False, TrueThenFalse()])
        assert_raises(RuntimeError, np.nonzero, a)

        a = np.array([[False], [TrueThenFalse()]])
        assert_raises(RuntimeError, np.nonzero, a)

    def test_nonzero_sideffects_structured_void(self):
        # Checks that structured void does not mutate alignment flag of
        # original array.
        arr = np.zeros(5, dtype="i1,i8,i8")  # `ones` may short-circuit
        assert arr.flags.aligned  # structs are considered "aligned"
        assert not arr["f2"].flags.aligned
        # make sure that nonzero/count_nonzero do not flip the flag:
        np.nonzero(arr)
        assert arr.flags.aligned
        np.count_nonzero(arr)
        assert arr.flags.aligned

    def test_nonzero_exception_safe(self):
        # gh-13930

        class ThrowsAfter:
            def __init__(self, iters):
                self.iters_left = iters

            def __bool__(self):
                if self.iters_left == 0:
                    raise ValueError("called `iters` times")

                self.iters_left -= 1
                return True

        """
        Test that a ValueError is raised instead of a SystemError

        If the __bool__ function is called after the error state is set,
        Python (cpython) will raise a SystemError.
        """

        # assert that an exception in first pass is handled correctly
        a = np.array([ThrowsAfter(5)]*10)
        assert_raises(ValueError, np.nonzero, a)

        # raise exception in second pass for 1-dimensional loop
        a = np.array([ThrowsAfter(15)]*10)
        assert_raises(ValueError, np.nonzero, a)

        # raise exception in second pass for n-dimensional loop
        a = np.array([[ThrowsAfter(15)]]*10)
        assert_raises(ValueError, np.nonzero, a)

    @pytest.mark.skipif(IS_WASM, reason="wasm doesn't have threads")
    def test_structured_threadsafety(self):
        # Nonzero (and some other functions) should be threadsafe for
        # structured datatypes, see gh-15387. This test can behave randomly.
        from concurrent.futures import ThreadPoolExecutor

        # Create a deeply nested dtype to make a failure more likely:
        dt = np.dtype([("", "f8")])
        dt = np.dtype([("", dt)])
        dt = np.dtype([("", dt)] * 2)
        # The array should be large enough to likely run into threading issues
        arr = np.random.uniform(size=(5000, 4)).view(dt)[:, 0]
        def func(arr):
            arr.nonzero()

        tpe = ThreadPoolExecutor(max_workers=8)
        futures = [tpe.submit(func, arr) for _ in range(10)]
        for f in futures:
            f.result()

        assert arr.dtype is dt


