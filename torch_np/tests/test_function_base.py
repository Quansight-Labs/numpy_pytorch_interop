import pytest

from pytest import raises as assert_raises

import torch_np as np
from torch_np.testing import assert_equal, assert_array_equal


class TestCopy:

    def test_basic(self):
        a = np.array([[1, 2], [3, 4]])
        a_copy = np.copy(a)
        assert_array_equal(a, a_copy)
        a_copy[0, 0] = 10
        assert_equal(a[0, 0], 1)
        assert_equal(a_copy[0, 0], 10)

    @pytest.mark.skip(reason="ndarray.flags not implemented")
    def test_order(self):
        # It turns out that people rely on np.copy() preserving order by
        # default; changing this broke scikit-learn:
        # github.com/scikit-learn/scikit-learn/commit/7842748cf777412c506a8c0ed28090711d3a3783  # noqa
        a = np.array([[1, 2], [3, 4]])
        assert (a.flags.c_contiguous)
        assert (not a.flags.f_contiguous)
        a_fort = np.array([[1, 2], [3, 4]], order="F")
        assert (not a_fort.flags.c_contiguous)
        assert (a_fort.flags.f_contiguous)
        a_copy = np.copy(a)
        assert (a_copy.flags.c_contiguous)
        assert (not a_copy.flags.f_contiguous)
        a_fort_copy = np.copy(a_fort)
        assert (not a_fort_copy.flags.c_contiguous)
        assert (a_fort_copy.flags.f_contiguous)


class TestArange:
    def test_infinite(self):
        assert_raises(
            ValueError, # "size exceeded",
            np.arange, 0, np.inf
        )

    def test_nan_step(self):
        assert_raises(
            ValueError, # "cannot compute length",
            np.arange, 0, 1, np.nan
        )

    def test_zero_step(self):
        assert_raises(ZeroDivisionError, np.arange, 0, 10, 0)
        assert_raises(ZeroDivisionError, np.arange, 0.0, 10.0, 0.0)

        # empty range
        assert_raises(ZeroDivisionError, np.arange, 0, 0, 0)
        assert_raises(ZeroDivisionError, np.arange, 0.0, 0.0, 0.0)

    def test_require_range(self):
        assert_raises(TypeError, np.arange)
        assert_raises(TypeError, np.arange, step=3)
        assert_raises(TypeError, np.arange, dtype='int64')

    @pytest.mark.xfail(reason='XXX: arange(start=0, stop, step=1)')
    def test_require_range_2(self):
        assert_raises(TypeError, np.arange, start=4)

    def test_start_stop_kwarg(self):
        keyword_stop = np.arange(stop=3)
        keyword_zerotostop = np.arange(start=0, stop=3)
        keyword_start_stop = np.arange(start=3, stop=9)

        assert len(keyword_stop) == 3
        assert len(keyword_zerotostop) == 3
        assert len(keyword_start_stop) == 6
        assert_array_equal(keyword_stop, keyword_zerotostop)

    @pytest.mark.xfail(reason='XXX: arange(..., dtype=bool)')
    def test_arange_booleans(self):
        # Arange makes some sense for booleans and works up to length 2.
        # But it is weird since `arange(2, 4, dtype=bool)` works.
        # Arguably, much or all of this could be deprecated/removed.
        res = np.arange(False, dtype=bool)
        assert_array_equal(res, np.array([], dtype="bool"))

        res = np.arange(True, dtype="bool")
        assert_array_equal(res, [False])

        res = np.arange(2, dtype="bool")
        assert_array_equal(res, [False, True])

        # This case is especially weird, but drops out without special case:
        res = np.arange(6, 8, dtype="bool")
        assert_array_equal(res, [True, True])

        with pytest.raises(TypeError):
            np.arange(3, dtype="bool")


    @pytest.mark.parametrize("which", [0, 1, 2])
    def test_error_paths_and_promotion(self, which):
        args = [0, 1, 2]  # start, stop, and step
        args[which] = np.float64(2.)  # should ensure float64 output

        assert np.arange(*args).dtype == np.float64

        # Cover stranger error path, test only to achieve code coverage!
        args[which] = [None, []]
        with pytest.raises(ValueError):
            # Fails discovering start dtype
            np.arange(*args)


