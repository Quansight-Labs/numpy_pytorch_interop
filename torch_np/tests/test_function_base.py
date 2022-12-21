import pytest

import torch_np as np
from torch_np.testing import assert_equal


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

