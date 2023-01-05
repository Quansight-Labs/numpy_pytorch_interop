""" Test printing of scalar types.

"""
import pytest

import torch_np as np
from torch_np.testing import assert_
from pytest import raises as assert_raises


class A:
    pass
class B(A, np.float64):
    pass

class C(B):
    pass
class D(C, B):
    pass

class B0(np.float64, A):
    pass
class C0(B0):
    pass

class HasNew:
    def __new__(cls, *args, **kwargs):
        return cls, args, kwargs

class B1(np.float64, HasNew):
    pass


@pytest.mark.xfail(reason='scalar repr: numpy plain to make more explicit')
class TestInherit:
    def test_init(self):
        x = B(1.0)
        assert_(str(x) == '1.0')
        y = C(2.0)
        assert_(str(y) == '2.0')
        z = D(3.0)
        assert_(str(z) == '3.0')

    def test_init2(self):
        x = B0(1.0)
        assert_(str(x) == '1.0')
        y = C0(2.0)
        assert_(str(y) == '2.0')

    def test_gh_15395(self):
        # HasNew is the second base, so `np.float64` should have priority
        x = B1(1.0)
        assert_(str(x) == '1.0')

        # previously caused RecursionError!?
        with pytest.raises(TypeError):
            B1(1.0, 2.0)

