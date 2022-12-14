import pytest

import numpy as np
import torch

import wrapper as w
import _unary_ufuncs

# These function receive one array_like arg and return one array_like result
one_arg_funcs = [w.asarray, w.empty_like, w.ones_like, w.zeros_like,
                 w.corrcoef, w.squeeze,
                 w.argmax,
                 # w.bincount,     # XXX: input dtypes
                 w.real, w.imag,
                 w.angle, w.real_if_close,
                 w.isreal, w.isrealobj, w.iscomplex, w.iscomplexobj,
                 w.isneginf, w.isposinf, w.i0
]

one_arg_funcs += [getattr(w, name) for name in _unary_ufuncs.__all__]


@pytest.mark.parametrize('func', one_arg_funcs)
class TestOneArg:
    """Base for smoke tests of one-arg functions.

    Accepts array_likes, torch.Tensors, w.ndarays; returns an ndarray
    """
    def test_asarray_tensor(self, func):
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        ta = func(t)

        assert isinstance(ta, w.ndarray)

    def test_asarray_list(self, func):
        lst = [[1, 2, 3], [4, 5, 6]]
        la = func(lst)

        assert isinstance(la, w.ndarray)

    def test_asarray_array(self, func):
        a = w.asarray([[1, 2, 3], [4, 5, 6]])
        la = func(a)

        assert isinstance(la, w.ndarray)

