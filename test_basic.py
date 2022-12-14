import functools
import pytest

import numpy as np
import torch

import wrapper as w
import _unary_ufuncs

# These function receive one array_like arg and return one array_like result
one_arg_funcs = [w.asarray, w.empty_like, w.ones_like, w.zeros_like,
                 functools.partial(w.full_like, fill_value=42),
                 w.corrcoef, w.squeeze,
                 w.argmax,
                 # w.bincount,     # XXX: input dtypes
                 w.prod,
                 w.real, w.imag,
                 w.angle, w.real_if_close,
                 w.isreal, w.isrealobj, w.iscomplex, w.iscomplexobj,
                 w.isneginf, w.isposinf, w.i0,
                 w.copy, w.array,]


one_arg_funcs += [getattr(w, name) for name in _unary_ufuncs.__all__]


@pytest.mark.parametrize('func', one_arg_funcs)
class TestOneArr:
    """Base for smoke tests of one-arg functions: (array_like) -> (array_like)

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


one_arg_axis_funcs = [w.argmax, w.prod]


@pytest.mark.parametrize('func', one_arg_axis_funcs)
@pytest.mark.parametrize('axis', [0, 1, -1, None])
class TestOneArrAndAxis:
    """Smoke test of functions (array_like, axis) -> array_like
    """
    def test_tensor(self, func, axis):
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        ta = func(t, axis=axis)
        assert isinstance(ta, w.ndarray)

    def test_list(self, func, axis):
        t = [[1, 2, 3], [4, 5, 6]]
        ta = func(t, axis=axis)
        assert isinstance(ta, w.ndarray)

    def test_array(self, func, axis):
        t = w.asarray([[1, 2, 3], [4, 5, 6]])
        ta = func(t, axis=axis)
        assert isinstance(ta, w.ndarray)


arr_shape_funcs = [w.reshape, w.ones_like, w.empty_like, w.ones_like,
                   functools.partial(w.full_like, fill_value=42),
                   w.broadcast_to,]


@pytest.mark.parametrize('func', arr_shape_funcs)
class TestOneArrAndShape:
    """Smoke test of functions (array_like, shape_like) -> array_like
    """
    shape = (2, 3)
    shape_arg_name = {w.reshape: 'newshape', }  # reshape expects `newshape`

    def test_tensor(self, func):
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])

        shape_dict = {self.shape_arg_name.get(func, 'shape'): self.shape}
        ta = func(t, **shape_dict)
        assert isinstance(ta, w.ndarray)
        assert ta.shape == self.shape

    def test_list(self, func):
        t = [[1, 2, 3], [4, 5, 6]]

        shape_dict = {self.shape_arg_name.get(func, 'shape'): self.shape}
        ta = func(t, **shape_dict)
        assert isinstance(ta, w.ndarray)
        assert ta.shape == self.shape

    def test_array(self, func):
        t = w.asarray([[1, 2, 3], [4, 5, 6]])

        shape_dict = {self.shape_arg_name.get(func, 'shape'): self.shape}
        ta = func(t, **shape_dict)
        assert isinstance(ta, w.ndarray)
        assert ta.shape == self.shape


one_arg_scalar_funcs = [(w.size, np.size),
                        (w.shape, np.shape),
                        (w.ndim, np.ndim)]


@pytest.mark.parametrize('func, np_func', one_arg_scalar_funcs)
class TestOneArrToScalar:
    """Smoke test of functions (array_like) -> scalar or python object.
    """
    def test_tensor(self, func, np_func):
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        ta = func(t)
        tn = np_func(np.asarray(t))

        assert not isinstance(ta, w.ndarray)
        assert ta == tn

    def test_list(self, func, np_func):
        t = [[1, 2, 3], [4, 5, 6]]
        ta = func(t)
        tn = np_func(t)

        assert not isinstance(ta, w.ndarray)
        assert ta == tn

    def test_array(self, func, np_func):
        t = w.asarray([[1, 2, 3], [4, 5, 6]])
        ta = func(t)
        tn = np_func(t)

        assert not isinstance(ta, w.ndarray)
        assert ta == tn


shape_funcs = [w.zeros, w.empty, w.ones,
               functools.partial(w.full, fill_value=42)]


@pytest.mark.parametrize('func', shape_funcs)
class TestShapeLikeToArray:
    """Smoke test (shape_like) -> array.
    """
    shape = (3, 4)

    def test_shape(self, func):
        a = func(self.shape)

        assert isinstance(a, w.ndarray)
        assert a.shape == self.shape


seq_funcs = [w.atleast_1d, w.atleast_2d, w.atleast_3d, w.broadcast_arrays]


@pytest.mark.parametrize('func', seq_funcs)
class TestSequenceOfArrays:
    """Smoke test (sequence of arrays) -> (sequence of arrays).
    """
    def test_single_tensor(self, func):
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        ta = func(t)

        # for a single argument, broadcast_arrays returns a tuple, while
        # atleast_?d return an array
        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = ta[0] if unpack else ta

        assert isinstance(res, w.ndarray)

    def test_single_list(self, func):
        lst = [[1, 2, 3], [4, 5, 6]]
        la = func(lst)

        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = la[0] if unpack else la

        assert isinstance(res, w.ndarray)

    def test_single_array(self, func):
        a = w.asarray([[1, 2, 3], [4, 5, 6]])
        la = func(a)

        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = la[0] if unpack else la

        assert isinstance(res, w.ndarray)

    def test_several(self, func):
        arys = (torch.Tensor([[1, 2, 3], [4, 5, 6]]),
                w.asarray([[1, 2, 3], [4, 5, 6]]),
                [[1, 2, 3], [4, 5, 6]],)

        result = func(*arys)
        assert isinstance(result, tuple)
        assert len(result) == len(arys)
        assert all(isinstance(_, w.ndarray) for _ in result)


seq_to_single_funcs = [w.concatenate]


@pytest.mark.parametrize('func', seq_to_single_funcs)
class TestSequenceOfArraysToSingle:
    """Smoke test (sequence of arrays) -> (array).
    """
    def test_several(self, func):
        arys = (torch.Tensor([[1, 2, 3], [4, 5, 6]]),
                w.asarray([[1, 2, 3], [4, 5, 6]]),
                [[1, 2, 3], [4, 5, 6]],)

        result = func(arys)
        assert isinstance(result, w.ndarray)


funcs_and_args = [
    (w.linspace, (0, 10, 11)),
    (w.eye, (5, 6)),
    (w.identity, (3,))
]


@pytest.mark.parametrize('func, args', funcs_and_args)
class TestPythonArgsToArray:
    """Smoke_test (sequence of scalars) -> (array)
    """
    def test_simple(self, func, args):
        a = func(*args)
        assert isinstance(a, w.ndarray)
