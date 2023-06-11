import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager

from numpy.compat import pickle

import pathlib
import builtins
from decimal import Decimal
import mmap

import torch_np as np

import numpy.core._multiarray_tests as _multiarray_tests
#from numpy.core._rational_tests import rational

from torch_np.testing import (
    assert_, assert_warns, assert_equal, assert_almost_equal,
    assert_array_equal, assert_raises_regex, assert_array_almost_equal,
    assert_allclose, # IS_PYPY, IS_PYSTON, HAS_REFCOUNT,
    assert_array_less,
    # runstring, temppath, 
    suppress_warnings, # break_cycles,
    )
from pytest import raises as assert_raises

IS_PYPY = False
IS_PYSTON = False
HAS_REFCOUNT = True

from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version

# Need to test an object that does not fully implement math interface
# from datetime import timedelta, datetime


# #### stubs to make pytest pass the collections stage ####

# defined in numpy/testing/_utils.py
def runstring(astr, dict):
    exec(astr, dict)

@contextmanager
def temppath(*args, **kwargs):
    """Context manager for temporary files.

    Context manager that returns the path to a closed temporary file. Its
    parameters are the same as for tempfile.mkstemp and are passed directly
    to that function. The underlying file is removed when the context is
    exited, so it should be closed at that time.

    Windows does not allow a temporary file to be opened if it is already
    open, so the underlying file must be closed after opening before it
    can be opened again.

    """
    fd, path = mkstemp(*args, **kwargs)
    os.close(fd)
    try:
        yield path
    finally:
        os.remove(path)


# FIXME
np.asanyarray = np.asarray
np.asfortranarray = np.asarray

# #### end stubs


def _aligned_zeros(shape, dtype=float, order="C", align=None):
    """
    Allocate a new ndarray with aligned memory.

    The ndarray is guaranteed *not* aligned to twice the requested alignment.
    Eg, if align=4, guarantees it is not aligned to 8. If align=None uses
    dtype.alignment."""
    dtype = np.dtype(dtype)
    if dtype == np.dtype(object):
        # Can't do this, fall back to standard allocation (which
        # should always be sufficiently aligned)
        if align is not None:
            raise ValueError("object array alignment not supported")
        return np.zeros(shape, dtype=dtype, order=order)
    if align is None:
        align = dtype.alignment
    if not hasattr(shape, '__len__'):
        shape = (shape,)
    size = functools.reduce(operator.mul, shape) * dtype.itemsize
    buf = np.empty(size + 2*align + 1, np.uint8)

    ptr = buf.__array_interface__['data'][0]
    offset = ptr % align
    if offset != 0:
        offset = align - offset
    if (ptr % (2*align)) == 0:
        offset += align

    # Note: slices producing 0-size arrays do not necessarily change
    # data pointer --- so we use and allocate size+1
    buf = buf[offset:offset+size+1][:-1]
    buf.fill(0)
    data = np.ndarray(shape, dtype, buf, order=order)
    return data


@pytest.mark.xfail(reason='TODO: flags')
class TestFlags:
    def setup_method(self):
        self.a = np.arange(10)

    def test_writeable(self):
        mydict = locals()
        self.a.flags.writeable = False
        assert_raises(ValueError, runstring, 'self.a[0] = 3', mydict)
        assert_raises(ValueError, runstring, 'self.a[0:1].itemset(3)', mydict)
        self.a.flags.writeable = True
        self.a[0] = 5
        self.a[0] = 0

    def test_writeable_any_base(self):
        # Ensure that any base being writeable is sufficient to change flag;
        # this is especially interesting for arrays from an array interface.
        arr = np.arange(10)

        class subclass(np.ndarray):
            pass

        # Create subclass so base will not be collapsed, this is OK to change
        view1 = arr.view(subclass)
        view2 = view1[...]
        arr.flags.writeable = False
        view2.flags.writeable = False
        view2.flags.writeable = True  # Can be set to True again.

        arr = np.arange(10)

        class frominterface:
            def __init__(self, arr):
                self.arr = arr
                self.__array_interface__ = arr.__array_interface__

        view1 = np.asarray(frominterface)
        view2 = view1[...]
        view2.flags.writeable = False
        view2.flags.writeable = True

        view1.flags.writeable = False
        view2.flags.writeable = False
        with assert_raises(ValueError):
            # Must assume not writeable, since only base is not:
            view2.flags.writeable = True

    def test_writeable_from_readonly(self):
        # gh-9440 - make sure fromstring, from buffer on readonly buffers
        # set writeable False
        data = b'\x00' * 100
        vals = np.frombuffer(data, 'B')
        assert_raises(ValueError, vals.setflags, write=True)
        types = np.dtype( [('vals', 'u1'), ('res3', 'S4')] )
        values = np.core.records.fromstring(data, types)
        vals = values['vals']
        assert_raises(ValueError, vals.setflags, write=True)

    def test_writeable_from_buffer(self):
        data = bytearray(b'\x00' * 100)
        vals = np.frombuffer(data, 'B')
        assert_(vals.flags.writeable)
        vals.setflags(write=False)
        assert_(vals.flags.writeable is False)
        vals.setflags(write=True)
        assert_(vals.flags.writeable)
        types = np.dtype( [('vals', 'u1'), ('res3', 'S4')] )
        values = np.core.records.fromstring(data, types)
        vals = values['vals']
        assert_(vals.flags.writeable)
        vals.setflags(write=False)
        assert_(vals.flags.writeable is False)
        vals.setflags(write=True)
        assert_(vals.flags.writeable)

    @pytest.mark.skipif(IS_PYPY, reason="PyPy always copies")
    def test_writeable_pickle(self):
        import pickle
        # Small arrays will be copied without setting base.
        # See condition for using PyArray_SetBaseObject in
        # array_setstate.
        a = np.arange(1000)
        for v in range(pickle.HIGHEST_PROTOCOL):
            vals = pickle.loads(pickle.dumps(a, v))
            assert_(vals.flags.writeable)
            assert_(isinstance(vals.base, bytes))

    def test_writeable_from_c_data(self):
        # Test that the writeable flag can be changed for an array wrapping
        # low level C-data, but not owning its data.
        # Also see that this is deprecated to change from python.
        from numpy.core._multiarray_tests import get_c_wrapping_array

        arr_writeable = get_c_wrapping_array(True)
        assert not arr_writeable.flags.owndata
        assert arr_writeable.flags.writeable
        view = arr_writeable[...]

        # Toggling the writeable flag works on the view:
        view.flags.writeable = False
        assert not view.flags.writeable
        view.flags.writeable = True
        assert view.flags.writeable
        # Flag can be unset on the arr_writeable:
        arr_writeable.flags.writeable = False

        arr_readonly = get_c_wrapping_array(False)
        assert not arr_readonly.flags.owndata
        assert not arr_readonly.flags.writeable

        for arr in [arr_writeable, arr_readonly]:
            view = arr[...]
            view.flags.writeable = False  # make sure it is readonly
            arr.flags.writeable = False
            assert not arr.flags.writeable

            with assert_raises(ValueError):
                view.flags.writeable = True

            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                with assert_raises(DeprecationWarning):
                    arr.flags.writeable = True

            with assert_warns(DeprecationWarning):
                arr.flags.writeable = True

    def test_warnonwrite(self):
        a = np.arange(10)
        a.flags._warn_on_write = True
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always')
            a[1] = 10
            a[2] = 10
            # only warn once
            assert_(len(w) == 1)

    @pytest.mark.parametrize(["flag", "flag_value", "writeable"],
            [("writeable", True, True),
             # Delete _warn_on_write after deprecation and simplify
             # the parameterization:
             ("_warn_on_write", True, False),
             ("writeable", False, False)])
    def test_readonly_flag_protocols(self, flag, flag_value, writeable):
        a = np.arange(10)
        setattr(a.flags, flag, flag_value)

        class MyArr():
            __array_struct__ = a.__array_struct__

        assert memoryview(a).readonly is not writeable
        assert a.__array_interface__['data'][1] is not writeable
        assert np.asarray(MyArr()).flags.writeable is writeable

    def test_otherflags(self):
        assert_equal(self.a.flags.carray, True)
        assert_equal(self.a.flags['C'], True)
        assert_equal(self.a.flags.farray, False)
        assert_equal(self.a.flags.behaved, True)
        assert_equal(self.a.flags.fnc, False)
        assert_equal(self.a.flags.forc, True)
        assert_equal(self.a.flags.owndata, True)
        assert_equal(self.a.flags.writeable, True)
        assert_equal(self.a.flags.aligned, True)
        assert_equal(self.a.flags.writebackifcopy, False)
        assert_equal(self.a.flags['X'], False)
        assert_equal(self.a.flags['WRITEBACKIFCOPY'], False)

    def test_string_align(self):
        a = np.zeros(4, dtype=np.dtype('|S4'))
        assert_(a.flags.aligned)
        # not power of two are accessed byte-wise and thus considered aligned
        a = np.zeros(5, dtype=np.dtype('|S4'))
        assert_(a.flags.aligned)

    def test_void_align(self):
        a = np.zeros(4, dtype=np.dtype([("a", "i4"), ("b", "i4")]))
        assert_(a.flags.aligned)


@pytest.mark.xfail(reason='TODO: hash')
class TestHash:
    # see #3793
    def test_int(self):
        for st, ut, s in [(np.int8, np.uint8, 8),
                          (np.int16, np.uint16, 16),
                          (np.int32, np.uint32, 32),
                          (np.int64, np.uint64, 64)]:
            for i in range(1, s):
                assert_equal(hash(st(-2**i)), hash(-2**i),
                             err_msg="%r: -2**%d" % (st, i))
                assert_equal(hash(st(2**(i - 1))), hash(2**(i - 1)),
                             err_msg="%r: 2**%d" % (st, i - 1))
                assert_equal(hash(st(2**i - 1)), hash(2**i - 1),
                             err_msg="%r: 2**%d - 1" % (st, i))

                i = max(i - 1, 1)
                assert_equal(hash(ut(2**(i - 1))), hash(2**(i - 1)),
                             err_msg="%r: 2**%d" % (ut, i - 1))
                assert_equal(hash(ut(2**i - 1)), hash(2**i - 1),
                             err_msg="%r: 2**%d - 1" % (ut, i))


@pytest.mark.xfail(reason='TODO: hash')
class TestAttributes:
    def setup_method(self):
        self.one = np.arange(10)
        self.two = np.arange(20).reshape(4, 5)
        self.three = np.arange(60, dtype=np.float64).reshape(2, 5, 6)

    def test_attributes(self):
        assert_equal(self.one.shape, (10,))
        assert_equal(self.two.shape, (4, 5))
        assert_equal(self.three.shape, (2, 5, 6))
        self.three.shape = (10, 3, 2)
        assert_equal(self.three.shape, (10, 3, 2))
        self.three.shape = (2, 5, 6)
        assert_equal(self.one.strides, (self.one.itemsize,))
        num = self.two.itemsize
        assert_equal(self.two.strides, (5*num, num))
        num = self.three.itemsize
        assert_equal(self.three.strides, (30*num, 6*num, num))
        assert_equal(self.one.ndim, 1)
        assert_equal(self.two.ndim, 2)
        assert_equal(self.three.ndim, 3)
        num = self.two.itemsize
        assert_equal(self.two.size, 20)
        assert_equal(self.two.nbytes, 20*num)
        assert_equal(self.two.itemsize, self.two.dtype.itemsize)
        assert_equal(self.two.base, np.arange(20))

    def test_dtypeattr(self):
        assert_equal(self.one.dtype, np.dtype(np.int_))
        assert_equal(self.three.dtype, np.dtype(np.float_))
        assert_equal(self.one.dtype.char, 'l')
        assert_equal(self.three.dtype.char, 'd')
        assert_(self.three.dtype.str[0] in '<>')
        assert_equal(self.one.dtype.str[1], 'i')
        assert_equal(self.three.dtype.str[1], 'f')

    def test_int_subclassing(self):
        # Regression test for https://github.com/numpy/numpy/pull/3526

        numpy_int = np.int_(0)

        # int_ doesn't inherit from Python int, because it's not fixed-width
        assert_(not isinstance(numpy_int, int))

    def test_stridesattr(self):
        x = self.one

        def make_array(size, offset, strides):
            return np.ndarray(size, buffer=x, dtype=int,
                              offset=offset*x.itemsize,
                              strides=strides*x.itemsize)

        assert_equal(make_array(4, 4, -1), np.array([4, 3, 2, 1]))
        assert_raises(ValueError, make_array, 4, 4, -2)
        assert_raises(ValueError, make_array, 4, 2, -1)
        assert_raises(ValueError, make_array, 8, 3, 1)
        assert_equal(make_array(8, 3, 0), np.array([3]*8))
        # Check behavior reported in gh-2503:
        assert_raises(ValueError, make_array, (2, 3), 5, np.array([-2, -3]))
        make_array(0, 0, 10)

    def test_set_stridesattr(self):
        x = self.one

        def make_array(size, offset, strides):
            try:
                r = np.ndarray([size], dtype=int, buffer=x,
                               offset=offset*x.itemsize)
            except Exception as e:
                raise RuntimeError(e)
            r.strides = strides = strides*x.itemsize
            return r

        assert_equal(make_array(4, 4, -1), np.array([4, 3, 2, 1]))
        assert_equal(make_array(7, 3, 1), np.array([3, 4, 5, 6, 7, 8, 9]))
        assert_raises(ValueError, make_array, 4, 4, -2)
        assert_raises(ValueError, make_array, 4, 2, -1)
        assert_raises(RuntimeError, make_array, 8, 3, 1)
        # Check that the true extent of the array is used.
        # Test relies on as_strided base not exposing a buffer.
        x = np.lib.stride_tricks.as_strided(np.arange(1), (10, 10), (0, 0))

        def set_strides(arr, strides):
            arr.strides = strides

        assert_raises(ValueError, set_strides, x, (10*x.itemsize, x.itemsize))

        # Test for offset calculations:
        x = np.lib.stride_tricks.as_strided(np.arange(10, dtype=np.int8)[-1],
                                                    shape=(10,), strides=(-1,))
        assert_raises(ValueError, set_strides, x[::-1], -1)
        a = x[::-1]
        a.strides = 1
        a[::2].strides = 2

        # test 0d
        arr_0d = np.array(0)
        arr_0d.strides = ()
        assert_raises(TypeError, set_strides, arr_0d, None)

    def test_fill(self):
        for t in "?bhilqpBHILQPfdgFDGO":
            x = np.empty((3, 2, 1), t)
            y = np.empty((3, 2, 1), t)
            x.fill(1)
            y[...] = 1
            assert_equal(x, y)

    def test_fill_max_uint64(self):
        x = np.empty((3, 2, 1), dtype=np.uint64)
        y = np.empty((3, 2, 1), dtype=np.uint64)
        value = 2**64 - 1
        y[...] = value
        x.fill(value)
        assert_array_equal(x, y)

    def test_fill_struct_array(self):
        # Filling from a scalar
        x = np.array([(0, 0.0), (1, 1.0)], dtype='i4,f8')
        x.fill(x[0])
        assert_equal(x['f1'][1], x['f1'][0])
        # Filling from a tuple that can be converted
        # to a scalar
        x = np.zeros(2, dtype=[('a', 'f8'), ('b', 'i4')])
        x.fill((3.5, -2))
        assert_array_equal(x['a'], [3.5, 3.5])
        assert_array_equal(x['b'], [-2, -2])

    def test_fill_readonly(self):
        # gh-22922
        a = np.zeros(11)
        a.setflags(write=False)
        with pytest.raises(ValueError, match=".*read-only"):
            a.fill(0)


class TestArrayConstruction:
    def test_array(self):
        d = np.ones(6)
        r = np.array([d, d])
        assert_equal(r, np.ones((2, 6)))

        d = np.ones(6)
        tgt = np.ones((2, 6))
        r = np.array([d, d])
        assert_equal(r, tgt)
        tgt[1] = 2
        r = np.array([d, d + 1])
        assert_equal(r, tgt)

        d = np.ones(6)
        r = np.array([[d, d]])
        assert_equal(r, np.ones((1, 2, 6)))

        d = np.ones(6)
        r = np.array([[d, d], [d, d]])
        assert_equal(r, np.ones((2, 2, 6)))

        d = np.ones((6, 6))
        r = np.array([d, d])
        assert_equal(r, np.ones((2, 6, 6)))

        tgt = np.ones((2, 3), dtype=bool)
        tgt[0, 2] = False
        tgt[1, 0:2] = False
        r = np.array([[True, True, False], [False, False, True]])
        assert_equal(r, tgt)
        r = np.array([[True, False], [True, False], [False, True]])
        assert_equal(r, tgt.T)

    @pytest.mark.skip(reason="object arrays")
    def test_array_object(self):
        d = np.ones((6, ))
        r = np.array([[d, d + 1], d + 2], dtype=object)
        assert_equal(len(r), 2)
        assert_equal(r[0], [d, d + 1])
        assert_equal(r[1], d + 2)

    def test_array_empty(self):
        assert_raises(TypeError, np.array)

    def test_0d_array_shape(self):
        assert np.ones(np.array(3)).shape == (3,)

    def test_array_copy_false(self):
        d = np.array([1, 2, 3])
        e = np.array(d, copy=False)
        d[1] = 3
        assert_array_equal(e, [1, 3, 3])

    @pytest.mark.xfail(reason="order='F'")
    def test_array_copy_false_2(self):
        d = np.array([1, 2, 3])
        e = np.array(d, copy=False, order='F')
        d[1] = 4
        assert_array_equal(e, [1, 4, 3])
        e[2] = 7
        assert_array_equal(d, [1, 4, 7])

    def test_array_copy_true(self):
        d = np.array([[1,2,3], [1, 2, 3]])
        e = np.array(d, copy=True)
        d[0, 1] = 3
        e[0, 2] = -7
        assert_array_equal(e, [[1, 2, -7], [1, 2, 3]])
        assert_array_equal(d, [[1, 3, 3], [1, 2, 3]])

    @pytest.mark.xfail(reason="order='F'")
    def test_array_copy_true_2(self):
        d = np.array([[1,2,3], [1, 2, 3]])
        e = np.array(d, copy=True, order='F')
        d[0, 1] = 5
        e[0, 2] = 7
        assert_array_equal(e, [[1, 3, 7], [1, 2, 3]])
        assert_array_equal(d, [[1, 5, 3], [1,2,3]])

    def test_array_cont(self):
        d = np.ones(10)[::2]
        assert_(np.ascontiguousarray(d).flags.c_contiguous)
        assert_(np.ascontiguousarray(d).flags.f_contiguous)
        assert_(np.asfortranarray(d).flags.c_contiguous)
        # assert_(np.asfortranarray(d).flags.f_contiguous)   # XXX: f ordering
        d = np.ones((10, 10))[::2,::2]
        assert_(np.ascontiguousarray(d).flags.c_contiguous)
        # assert_(np.asfortranarray(d).flags.f_contiguous)

    @pytest.mark.parametrize("func",
            [np.array,
             np.asarray,
             np.asanyarray,
             np.ascontiguousarray,
             np.asfortranarray])
    def test_bad_arguments_error(self, func):
        with pytest.raises(TypeError):
            func(3, dtype="bad dtype")
        with pytest.raises(TypeError):
            func()  # missing arguments
        with pytest.raises(TypeError):
            func(1, 2, 3, 4, 5, 6, 7, 8)  # too many arguments

    @pytest.mark.skip(reason="np.array w/keyword argument")
    @pytest.mark.parametrize("func",
            [np.array,
             np.asarray,
             np.asanyarray,
             np.ascontiguousarray,
             np.asfortranarray])
    def test_array_as_keyword(self, func):
        # This should likely be made positional only, but do not change
        # the name accidentally.
        if func is np.array:
            func(object=3)
        else:
            func(a=3)


class TestAssignment:
    def test_assignment_broadcasting(self):
        a = np.arange(6).reshape(2, 3)

        # Broadcasting the input to the output
        a[...] = np.arange(3)
        assert_equal(a, [[0, 1, 2], [0, 1, 2]])
        a[...] = np.arange(2).reshape(2, 1)
        assert_equal(a, [[0, 0, 0], [1, 1, 1]])

        # For compatibility with <= 1.5, a limited version of broadcasting
        # the output to the input.
        #
        # This behavior is inconsistent with NumPy broadcasting
        # in general, because it only uses one of the two broadcasting
        # rules (adding a new "1" dimension to the left of the shape),
        # applied to the output instead of an input. In NumPy 2.0, this kind
        # of broadcasting assignment will likely be disallowed.
        a[...] = np.flip(np.arange(6)).reshape(1, 2, 3)
        assert_equal(a, [[5, 4, 3], [2, 1, 0]])
        # The other type of broadcasting would require a reduction operation.

        def assign(a, b):
            a[...] = b

        assert_raises((RuntimeError, ValueError), assign, a, np.arange(12).reshape(2, 2, 3))

    def test_assignment_errors(self):
        # Address issue #2276
        class C:
            pass
        a = np.zeros(1)

        def assign(v):
            a[0] = v

        assert_raises((RuntimeError, TypeError), assign, C())
        # assert_raises((TypeError, ValueError), assign, [1])  # numpy raises, we do not

    @pytest.mark.skip(reason="object arrays")
    def test_unicode_assignment(self):
        # gh-5049
        from numpy.core.numeric import set_string_function

        @contextmanager
        def inject_str(s):
            """ replace ndarray.__str__ temporarily """
            set_string_function(lambda x: s, repr=False)
            try:
                yield
            finally:
                set_string_function(None, repr=False)

        a1d = np.array(['test'])
        a0d = np.array('done')
        with inject_str('bad'):
            a1d[0] = a0d  # previously this would invoke __str__
        assert_equal(a1d[0], 'done')

        # this would crash for the same reason
        np.array([np.array('\xe5\xe4\xf6')])

    @pytest.mark.skip(reason="object arrays")
    def test_stringlike_empty_list(self):
        # gh-8902
        u = np.array(['done'])
        b = np.array([b'done'])

        class bad_sequence:
            def __getitem__(self): pass
            def __len__(self): raise RuntimeError

        assert_raises(ValueError, operator.setitem, u, 0, [])
        assert_raises(ValueError, operator.setitem, b, 0, [])

        assert_raises(ValueError, operator.setitem, u, 0, bad_sequence())
        assert_raises(ValueError, operator.setitem, b, 0, bad_sequence())

    @pytest.mark.skip(reason="longdouble")
    def test_longdouble_assignment(self):
        # only relevant if longdouble is larger than float
        # we're looking for loss of precision

        for dtype in (np.longdouble, np.longcomplex):
            # gh-8902
            tinyb = np.nextafter(np.longdouble(0), 1).astype(dtype)
            tinya = np.nextafter(np.longdouble(0), -1).astype(dtype)

            # construction
            tiny1d = np.array([tinya])
            assert_equal(tiny1d[0], tinya)

            # scalar = scalar
            tiny1d[0] = tinyb
            assert_equal(tiny1d[0], tinyb)

            # 0d = scalar
            tiny1d[0, ...] = tinya
            assert_equal(tiny1d[0], tinya)

            # 0d = 0d
            tiny1d[0, ...] = tinyb[...]
            assert_equal(tiny1d[0], tinyb)

            # scalar = 0d
            tiny1d[0] = tinyb[...]
            assert_equal(tiny1d[0], tinyb)

            arr = np.array([np.array(tinya)])
            assert_equal(arr[0], tinya)

    @pytest.mark.skip(reason="object arrays")
    def test_cast_to_string(self):
        # cast to str should do "str(scalar)", not "str(scalar.item())"
        # Example: In python2, str(float) is truncated, so we want to avoid
        # str(np.float64(...).item()) as this would incorrectly truncate.
        a = np.zeros(1, dtype='S20')
        a[:] = np.array(['1.12345678901234567890'], dtype='f8')
        assert_equal(a[0], b"1.1234567890123457")


@pytest.mark.xfail(reason='TODO: dtypes')
class TestDtypedescr:
    def test_construction(self):
        d1 = np.dtype('i4')
        assert_equal(d1, np.dtype(np.int32))
        d2 = np.dtype('f8')
        assert_equal(d2, np.dtype(np.float64))

    def test_byteorders(self):
        assert_(np.dtype('<i4') != np.dtype('>i4'))
        assert_(np.dtype([('a', '<i4')]) != np.dtype([('a', '>i4')]))

    def test_structured_non_void(self):
        fields = [('a', '<i2'), ('b', '<i2')]
        dt_int = np.dtype(('i4', fields))
        assert_equal(str(dt_int), "(numpy.int32, [('a', '<i2'), ('b', '<i2')])")

        # gh-9821
        arr_int = np.zeros(4, dt_int)
        assert_equal(repr(arr_int),
            "array([0, 0, 0, 0], dtype=(numpy.int32, [('a', '<i2'), ('b', '<i2')]))")


@pytest.mark.xfail(reason='TODO: zero-rank?')
class TestZeroRank:
    def setup_method(self):
        self.d = np.array(0), np.array('x', object)

    def test_ellipsis_subscript(self):
        a, b = self.d
        assert_equal(a[...], 0)
        assert_equal(b[...], 'x')
        assert_(a[...].base is a)  # `a[...] is a` in numpy <1.9.
        assert_(b[...].base is b)  # `b[...] is b` in numpy <1.9.

    def test_empty_subscript(self):
        a, b = self.d
        assert_equal(a[()], 0)
        assert_equal(b[()], 'x')
        assert_(type(a[()]) is a.dtype.type)
        assert_(type(b[()]) is str)

    def test_invalid_subscript(self):
        a, b = self.d
        assert_raises(IndexError, lambda x: x[0], a)
        assert_raises(IndexError, lambda x: x[0], b)
        assert_raises(IndexError, lambda x: x[np.array([], int)], a)
        assert_raises(IndexError, lambda x: x[np.array([], int)], b)

    def test_ellipsis_subscript_assignment(self):
        a, b = self.d
        a[...] = 42
        assert_equal(a, 42)
        b[...] = ''
        assert_equal(b.item(), '')

    def test_empty_subscript_assignment(self):
        a, b = self.d
        a[()] = 42
        assert_equal(a, 42)
        b[()] = ''
        assert_equal(b.item(), '')

    def test_invalid_subscript_assignment(self):
        a, b = self.d

        def assign(x, i, v):
            x[i] = v

        assert_raises(IndexError, assign, a, 0, 42)
        assert_raises(IndexError, assign, b, 0, '')
        assert_raises(ValueError, assign, a, (), '')

    def test_newaxis(self):
        a, b = self.d
        assert_equal(a[np.newaxis].shape, (1,))
        assert_equal(a[..., np.newaxis].shape, (1,))
        assert_equal(a[np.newaxis, ...].shape, (1,))
        assert_equal(a[..., np.newaxis].shape, (1,))
        assert_equal(a[np.newaxis, ..., np.newaxis].shape, (1, 1))
        assert_equal(a[..., np.newaxis, np.newaxis].shape, (1, 1))
        assert_equal(a[np.newaxis, np.newaxis, ...].shape, (1, 1))
        assert_equal(a[(np.newaxis,)*10].shape, (1,)*10)

    def test_invalid_newaxis(self):
        a, b = self.d

        def subscript(x, i):
            x[i]

        assert_raises(IndexError, subscript, a, (np.newaxis, 0))
        assert_raises(IndexError, subscript, a, (np.newaxis,)*50)

    def test_constructor(self):
        x = np.ndarray(())
        x[()] = 5
        assert_equal(x[()], 5)
        y = np.ndarray((), buffer=x)
        y[()] = 6
        assert_equal(x[()], 6)

        # strides and shape must be the same length
        with pytest.raises(ValueError):
            np.ndarray((2,), strides=())
        with pytest.raises(ValueError):
            np.ndarray((), strides=(2,))

    def test_output(self):
        x = np.array(2)
        assert_raises(ValueError, np.add, x, [1], x)

    def test_real_imag(self):
        # contiguity checks are for gh-11245
        x = np.array(1j)
        xr = x.real
        xi = x.imag

        assert_equal(xr, np.array(0))
        assert_(type(xr) is np.ndarray)
        assert_equal(xr.flags.contiguous, True)
        assert_equal(xr.flags.f_contiguous, True)

        assert_equal(xi, np.array(1))
        assert_(type(xi) is np.ndarray)
        assert_equal(xi.flags.contiguous, True)
        assert_equal(xi.flags.f_contiguous, True)


class TestScalarIndexing:
    def setup_method(self):
        self.d = np.array([0, 1])[0]

    def test_ellipsis_subscript(self):
        a = self.d
        assert_equal(a[...], 0)
        assert_equal(a[...].shape, ())

    def test_empty_subscript(self):
        a = self.d
        assert_equal(a[()], 0)
        assert_equal(a[()].shape, ())

    def test_invalid_subscript(self):
        a = self.d
        assert_raises(IndexError, lambda x: x[0], a)
        assert_raises(IndexError, lambda x: x[np.array([], int)], a)

    def test_invalid_subscript_assignment(self):
        a = self.d

        def assign(x, i, v):
            x[i] = v

        assert_raises((IndexError, TypeError), assign, a, 0, 42)

    def test_newaxis(self):
        a = self.d
        assert_equal(a[np.newaxis].shape, (1,))
        assert_equal(a[..., np.newaxis].shape, (1,))
        assert_equal(a[np.newaxis, ...].shape, (1,))
        assert_equal(a[..., np.newaxis].shape, (1,))
        assert_equal(a[np.newaxis, ..., np.newaxis].shape, (1, 1))
        assert_equal(a[..., np.newaxis, np.newaxis].shape, (1, 1))
        assert_equal(a[np.newaxis, np.newaxis, ...].shape, (1, 1))
        assert_equal(a[(np.newaxis,)*10].shape, (1,)*10)

    def test_invalid_newaxis(self):
        a = self.d

        def subscript(x, i):
            x[i]

        assert_raises(IndexError, subscript, a, (np.newaxis, 0))

        # this assersion fails because 50 > NPY_MAXDIMS = 32
        # assert_raises(IndexError, subscript, a, (np.newaxis,)*50)

    @pytest.mark.xfail(reason="pytorch disallows overlapping assignments")
    def test_overlapping_assignment(self):
        # With positive strides
        a = np.arange(4)
        a[:-1] = a[1:]
        assert_equal(a, [1, 2, 3, 3])

        a = np.arange(4)
        a[1:] = a[:-1]
        assert_equal(a, [0, 0, 1, 2])

        # With positive and negative strides
        a = np.arange(4)
        a[:] = a[::-1]
        assert_equal(a, [3, 2, 1, 0])

        a = np.arange(6).reshape(2, 3)
        a[::-1,:] = a[:, ::-1]
        assert_equal(a, [[5, 4, 3], [2, 1, 0]])

        a = np.arange(6).reshape(2, 3)
        a[::-1, ::-1] = a[:, ::-1]
        assert_equal(a, [[3, 4, 5], [0, 1, 2]])

        # With just one element overlapping
        a = np.arange(5)
        a[:3] = a[2:]
        assert_equal(a, [2, 3, 4, 3, 4])

        a = np.arange(5)
        a[2:] = a[:3]
        assert_equal(a, [0, 1, 0, 1, 2])

        a = np.arange(5)
        a[2::-1] = a[2:]
        assert_equal(a, [4, 3, 2, 3, 4])

        a = np.arange(5)
        a[2:] = a[2::-1]
        assert_equal(a, [0, 1, 2, 1, 0])

        a = np.arange(5)
        a[2::-1] = a[:1:-1]
        assert_equal(a, [2, 3, 4, 3, 4])

        a = np.arange(5)
        a[:1:-1] = a[2::-1]
        assert_equal(a, [0, 1, 0, 1, 2])


@pytest.mark.xfail(reason='TODO')
class TestCreation:
    """
    Test the np.array constructor
    """
    def test_from_attribute(self):
        class x:
            def __array__(self, dtype=None):
                pass

        assert_raises(ValueError, np.array, x())

    def test_from_string(self):
        types = np.typecodes['AllInteger'] + np.typecodes['Float']
        nstr = ['123', '123']
        result = np.array([123, 123], dtype=int)
        for type in types:
            msg = 'String conversion for %s' % type
            assert_equal(np.array(nstr, dtype=type), result, err_msg=msg)

    def test_void(self):
        arr = np.array([], dtype='V')
        assert arr.dtype == 'V8'  # current default
        # Same length scalars (those that go to the same void) work:
        arr = np.array([b"1234", b"1234"], dtype="V")
        assert arr.dtype == "V4"

        # Promoting different lengths will fail (pre 1.20 this worked)
        # by going via S5 and casting to V5.
        with pytest.raises(TypeError):
            np.array([b"1234", b"12345"], dtype="V")
        with pytest.raises(TypeError):
            np.array([b"12345", b"1234"], dtype="V")

        # Check the same for the casting path:
        arr = np.array([b"1234", b"1234"], dtype="O").astype("V")
        assert arr.dtype == "V4"
        with pytest.raises(TypeError):
            np.array([b"1234", b"12345"], dtype="O").astype("V")

    @pytest.mark.parametrize("idx",
            [pytest.param(Ellipsis, id="arr"), pytest.param((), id="scalar")])
    def test_structured_void_promotion(self, idx):
        arr = np.array(
            [np.array(1, dtype="i,i")[idx], np.array(2, dtype='i,i')[idx]],
            dtype="V")
        assert_array_equal(arr, np.array([(1, 1), (2, 2)], dtype="i,i"))
        # The following fails to promote the two dtypes, resulting in an error
        with pytest.raises(TypeError):
            np.array(
                [np.array(1, dtype="i,i")[idx], np.array(2, dtype='i,i,i')[idx]],
                dtype="V")


    def test_too_big_error(self):
        # 45341 is the smallest integer greater than sqrt(2**31 - 1).
        # 3037000500 is the smallest integer greater than sqrt(2**63 - 1).
        # We want to make sure that the square byte array with those dimensions
        # is too big on 32 or 64 bit systems respectively.
        if np.iinfo('intp').max == 2**31 - 1:
            shape = (46341, 46341)
        elif np.iinfo('intp').max == 2**63 - 1:
            shape = (3037000500, 3037000500)
        else:
            return
        assert_raises(ValueError, np.empty, shape, dtype=np.int8)
        assert_raises(ValueError, np.zeros, shape, dtype=np.int8)
        assert_raises(ValueError, np.ones, shape, dtype=np.int8)

    @pytest.mark.skipif(np.dtype(np.intp).itemsize != 8,
                        reason="malloc may not fail on 32 bit systems")
    def test_malloc_fails(self):
        # This test is guaranteed to fail due to a too large allocation
        with assert_raises(np.core._exceptions._ArrayMemoryError):
            np.empty(np.iinfo(np.intp).max, dtype=np.uint8)

    def test_zeros(self):
        types = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        for dt in types:
            d = np.zeros((13,), dtype=dt)
            assert_equal(np.count_nonzero(d), 0)
            # true for ieee floats
            assert_equal(d.sum(), 0)
            assert_(not d.any())

            d = np.zeros(2, dtype='(2,4)i4')
            assert_equal(np.count_nonzero(d), 0)
            assert_equal(d.sum(), 0)
            assert_(not d.any())

            d = np.zeros(2, dtype='4i4')
            assert_equal(np.count_nonzero(d), 0)
            assert_equal(d.sum(), 0)
            assert_(not d.any())

            d = np.zeros(2, dtype='(2,4)i4, (2,4)i4')
            assert_equal(np.count_nonzero(d), 0)

    @pytest.mark.slow
    def test_zeros_big(self):
        # test big array as they might be allocated different by the system
        types = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        for dt in types:
            d = np.zeros((30 * 1024**2,), dtype=dt)
            assert_(not d.any())
            # This test can fail on 32-bit systems due to insufficient
            # contiguous memory. Deallocating the previous array increases the
            # chance of success.
            del(d)

    def test_zeros_obj(self):
        # test initialization from PyLong(0)
        d = np.zeros((13,), dtype=object)
        assert_array_equal(d, [0] * 13)
        assert_equal(np.count_nonzero(d), 0)

    def test_zeros_obj_obj(self):
        d = np.zeros(10, dtype=[('k', object, 2)])
        assert_array_equal(d['k'], 0)

    def test_zeros_like_like_zeros(self):
        # test zeros_like returns the same as zeros
        for c in np.typecodes['All']:
            if c == 'V':
                continue
            d = np.zeros((3,3), dtype=c)
            assert_array_equal(np.zeros_like(d), d)
            assert_equal(np.zeros_like(d).dtype, d.dtype)
        # explicitly check some special cases
        d = np.zeros((3,3), dtype='S5')
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        d = np.zeros((3,3), dtype='U5')
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)

        d = np.zeros((3,3), dtype='<i4')
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        d = np.zeros((3,3), dtype='>i4')
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)

        d = np.zeros((3,3), dtype='<M8[s]')
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        d = np.zeros((3,3), dtype='>M8[s]')
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)

        d = np.zeros((3,3), dtype='f4,f4')
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)

    def test_empty_unicode(self):
        # don't throw decode errors on garbage memory
        for i in range(5, 100, 5):
            d = np.empty(i, dtype='U')
            str(d)

    def test_sequence_non_homogeneous(self):
        assert_equal(np.array([4, 2**80]).dtype, object)
        assert_equal(np.array([4, 2**80, 4]).dtype, object)
        assert_equal(np.array([2**80, 4]).dtype, object)
        assert_equal(np.array([2**80] * 3).dtype, object)
        assert_equal(np.array([[1, 1],[1j, 1j]]).dtype, complex)
        assert_equal(np.array([[1j, 1j],[1, 1]]).dtype, complex)
        assert_equal(np.array([[1, 1, 1],[1, 1j, 1.], [1, 1, 1]]).dtype, complex)

    def test_non_sequence_sequence(self):
        """Should not segfault.

        Class Fail breaks the sequence protocol for new style classes, i.e.,
        those derived from object. Class Map is a mapping type indicated by
        raising a ValueError. At some point we may raise a warning instead
        of an error in the Fail case.

        """
        class Fail:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise ValueError()

        class Map:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise KeyError()

        a = np.array([Map()])
        assert_(a.shape == (1,))
        assert_(a.dtype == np.dtype(object))
        assert_raises(ValueError, np.array, [Fail()])

    def test_no_len_object_type(self):
        # gh-5100, want object array from iterable object without len()
        class Point2:
            def __init__(self):
                pass

            def __getitem__(self, ind):
                if ind in [0, 1]:
                    return ind
                else:
                    raise IndexError()
        d = np.array([Point2(), Point2(), Point2()])
        assert_equal(d.dtype, np.dtype(object))

    def test_false_len_sequence(self):
        # gh-7264, segfault for this example
        class C:
            def __getitem__(self, i):
                raise IndexError
            def __len__(self):
                return 42

        a = np.array(C()) # segfault?
        assert_equal(len(a), 0)

    def test_false_len_iterable(self):
        # Special case where a bad __getitem__ makes us fall back on __iter__:
        class C:
            def __getitem__(self, x):
                raise Exception
            def __iter__(self):
                return iter(())
            def __len__(self):
                return 2

        a = np.empty(2)
        with assert_raises(ValueError):
            a[:] = C()  # Segfault!

        np.array(C()) == list(C())

    def test_failed_len_sequence(self):
        # gh-7393
        class A:
            def __init__(self, data):
                self._data = data
            def __getitem__(self, item):
                return type(self)(self._data[item])
            def __len__(self):
                return len(self._data)

        # len(d) should give 3, but len(d[0]) will fail
        d = A([1,2,3])
        assert_equal(len(np.array(d)), 3)

    def test_array_too_big(self):
        # Test that array creation succeeds for arrays addressable by intp
        # on the byte level and fails for too large arrays.
        buf = np.zeros(100)

        max_bytes = np.iinfo(np.intp).max
        for dtype in ["intp", "S20", "b"]:
            dtype = np.dtype(dtype)
            itemsize = dtype.itemsize

            np.ndarray(buffer=buf, strides=(0,),
                       shape=(max_bytes//itemsize,), dtype=dtype)
            assert_raises(ValueError, np.ndarray, buffer=buf, strides=(0,),
                          shape=(max_bytes//itemsize + 1,), dtype=dtype)

    def _ragged_creation(self, seq):
        # without dtype=object, the ragged object raises
        with pytest.raises(ValueError, match=".*detected shape was"):
            a = np.array(seq)

        return np.array(seq, dtype=object)

    def test_ragged_ndim_object(self):
        # Lists of mismatching depths are treated as object arrays
        a = self._ragged_creation([[1], 2, 3])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        a = self._ragged_creation([1, [2], 3])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        a = self._ragged_creation([1, 2, [3]])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

    def test_ragged_shape_object(self):
        # The ragged dimension of a list is turned into an object array
        a = self._ragged_creation([[1, 1], [2], [3]])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        a = self._ragged_creation([[1], [2, 2], [3]])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        a = self._ragged_creation([[1], [2], [3, 3]])
        assert a.shape == (3,)
        assert a.dtype == object

    def test_array_of_ragged_array(self):
        outer = np.array([None, None])
        outer[0] = outer[1] = np.array([1, 2, 3])
        assert np.array(outer).shape == (2,)
        assert np.array([outer]).shape == (1, 2)

        outer_ragged = np.array([None, None])
        outer_ragged[0] = np.array([1, 2, 3])
        outer_ragged[1] = np.array([1, 2, 3, 4])
        # should both of these emit deprecation warnings?
        assert np.array(outer_ragged).shape == (2,)
        assert np.array([outer_ragged]).shape == (1, 2,)

    def test_deep_nonragged_object(self):
        # None of these should raise, even though they are missing dtype=object
        a = np.array([[[Decimal(1)]]])
        a = np.array([1, Decimal(1)])
        a = np.array([[1], [Decimal(1)]])

    @pytest.mark.parametrize("dtype", [object, "O,O", "O,(3)O", "(2,3)O"])
    @pytest.mark.parametrize("function", [
            np.ndarray, np.empty,
            lambda shape, dtype: np.empty_like(np.empty(shape, dtype=dtype))])
    def test_object_initialized_to_None(self, function, dtype):
        # NumPy has support for object fields to be NULL (meaning None)
        # but generally, we should always fill with the proper None, and
        # downstream may rely on that.  (For fully initialized arrays!)
        arr = function(3, dtype=dtype)
        # We expect a fill value of None, which is not NULL:
        expected = np.array(None).tobytes()
        expected = expected * (arr.nbytes // len(expected))
        assert arr.tobytes() == expected


class TestBool:
    @pytest.mark.xfail(reason="bools not interned")
    def test_test_interning(self):
        a0 = np.bool_(0)
        b0 = np.bool_(False)
        assert_(a0 is b0)
        a1 = np.bool_(1)
        b1 = np.bool_(True)
        assert_(a1 is b1)
        assert_(np.array([True])[0] is a1)
        assert_(np.array(True)[()] is a1)

    def test_sum(self):
        d = np.ones(101, dtype=bool)
        assert_equal(d.sum(), d.size)
        assert_equal(d[::2].sum(), d[::2].size)
        #assert_equal(d[::-2].sum(), d[::-2].size)

    @pytest.mark.xfail(reason="frombuffer")
    def test_sum_2(self):
        d = np.frombuffer(b'\xff\xff' * 100, dtype=bool)
        assert_equal(d.sum(), d.size)
        assert_equal(d[::2].sum(), d[::2].size)
        assert_equal(d[::-2].sum(), d[::-2].size)

    def check_count_nonzero(self, power, length):
        powers = [2 ** i for i in range(length)]
        for i in range(2**power):
            l = [(i & x) != 0 for x in powers]
            a = np.array(l, dtype=bool)
            c = builtins.sum(l)
            assert_equal(np.count_nonzero(a), c)
            av = a.view(np.uint8)
            av *= 3
            assert_equal(np.count_nonzero(a), c)
            av *= 4
            assert_equal(np.count_nonzero(a), c)
            av[av != 0] = 0xFF
            assert_equal(np.count_nonzero(a), c)

    def test_count_nonzero(self):
        # check all 12 bit combinations in a length 17 array
        # covers most cases of the 16 byte unrolled code
        self.check_count_nonzero(12, 17)

    @pytest.mark.slow
    def test_count_nonzero_all(self):
        # check all combinations in a length 17 array
        # covers all cases of the 16 byte unrolled code
        self.check_count_nonzero(17, 17)

    def test_count_nonzero_unaligned(self):
        # prevent mistakes as e.g. gh-4060
        for o in range(7):
            a = np.zeros((18,), dtype=bool)[o+1:]
            a[:o] = True
            assert_equal(np.count_nonzero(a), builtins.sum(a.tolist()))
            a = np.ones((18,), dtype=bool)[o+1:]
            a[:o] = False
            assert_equal(np.count_nonzero(a), builtins.sum(a.tolist()))

    def _test_cast_from_flexible(self, dtype):
        # empty string -> false
        for n in range(3):
            v = np.array(b'', (dtype, n))
            assert_equal(bool(v), False)
            assert_equal(bool(v[()]), False)
            assert_equal(v.astype(bool), False)
            assert_(isinstance(v.astype(bool), np.ndarray))
            assert_(v[()].astype(bool) is np.False_)

        # anything else -> true
        for n in range(1, 4):
            for val in [b'a', b'0', b' ']:
                v = np.array(val, (dtype, n))
                assert_equal(bool(v), True)
                assert_equal(bool(v[()]), True)
                assert_equal(v.astype(bool), True)
                assert_(isinstance(v.astype(bool), np.ndarray))
                assert_(v[()].astype(bool) is np.True_)

    @pytest.mark.skip(reason="np.void")
    def test_cast_from_void(self):
        self._test_cast_from_flexible(np.void)

    @pytest.mark.xfail(reason="See gh-9847")
    def test_cast_from_unicode(self):
        self._test_cast_from_flexible(np.unicode_)

    @pytest.mark.xfail(reason="See gh-9847")
    def test_cast_from_bytes(self):
        self._test_cast_from_flexible(np.bytes_)



class TestMethods:

    sort_kinds = ['quicksort', 'heapsort', 'stable']

    @pytest.mark.xfail(reason="all(..., where=...)")
    def test_all_where(self):
        a = np.array([[True, False, True],
                      [False, False, False],
                      [True, True, True]])
        wh_full = np.array([[True, False, True],
                            [False, False, False],
                            [True, False, True]])
        wh_lower = np.array([[False],
                             [False],
                             [True]])
        for _ax in [0, None]:
            assert_equal(a.all(axis=_ax, where=wh_lower),
                        np.all(a[wh_lower[:,0],:], axis=_ax))
            assert_equal(np.all(a, axis=_ax, where=wh_lower),
                         a[wh_lower[:,0],:].all(axis=_ax))

        assert_equal(a.all(where=wh_full), True)
        assert_equal(np.all(a, where=wh_full), True)
        assert_equal(a.all(where=False), True)
        assert_equal(np.all(a, where=False), True)

    @pytest.mark.xfail(reason="any(..., where=...)")
    def test_any_where(self):
        a = np.array([[True, False, True],
                      [False, False, False],
                      [True, True, True]])
        wh_full = np.array([[False, True, False],
                            [True, True, True],
                            [False, False, False]])
        wh_middle = np.array([[False],
                              [True],
                              [False]])
        for _ax in [0, None]:
            assert_equal(a.any(axis=_ax, where=wh_middle),
                         np.any(a[wh_middle[:,0],:], axis=_ax))
            assert_equal(np.any(a, axis=_ax, where=wh_middle),
                         a[wh_middle[:,0],:].any(axis=_ax))
        assert_equal(a.any(where=wh_full), False)
        assert_equal(np.any(a, where=wh_full), False)
        assert_equal(a.any(where=False), False)
        assert_equal(np.any(a, where=False), False)

    @pytest.mark.xfail(reason="TODO: compress")
    def test_compress(self):
        tgt = [[5, 6, 7, 8, 9]]
        arr = np.arange(10).reshape(2, 5)
        out = arr.compress([0, 1], axis=0)
        assert_equal(out, tgt)

        tgt = [[1, 3], [6, 8]]
        out = arr.compress([0, 1, 0, 1, 0], axis=1)
        assert_equal(out, tgt)

        tgt = [[1], [6]]
        arr = np.arange(10).reshape(2, 5)
        out = arr.compress([0, 1], axis=1)
        assert_equal(out, tgt)

        arr = np.arange(10).reshape(2, 5)
        out = arr.compress([0, 1])
        assert_equal(out, 1)

    def test_choose(self):
        x = 2*np.ones((3,), dtype=int)
        y = 3*np.ones((3,), dtype=int)
        x2 = 2*np.ones((2, 3), dtype=int)
        y2 = 3*np.ones((2, 3), dtype=int)
        ind = np.array([0, 0, 1])

        A = ind.choose((x, y))
        assert_equal(A, [2, 2, 3])

        A = ind.choose((x2, y2))
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])

        A = ind.choose((x, y2))
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])

        out = np.array(0)
        ret = np.choose(np.array(1), [10, 20, 30], out=out)
        assert out is ret
        assert_equal(out[()], 20)

    @pytest.mark.xfail(reason="choose(..., mode=...) not implemented")
    def test_choose_2(self):
        # gh-6272 check overlap on out
        x = np.arange(5)
        y = np.choose([0,0,0], [x[:3], x[:3], x[:3]], out=x[1:4], mode='wrap')
        assert_equal(y, np.array([0, 1, 2]))

    def test_prod(self):
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]

        for ctype in [np.int16, np.int32,
                      np.float32, np.float64, np.complex64, np.complex128]:
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)
            if ctype in ['1', 'b']:
                assert_raises(ArithmeticError, a.prod)
                assert_raises(ArithmeticError, a2.prod, axis=1)
            else:
                assert_equal(a.prod(axis=0), 26400)
                assert_array_equal(a2.prod(axis=0),
                                   np.array([50, 36, 84, 180], ctype))
                assert_array_equal(a2.prod(axis=-1),
                                   np.array([24, 1890, 600], ctype))

    def test_repeat(self):
        m = np.array([1, 2, 3, 4, 5, 6])
        m_rect = m.reshape((2, 3))

        A = m.repeat([1, 3, 2, 1, 1, 2])
        assert_equal(A, [1, 2, 2, 2, 3,
                         3, 4, 5, 6, 6])

        A = m.repeat(2)
        assert_equal(A, [1, 1, 2, 2, 3, 3,
                         4, 4, 5, 5, 6, 6])

        A = m_rect.repeat([2, 1], axis=0)
        assert_equal(A, [[1, 2, 3],
                         [1, 2, 3],
                         [4, 5, 6]])

        A = m_rect.repeat([1, 3, 2], axis=1)
        assert_equal(A, [[1, 2, 2, 2, 3, 3],
                         [4, 5, 5, 5, 6, 6]])

        A = m_rect.repeat(2, axis=0)
        assert_equal(A, [[1, 2, 3],
                         [1, 2, 3],
                         [4, 5, 6],
                         [4, 5, 6]])

        A = m_rect.repeat(2, axis=1)
        assert_equal(A, [[1, 1, 2, 2, 3, 3],
                         [4, 4, 5, 5, 6, 6]])

    @pytest.mark.xfail(reason="reshape(..., order='F')")
    def test_reshape(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        assert_equal(arr.reshape(2, 6), tgt)

        tgt = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        assert_equal(arr.reshape(3, 4), tgt)

        tgt = [[1, 10, 8, 6], [4, 2, 11, 9], [7, 5, 3, 12]]
        assert_equal(arr.reshape((3, 4), order='F'), tgt)

        tgt = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
        assert_equal(arr.T.reshape((3, 4), order='C'), tgt)

    def test_round(self):
        def check_round(arr, expected, *round_args):
            assert_equal(arr.round(*round_args), expected)
            # With output array
            out = np.zeros_like(arr)
            res = arr.round(*round_args, out=out)
            assert_equal(out, expected)
            assert out is res

        check_round(np.array([1.2, 1.5]), [1, 2])
        check_round(np.array(1.5), 2)
        check_round(np.array([12.2, 15.5]), [10, 20], -1)
        check_round(np.array([12.15, 15.51]), [12.2, 15.5], 1)
        # Complex rounding
        check_round(np.array([4.5 + 1.5j]), [4 + 2j])
        check_round(np.array([12.5 + 15.5j]), [10 + 20j], -1)

    def test_squeeze(self):
        a = np.array([[[1], [2], [3]]])
        assert_equal(a.squeeze(), [1, 2, 3])
        assert_equal(a.squeeze(axis=(0,)), [[1], [2], [3]])
        #  assert_raises(ValueError, a.squeeze, axis=(1,))   # a noop in pytorch
        assert_equal(a.squeeze(axis=(2,)), [[1, 2, 3]])

    def test_transpose(self):
        a = np.array([[1, 2], [3, 4]])
        assert_equal(a.transpose(), [[1, 3], [2, 4]])
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0))
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0, 0))
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0, 1, 2))

    def test_sort(self):
        # test ordering for floats and complex containing nans. It is only
        # necessary to check the less-than comparison, so sorts that
        # only follow the insertion sort path are sufficient. We only
        # test doubles and complex doubles as the logic is the same.

        # check doubles
        msg = "Test real sort order with nans"
        a = np.array([np.nan, 1, 0])
        b = np.sort(a)
        assert_equal(b, np.flip(a), msg)

    @pytest.mark.xfail(reason='sort complex')
    def test_sort_complex(self):
        # check complex
        msg = "Test complex sort order with nans"
        a = np.zeros(9, dtype=np.complex128)
        a.real += [np.nan, np.nan, np.nan, 1, 0, 1, 1, 0, 0]
        a.imag += [np.nan, 1, 0, np.nan, np.nan, 1, 0, 1, 0]
        b = np.sort(a)
        assert_equal(b, a[::-1], msg)

    # all c scalar sorts use the same code with different types
    # so it suffices to run a quick check with one type. The number
    # of sorted items must be greater than ~50 to check the actual
    # algorithm because quick and merge sort fall over to insertion
    # sort for small arrays.

    @pytest.mark.parametrize('dtype', [np.uint8,
                                       np.float16, np.float32, np.float64])
    def test_sort_unsigned(self, dtype):
        a = np.arange(101, dtype=dtype)
        b = np.flip(a)
        for kind in self.sort_kinds:
            msg = "scalar sort, kind=%s" % kind
            c = a.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)
            c = b.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)

    @pytest.mark.parametrize('dtype',
                             [np.int8, np.int16, np.int32, np.int64, np.float16,
                              np.float32, np.float64])
    def test_sort_signed(self, dtype):
        a = np.arange(-50, 51, dtype=dtype)
        b = np.flip(a)
        for kind in self.sort_kinds:
            msg = "scalar sort, kind=%s" % (kind)
            c = a.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)
            c = b.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)

    @pytest.mark.xfail(reason='sort complex')
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    @pytest.mark.parametrize('part', ['real', 'imag'])
    def test_sort_complex(self, part, dtype):
        # test complex sorts. These use the same code as the scalars
        # but the compare function differs.
        cdtype = {
            np.single: np.csingle,
            np.double: np.cdouble,
        }[dtype]
        a = np.arange(-50, 51, dtype=dtype)
        b = a[::-1].copy()
        ai = (a * (1+1j)).astype(cdtype)
        bi = (b * (1+1j)).astype(cdtype)
        setattr(ai, part, 1)
        setattr(bi, part, 1)
        for kind in self.sort_kinds:
            msg = "complex sort, %s part == 1, kind=%s" % (part, kind)
            c = ai.copy()
            c.sort(kind=kind)
            assert_equal(c, ai, msg)
            c = bi.copy()
            c.sort(kind=kind)
            assert_equal(c, ai, msg)

    def test_sort_axis(self):
        # check axis handling. This should be the same for all type
        # specific sorts, so we only check it for one type and one kind
        a = np.array([[3, 2], [1, 0]])
        b = np.array([[1, 0], [3, 2]])
        c = np.array([[2, 3], [0, 1]])
        d = a.copy()
        d.sort(axis=0)
        assert_equal(d, b, "test sort with axis=0")
        d = a.copy()
        d.sort(axis=1)
        assert_equal(d, c, "test sort with axis=1")
        d = a.copy()
        d.sort()
        assert_equal(d, c, "test sort with default axis")

    def test_sort_size_0(self):
        # check axis handling for multidimensional empty arrays
        a = np.array([])
        a = a.reshape(3, 2, 1, 0)
        for axis in range(-a.ndim, a.ndim):
            msg = 'test empty array sort with axis={0}'.format(axis)
            assert_equal(np.sort(a, axis=axis), a, msg)
        msg = 'test empty array sort with axis=None'
        assert_equal(np.sort(a, axis=None), a.ravel(), msg)

    @pytest.mark.skip(reason='waaay tooo sloooow')
    def test_sort_degraded(self):
        # test degraded dataset would take minutes to run with normal qsort
        d = np.arange(1000000)
        do = d.copy()
        x = d
        # create a median of 3 killer where each median is the sorted second
        # last element of the quicksort partition
        while x.size > 3:
            mid = x.size // 2
            x[mid], x[-2] = x[-2], x[mid]
            x = x[:-2]

        assert_equal(np.sort(d), do)
        assert_equal(d[np.argsort(d)], do)

    @pytest.mark.xfail(reason="order='F'")
    def test_copy(self):
        def assert_fortran(arr):
            assert_(arr.flags.fortran)
            assert_(arr.flags.f_contiguous)
            assert_(not arr.flags.c_contiguous)

        def assert_c(arr):
            assert_(not arr.flags.fortran)
            assert_(not arr.flags.f_contiguous)
            assert_(arr.flags.c_contiguous)

        a = np.empty((2, 2), order='F')
        # Test copying a Fortran array
        assert_c(a.copy())
        assert_c(a.copy('C'))
        assert_fortran(a.copy('F'))
        assert_fortran(a.copy('A'))

        # Now test starting with a C array.
        a = np.empty((2, 2), order='C')
        assert_c(a.copy())
        assert_c(a.copy('C'))
        assert_fortran(a.copy('F'))
        assert_c(a.copy('A'))

    @pytest.mark.skip(reason="no .ctypes attribute")
    @pytest.mark.parametrize("dtype", [np.int32])
    def test__deepcopy__(self, dtype):
        # Force the entry of NULLs into array
        a = np.empty(4, dtype=dtype)
        ctypes.memset(a.ctypes.data, 0, a.nbytes)

        # Ensure no error is raised, see gh-21833
        b = a.__deepcopy__({})

        a[0] = 42
        with pytest.raises(AssertionError):
            assert_array_equal(a, b)

    def test_argsort(self):
        # all c scalar argsorts use the same code with different types
        # so it suffices to run a quick check with one type. The number
        # of sorted items must be greater than ~50 to check the actual
        # algorithm because quick and merge sort fall over to insertion
        # sort for small arrays.

        for dtype in [np.int32, np.uint8, np.float32]:
            a = np.arange(101, dtype=dtype)
            b = np.flip(a)
            for kind in self.sort_kinds:
                msg = "scalar argsort, kind=%s, dtype=%s" % (kind, dtype)
                assert_equal(a.copy().argsort(kind=kind), a, msg)
                assert_equal(b.copy().argsort(kind=kind), b, msg)

    @pytest.mark.skip(reason='argsort complex')
    def test_argsort_complex(self):
        a = np.arange(101, dtype=np.float32)
        b = np.flip(a)

        # test complex argsorts. These use the same code as the scalars
        # but the compare function differs.
        ai = a*1j + 1
        bi = b*1j + 1
        for kind in self.sort_kinds:
            msg = "complex argsort, kind=%s" % kind
            assert_equal(ai.copy().argsort(kind=kind), a, msg)
            assert_equal(bi.copy().argsort(kind=kind), b, msg)
        ai = a + 1j
        bi = b + 1j
        for kind in self.sort_kinds:
            msg = "complex argsort, kind=%s" % kind
            assert_equal(ai.copy().argsort(kind=kind), a, msg)
            assert_equal(bi.copy().argsort(kind=kind), b, msg)

        # test argsort of complex arrays requiring byte-swapping, gh-5441
        for endianness in '<>':
            for dt in np.typecodes['Complex']:
                arr = np.array([1+3.j, 2+2.j, 3+1.j], dtype=endianness + dt)
                msg = 'byte-swapped complex argsort, dtype={0}'.format(dt)
                assert_equal(arr.argsort(),
                             np.arange(len(arr), dtype=np.intp), msg)

    @pytest.mark.xfail(reason='argsort axis TODO')
    def test_argsort_axis(self):
        # check axis handling. This should be the same for all type
        # specific argsorts, so we only check it for one type and one kind
        a = np.array([[3, 2], [1, 0]])
        b = np.array([[1, 1], [0, 0]])
        c = np.array([[1, 0], [1, 0]])
        assert_equal(a.copy().argsort(axis=0), b)
        assert_equal(a.copy().argsort(axis=1), c)
        assert_equal(a.copy().argsort(), c)

        # check axis handling for multidimensional empty arrays
        a = np.array([])
        a = a.reshape(3, 2, 1, 0)
        for axis in range(-a.ndim, a.ndim):
            msg = 'test empty array argsort with axis={0}'.format(axis)
            assert_equal(np.argsort(a, axis=axis),
                         np.zeros_like(a, dtype=np.intp), msg)
        msg = 'test empty array argsort with axis=None'
        assert_equal(np.argsort(a, axis=None),
                     np.zeros_like(a.ravel(), dtype=np.intp), msg)

        # check that stable argsorts are stable
        r = np.arange(100)
        # scalars
        a = np.zeros(100)
        assert_equal(a.argsort(kind='m'), r)
        # complex
        a = np.zeros(100, dtype=complex)
        assert_equal(a.argsort(kind='m'), r)
        # string
        a = np.array(['aaaaaaaaa' for i in range(100)])
        assert_equal(a.argsort(kind='m'), r)
        # unicode
        a = np.array(['aaaaaaaaa' for i in range(100)], dtype=np.unicode_)
        assert_equal(a.argsort(kind='m'), r)

    @pytest.mark.xfail(reason='TODO: searchsorted with nans differs in pytorch')
    @pytest.mark.parametrize('a', [
        np.array([0, 1, np.nan], dtype=np.float16),
        np.array([0, 1, np.nan], dtype=np.float32),
        np.array([0, 1, np.nan]),
    ])
    def test_searchsorted_floats(self, a):
        # test for floats arrays containing nans. Explicitly test
        # half, single, and double precision floats to verify that
        # the NaN-handling is correct.
        msg = "Test real (%s) searchsorted with nans, side='l'" % a.dtype
        b = a.searchsorted(a, side='left')
        assert_equal(b, np.arange(3), msg)
        msg = "Test real (%s) searchsorted with nans, side='r'" % a.dtype
        b = a.searchsorted(a, side='right')
        assert_equal(b, np.arange(1, 4), msg)
        # check keyword arguments
        a.searchsorted(v=1)
        x = np.array([0, 1, np.nan], dtype='float32')
        y = np.searchsorted(x, x[-1])
        assert_equal(y, 2)

    @pytest.mark.xfail(reason="'searchsorted_out_cpu' not implemented for 'ComplexDouble'")
    def test_searchsorted_complex(self):
        # test for complex arrays containing nans.
        # The search sorted routines use the compare functions for the
        # array type, so this checks if that is consistent with the sort
        # order.
        # check double complex
        a = np.zeros(9, dtype=np.complex128)
        a.real += [0, 0, 1, 1, 0, 1, np.nan, np.nan, np.nan]
        a.imag += [0, 1, 0, 1, np.nan, np.nan, 0, 1, np.nan]
        msg = "Test complex searchsorted with nans, side='l'"
        b = a.searchsorted(a, side='left')
        assert_equal(b, np.arange(9), msg)
        msg = "Test complex searchsorted with nans, side='r'"
        b = a.searchsorted(a, side='right')
        assert_equal(b, np.arange(1, 10), msg)
        msg = "Test searchsorted with little endian, side='l'"
        a = np.array([0, 128], dtype='<i4')
        b = a.searchsorted(np.array(128, dtype='<i4'))
        assert_equal(b, 1, msg)
        msg = "Test searchsorted with big endian, side='l'"
        a = np.array([0, 128], dtype='>i4')
        b = a.searchsorted(np.array(128, dtype='>i4'))
        assert_equal(b, 1, msg)

    def test_searchsorted_n_elements(self):
        # Check 0 elements
        a = np.ones(0)
        b = a.searchsorted([0, 1, 2], 'left')
        assert_equal(b, [0, 0, 0])
        b = a.searchsorted([0, 1, 2], 'right')
        assert_equal(b, [0, 0, 0])
        a = np.ones(1)
        # Check 1 element
        b = a.searchsorted([0, 1, 2], 'left')
        assert_equal(b, [0, 0, 1])
        b = a.searchsorted([0, 1, 2], 'right')
        assert_equal(b, [0, 1, 1])
        # Check all elements equal
        a = np.ones(2)
        b = a.searchsorted([0, 1, 2], 'left')
        assert_equal(b, [0, 0, 2])
        b = a.searchsorted([0, 1, 2], 'right')
        assert_equal(b, [0, 2, 2])

    @pytest.mark.xfail(reason="RuntimeError: self.storage_offset() must be divisible by 8")
    def test_searchsorted_unaligned_array(self):
        # Test searching unaligned array
        a = np.arange(10)
        aligned = np.empty(a.itemsize * a.size + 1, dtype='uint8')
        unaligned = aligned[1:].view(a.dtype)
        unaligned[:] = a
        # Test searching unaligned array
        b = unaligned.searchsorted(a, 'left')
        assert_equal(b, a)
        b = unaligned.searchsorted(a, 'right')
        assert_equal(b, a + 1)
        # Test searching for unaligned keys
        b = a.searchsorted(unaligned, 'left')
        assert_equal(b, a)
        b = a.searchsorted(unaligned, 'right')
        assert_equal(b, a + 1)

    def test_searchsorted_resetting(self):
        # Test smart resetting of binsearch indices
        a = np.arange(5)
        b = a.searchsorted([6, 5, 4], 'left')
        assert_equal(b, [5, 5, 4])
        b = a.searchsorted([6, 5, 4], 'right')
        assert_equal(b, [5, 5, 5])

    def test_searchsorted_type_specific(self):
        # Test all type specific binary search functions
        types = ''.join((np.typecodes['AllInteger'], np.typecodes['Float']))
        for dt in types:
            if dt == '?':
                a = np.arange(2, dtype=dt)
                out = np.arange(2)
            else:
                a = np.arange(0, 5, dtype=dt)
                out = np.arange(5)
            b = a.searchsorted(a, 'left')
            assert_equal(b, out)
            b = a.searchsorted(a, 'right')
            assert_equal(b, out + 1)

    @pytest.mark.xfail(reason="ndarray ctor")
    def test_searchsorted_type_specific_2(self):
        # Test all type specific binary search functions
        types = ''.join((np.typecodes['AllInteger'], np.typecodes['AllFloat'],
                         '?'))
        for dt in types:
            if dt == '?':
                a = np.arange(2, dtype=dt)
                out = np.arange(2)
            else:
                a = np.arange(0, 5, dtype=dt)
                out = np.arange(5)

            # Test empty array, use a fresh array to get warnings in
            # valgrind if access happens.
            e = np.ndarray(shape=0, buffer=b'', dtype=dt)
            b = e.searchsorted(a, 'left')
            assert_array_equal(b, np.zeros(len(a), dtype=np.intp))
            b = a.searchsorted(e, 'left')
            assert_array_equal(b, np.zeros(0, dtype=np.intp))

    def test_searchsorted_with_invalid_sorter(self):
        a = np.array([5, 2, 1, 3, 4])
        s = np.argsort(a)
        assert_raises((TypeError, RuntimeError), np.searchsorted, a, 0, sorter=[1.1])
        assert_raises((ValueError, RuntimeError), np.searchsorted, a, 0, sorter=[1, 2, 3, 4])
        assert_raises((ValueError, RuntimeError), np.searchsorted, a, 0, sorter=[1, 2, 3, 4, 5, 6])

        # bounds check : XXX torch does not raise
        # assert_raises(ValueError, np.searchsorted, a, 4, sorter=[0, 1, 2, 3, 5])
        # assert_raises(ValueError, np.searchsorted, a, 0, sorter=[-1, 0, 1, 2, 3])
        # assert_raises(ValueError, np.searchsorted, a, 0, sorter=[4, 0, -1, 2, 3])

    @pytest.mark.xfail(reason='self.storage_offset() must be divisible by 8')
    def test_searchsorted_with_sorter(self):
        a = np.random.rand(300)
        s = a.argsort()
        b = np.sort(a)
        k = np.linspace(0, 1, 20)
        assert_equal(b.searchsorted(k), a.searchsorted(k, sorter=s))

        a = np.array([0, 1, 2, 3, 5]*20)
        s = a.argsort()
        k = [0, 1, 2, 3, 5]
        expected = [0, 20, 40, 60, 80]
        assert_equal(a.searchsorted(k, side='left', sorter=s), expected)
        expected = [20, 40, 60, 80, 100]
        assert_equal(a.searchsorted(k, side='right', sorter=s), expected)

        # Test searching unaligned array
        keys = np.arange(10)
        a = keys.copy()
        np.random.shuffle(s)
        s = a.argsort()
        aligned = np.empty(a.itemsize * a.size + 1, dtype='uint8')
        unaligned = aligned[1:].view(a.dtype)
        # Test searching unaligned array
        unaligned[:] = a
        b = unaligned.searchsorted(keys, 'left', s)
        assert_equal(b, keys)
        b = unaligned.searchsorted(keys, 'right', s)
        assert_equal(b, keys + 1)
        # Test searching for unaligned keys
        unaligned[:] = keys
        b = a.searchsorted(unaligned, 'left', s)
        assert_equal(b, keys)
        b = a.searchsorted(unaligned, 'right', s)
        assert_equal(b, keys + 1)

        # Test all type specific indirect binary search functions
        types = ''.join((np.typecodes['AllInteger'], np.typecodes['AllFloat'],
                         '?'))
        for dt in types:
            if dt == '?':
                a = np.array([1, 0], dtype=dt)
                # We want the sorter array to be of a type that is different
                # from np.intp in all platforms, to check for #4698
                s = np.array([1, 0], dtype=np.int16)
                out = np.array([1, 0])
            else:
                a = np.array([3, 4, 1, 2, 0], dtype=dt)
                # We want the sorter array to be of a type that is different
                # from np.intp in all platforms, to check for #4698
                s = np.array([4, 2, 3, 0, 1], dtype=np.int16)
                out = np.array([3, 4, 1, 2, 0], dtype=np.intp)
            b = a.searchsorted(a, 'left', s)
            assert_equal(b, out)
            b = a.searchsorted(a, 'right', s)
            assert_equal(b, out + 1)
            # Test empty array, use a fresh array to get warnings in
            # valgrind if access happens.
            e = np.ndarray(shape=0, buffer=b'', dtype=dt)
            b = e.searchsorted(a, 'left', s[:0])
            assert_array_equal(b, np.zeros(len(a), dtype=np.intp))
            b = a.searchsorted(e, 'left', s)
            assert_array_equal(b, np.zeros(0, dtype=np.intp))

        # Test non-contiguous sorter array
        a = np.array([3, 4, 1, 2, 0])
        srt = np.empty((10,), dtype=np.intp)
        srt[1::2] = -1
        srt[::2] = [4, 2, 3, 0, 1]
        s = srt[::2]
        out = np.array([3, 4, 1, 2, 0], dtype=np.intp)
        b = a.searchsorted(a, 'left', s)
        assert_equal(b, out)
        b = a.searchsorted(a, 'right', s)
        assert_equal(b, out + 1)

    @pytest.mark.xfail(reason="TODO argpartition")
    @pytest.mark.parametrize("dtype", np.typecodes["All"])
    def test_argpartition_out_of_range(self, dtype):
        # Test out of range values in kth raise an error, gh-5469
        d = np.arange(10).astype(dtype=dtype)
        assert_raises(ValueError, d.argpartition, 10)
        assert_raises(ValueError, d.argpartition, -11)

    @pytest.mark.xfail(reason="TODO partition")
    @pytest.mark.parametrize("dtype", np.typecodes["All"])
    def test_partition_out_of_range(self, dtype):
        # Test out of range values in kth raise an error, gh-5469
        d = np.arange(10).astype(dtype=dtype)
        assert_raises(ValueError, d.partition, 10)
        assert_raises(ValueError, d.partition, -11)

    @pytest.mark.xfail(reason="TODO argpartition")
    def test_argpartition_integer(self):
        # Test non-integer values in kth raise an error/
        d = np.arange(10)
        assert_raises(TypeError, d.argpartition, 9.)
        # Test also for generic type argpartition, which uses sorting
        # and used to not bound check kth
        d_obj = np.arange(10, dtype=object)
        assert_raises(TypeError, d_obj.argpartition, 9.)

    @pytest.mark.xfail(reason="TODO partition")
    def test_partition_integer(self):
        # Test out of range values in kth raise an error, gh-5469
        d = np.arange(10)
        assert_raises(TypeError, d.partition, 9.)
        # Test also for generic type partition, which uses sorting
        # and used to not bound check kth
        d_obj = np.arange(10, dtype=object)
        assert_raises(TypeError, d_obj.partition, 9.)

    @pytest.mark.xfail(reason="TODO partition")
    @pytest.mark.parametrize("kth_dtype", np.typecodes["AllInteger"])
    def test_partition_empty_array(self, kth_dtype):
        # check axis handling for multidimensional empty arrays
        kth = np.array(0, dtype=kth_dtype)[()]
        a = np.array([])
        a.shape = (3, 2, 1, 0)
        for axis in range(-a.ndim, a.ndim):
            msg = 'test empty array partition with axis={0}'.format(axis)
            assert_equal(np.partition(a, kth, axis=axis), a, msg)
        msg = 'test empty array partition with axis=None'
        assert_equal(np.partition(a, kth, axis=None), a.ravel(), msg)

    @pytest.mark.xfail(reason="TODO argpartition")
    @pytest.mark.parametrize("kth_dtype", np.typecodes["AllInteger"])
    def test_argpartition_empty_array(self, kth_dtype):
        # check axis handling for multidimensional empty arrays
        kth = np.array(0, dtype=kth_dtype)[()]
        a = np.array([])
        a.shape = (3, 2, 1, 0)
        for axis in range(-a.ndim, a.ndim):
            msg = 'test empty array argpartition with axis={0}'.format(axis)
            assert_equal(np.partition(a, kth, axis=axis),
                         np.zeros_like(a, dtype=np.intp), msg)
        msg = 'test empty array argpartition with axis=None'
        assert_equal(np.partition(a, kth, axis=None),
                     np.zeros_like(a.ravel(), dtype=np.intp), msg)

    @pytest.mark.xfail(reason="TODO partition")
    def test_partition(self):
        d = np.arange(10)
        assert_raises(TypeError, np.partition, d, 2, kind=1)
        assert_raises(ValueError, np.partition, d, 2, kind="nonsense")
        assert_raises(ValueError, np.argpartition, d, 2, kind="nonsense")
        assert_raises(ValueError, d.partition, 2, axis=0, kind="nonsense")
        assert_raises(ValueError, d.argpartition, 2, axis=0, kind="nonsense")
        for k in ("introselect",):
            d = np.array([])
            assert_array_equal(np.partition(d, 0, kind=k), d)
            assert_array_equal(np.argpartition(d, 0, kind=k), d)
            d = np.ones(1)
            assert_array_equal(np.partition(d, 0, kind=k)[0], d)
            assert_array_equal(d[np.argpartition(d, 0, kind=k)],
                               np.partition(d, 0, kind=k))

            # kth not modified
            kth = np.array([30, 15, 5])
            okth = kth.copy()
            np.partition(np.arange(40), kth)
            assert_array_equal(kth, okth)

            for r in ([2, 1], [1, 2], [1, 1]):
                d = np.array(r)
                tgt = np.sort(d)
                assert_array_equal(np.partition(d, 0, kind=k)[0], tgt[0])
                assert_array_equal(np.partition(d, 1, kind=k)[1], tgt[1])
                assert_array_equal(d[np.argpartition(d, 0, kind=k)],
                                   np.partition(d, 0, kind=k))
                assert_array_equal(d[np.argpartition(d, 1, kind=k)],
                                   np.partition(d, 1, kind=k))
                for i in range(d.size):
                    d[i:].partition(0, kind=k)
                assert_array_equal(d, tgt)

            for r in ([3, 2, 1], [1, 2, 3], [2, 1, 3], [2, 3, 1],
                      [1, 1, 1], [1, 2, 2], [2, 2, 1], [1, 2, 1]):
                d = np.array(r)
                tgt = np.sort(d)
                assert_array_equal(np.partition(d, 0, kind=k)[0], tgt[0])
                assert_array_equal(np.partition(d, 1, kind=k)[1], tgt[1])
                assert_array_equal(np.partition(d, 2, kind=k)[2], tgt[2])
                assert_array_equal(d[np.argpartition(d, 0, kind=k)],
                                   np.partition(d, 0, kind=k))
                assert_array_equal(d[np.argpartition(d, 1, kind=k)],
                                   np.partition(d, 1, kind=k))
                assert_array_equal(d[np.argpartition(d, 2, kind=k)],
                                   np.partition(d, 2, kind=k))
                for i in range(d.size):
                    d[i:].partition(0, kind=k)
                assert_array_equal(d, tgt)

            d = np.ones(50)
            assert_array_equal(np.partition(d, 0, kind=k), d)
            assert_array_equal(d[np.argpartition(d, 0, kind=k)],
                               np.partition(d, 0, kind=k))

            # sorted
            d = np.arange(49)
            assert_equal(np.partition(d, 5, kind=k)[5], 5)
            assert_equal(np.partition(d, 15, kind=k)[15], 15)
            assert_array_equal(d[np.argpartition(d, 5, kind=k)],
                               np.partition(d, 5, kind=k))
            assert_array_equal(d[np.argpartition(d, 15, kind=k)],
                               np.partition(d, 15, kind=k))

            # rsorted
            d = np.arange(47)[::-1]
            assert_equal(np.partition(d, 6, kind=k)[6], 6)
            assert_equal(np.partition(d, 16, kind=k)[16], 16)
            assert_array_equal(d[np.argpartition(d, 6, kind=k)],
                               np.partition(d, 6, kind=k))
            assert_array_equal(d[np.argpartition(d, 16, kind=k)],
                               np.partition(d, 16, kind=k))

            assert_array_equal(np.partition(d, -6, kind=k),
                               np.partition(d, 41, kind=k))
            assert_array_equal(np.partition(d, -16, kind=k),
                               np.partition(d, 31, kind=k))
            assert_array_equal(d[np.argpartition(d, -6, kind=k)],
                               np.partition(d, 41, kind=k))

            # median of 3 killer, O(n^2) on pure median 3 pivot quickselect
            # exercises the median of median of 5 code used to keep O(n)
            d = np.arange(1000000)
            x = np.roll(d, d.size // 2)
            mid = x.size // 2 + 1
            assert_equal(np.partition(x, mid)[mid], mid)
            d = np.arange(1000001)
            x = np.roll(d, d.size // 2 + 1)
            mid = x.size // 2 + 1
            assert_equal(np.partition(x, mid)[mid], mid)

            # max
            d = np.ones(10)
            d[1] = 4
            assert_equal(np.partition(d, (2, -1))[-1], 4)
            assert_equal(np.partition(d, (2, -1))[2], 1)
            assert_equal(d[np.argpartition(d, (2, -1))][-1], 4)
            assert_equal(d[np.argpartition(d, (2, -1))][2], 1)
            d[1] = np.nan
            assert_(np.isnan(d[np.argpartition(d, (2, -1))][-1]))
            assert_(np.isnan(np.partition(d, (2, -1))[-1]))

            # equal elements
            d = np.arange(47) % 7
            tgt = np.sort(np.arange(47) % 7)
            np.random.shuffle(d)
            for i in range(d.size):
                assert_equal(np.partition(d, i, kind=k)[i], tgt[i])
            assert_array_equal(d[np.argpartition(d, 6, kind=k)],
                               np.partition(d, 6, kind=k))
            assert_array_equal(d[np.argpartition(d, 16, kind=k)],
                               np.partition(d, 16, kind=k))
            for i in range(d.size):
                d[i:].partition(0, kind=k)
            assert_array_equal(d, tgt)

            d = np.array([0, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                          7, 7, 7, 7, 7, 9])
            kth = [0, 3, 19, 20]
            assert_equal(np.partition(d, kth, kind=k)[kth], (0, 3, 7, 7))
            assert_equal(d[np.argpartition(d, kth, kind=k)][kth], (0, 3, 7, 7))

            d = np.array([2, 1])
            d.partition(0, kind=k)
            assert_raises(ValueError, d.partition, 2)
            assert_raises(np.AxisError, d.partition, 3, axis=1)
            assert_raises(ValueError, np.partition, d, 2)
            assert_raises(np.AxisError, np.partition, d, 2, axis=1)
            assert_raises(ValueError, d.argpartition, 2)
            assert_raises(np.AxisError, d.argpartition, 3, axis=1)
            assert_raises(ValueError, np.argpartition, d, 2)
            assert_raises(np.AxisError, np.argpartition, d, 2, axis=1)
            d = np.arange(10).reshape((2, 5))
            d.partition(1, axis=0, kind=k)
            d.partition(4, axis=1, kind=k)
            np.partition(d, 1, axis=0, kind=k)
            np.partition(d, 4, axis=1, kind=k)
            np.partition(d, 1, axis=None, kind=k)
            np.partition(d, 9, axis=None, kind=k)
            d.argpartition(1, axis=0, kind=k)
            d.argpartition(4, axis=1, kind=k)
            np.argpartition(d, 1, axis=0, kind=k)
            np.argpartition(d, 4, axis=1, kind=k)
            np.argpartition(d, 1, axis=None, kind=k)
            np.argpartition(d, 9, axis=None, kind=k)
            assert_raises(ValueError, d.partition, 2, axis=0)
            assert_raises(ValueError, d.partition, 11, axis=1)
            assert_raises(TypeError, d.partition, 2, axis=None)
            assert_raises(ValueError, np.partition, d, 9, axis=1)
            assert_raises(ValueError, np.partition, d, 11, axis=None)
            assert_raises(ValueError, d.argpartition, 2, axis=0)
            assert_raises(ValueError, d.argpartition, 11, axis=1)
            assert_raises(ValueError, np.argpartition, d, 9, axis=1)
            assert_raises(ValueError, np.argpartition, d, 11, axis=None)

            td = [(dt, s) for dt in [np.int32, np.float32, np.complex64]
                  for s in (9, 16)]
            for dt, s in td:
                aae = assert_array_equal
                at = assert_

                d = np.arange(s, dtype=dt)
                np.random.shuffle(d)
                d1 = np.tile(np.arange(s, dtype=dt), (4, 1))
                map(np.random.shuffle, d1)
                d0 = np.transpose(d1)
                for i in range(d.size):
                    p = np.partition(d, i, kind=k)
                    assert_equal(p[i], i)
                    # all before are smaller
                    assert_array_less(p[:i], p[i])
                    # all after are larger
                    assert_array_less(p[i], p[i + 1:])
                    aae(p, d[np.argpartition(d, i, kind=k)])

                    p = np.partition(d1, i, axis=1, kind=k)
                    aae(p[:, i], np.array([i] * d1.shape[0], dtype=dt))
                    # array_less does not seem to work right
                    at((p[:, :i].T <= p[:, i]).all(),
                       msg="%d: %r <= %r" % (i, p[:, i], p[:, :i].T))
                    at((p[:, i + 1:].T > p[:, i]).all(),
                       msg="%d: %r < %r" % (i, p[:, i], p[:, i + 1:].T))
                    aae(p, d1[np.arange(d1.shape[0])[:, None],
                        np.argpartition(d1, i, axis=1, kind=k)])

                    p = np.partition(d0, i, axis=0, kind=k)
                    aae(p[i, :], np.array([i] * d1.shape[0], dtype=dt))
                    # array_less does not seem to work right
                    at((p[:i, :] <= p[i, :]).all(),
                       msg="%d: %r <= %r" % (i, p[i, :], p[:i, :]))
                    at((p[i + 1:, :] > p[i, :]).all(),
                       msg="%d: %r < %r" % (i, p[i, :], p[:, i + 1:]))
                    aae(p, d0[np.argpartition(d0, i, axis=0, kind=k),
                        np.arange(d0.shape[1])[None, :]])

                    # check inplace
                    dc = d.copy()
                    dc.partition(i, kind=k)
                    assert_equal(dc, np.partition(d, i, kind=k))
                    dc = d0.copy()
                    dc.partition(i, axis=0, kind=k)
                    assert_equal(dc, np.partition(d0, i, axis=0, kind=k))
                    dc = d1.copy()
                    dc.partition(i, axis=1, kind=k)
                    assert_equal(dc, np.partition(d1, i, axis=1, kind=k))

    def assert_partitioned(self, d, kth):
        prev = 0
        for k in np.sort(kth):
            assert_array_less(d[prev:k], d[k], err_msg='kth %d' % k)
            assert_((d[k:] >= d[k]).all(),
                    msg="kth %d, %r not greater equal %d" % (k, d[k:], d[k]))
            prev = k + 1

    @pytest.mark.xfail(reason="TODO partition")
    def test_partition_iterative(self):
            d = np.arange(17)
            kth = (0, 1, 2, 429, 231)
            assert_raises(ValueError, d.partition, kth)
            assert_raises(ValueError, d.argpartition, kth)
            d = np.arange(10).reshape((2, 5))
            assert_raises(ValueError, d.partition, kth, axis=0)
            assert_raises(ValueError, d.partition, kth, axis=1)
            assert_raises(ValueError, np.partition, d, kth, axis=1)
            assert_raises(ValueError, np.partition, d, kth, axis=None)

            d = np.array([3, 4, 2, 1])
            p = np.partition(d, (0, 3))
            self.assert_partitioned(p, (0, 3))
            self.assert_partitioned(d[np.argpartition(d, (0, 3))], (0, 3))

            assert_array_equal(p, np.partition(d, (-3, -1)))
            assert_array_equal(p, d[np.argpartition(d, (-3, -1))])

            d = np.arange(17)
            np.random.shuffle(d)
            d.partition(range(d.size))
            assert_array_equal(np.arange(17), d)
            np.random.shuffle(d)
            assert_array_equal(np.arange(17), d[d.argpartition(range(d.size))])

            # test unsorted kth
            d = np.arange(17)
            np.random.shuffle(d)
            keys = np.array([1, 3, 8, -2])
            np.random.shuffle(d)
            p = np.partition(d, keys)
            self.assert_partitioned(p, keys)
            p = d[np.argpartition(d, keys)]
            self.assert_partitioned(p, keys)
            np.random.shuffle(keys)
            assert_array_equal(np.partition(d, keys), p)
            assert_array_equal(d[np.argpartition(d, keys)], p)

            # equal kth
            d = np.arange(20)[::-1]
            self.assert_partitioned(np.partition(d, [5]*4), [5])
            self.assert_partitioned(np.partition(d, [5]*4 + [6, 13]),
                                    [5]*4 + [6, 13])
            self.assert_partitioned(d[np.argpartition(d, [5]*4)], [5])
            self.assert_partitioned(d[np.argpartition(d, [5]*4 + [6, 13])],
                                    [5]*4 + [6, 13])

            d = np.arange(12)
            np.random.shuffle(d)
            d1 = np.tile(np.arange(12), (4, 1))
            map(np.random.shuffle, d1)
            d0 = np.transpose(d1)

            kth = (1, 6, 7, -1)
            p = np.partition(d1, kth, axis=1)
            pa = d1[np.arange(d1.shape[0])[:, None],
                    d1.argpartition(kth, axis=1)]
            assert_array_equal(p, pa)
            for i in range(d1.shape[0]):
                self.assert_partitioned(p[i,:], kth)
            p = np.partition(d0, kth, axis=0)
            pa = d0[np.argpartition(d0, kth, axis=0),
                    np.arange(d0.shape[1])[None,:]]
            assert_array_equal(p, pa)
            for i in range(d0.shape[1]):
                self.assert_partitioned(p[:, i], kth)

    @pytest.mark.xfail(reason="TODO partition")
    def test_partition_fuzz(self):
        # a few rounds of random data testing
        for j in range(10, 30):
            for i in range(1, j - 2):
                d = np.arange(j)
                np.random.shuffle(d)
                d = d % np.random.randint(2, 30)
                idx = np.random.randint(d.size)
                kth = [0, idx, i, i + 1]
                tgt = np.sort(d)[kth]
                assert_array_equal(np.partition(d, kth)[kth], tgt,
                                   err_msg="data: %r\n kth: %r" % (d, kth))

    @pytest.mark.xfail(reason="TODO partition")
    @pytest.mark.parametrize("kth_dtype", np.typecodes["AllInteger"])
    def test_argpartition_gh5524(self, kth_dtype):
        #  A test for functionality of argpartition on lists.
        kth = np.array(1, dtype=kth_dtype)[()]
        d = [6, 7, 3, 2, 9, 0]
        p = np.argpartition(d, kth)
        self.assert_partitioned(np.array(d)[p],[1])

    @pytest.mark.xfail(reason="TODO order='F'")
    def test_flatten(self):
        x0 = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], np.int32)
        y0 = np.array([1, 2, 3, 4, 5, 6], np.int32)
        y0f = np.array([1, 4, 2, 5, 3, 6], np.int32)
        y1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], np.int32)
        y1f = np.array([1, 5, 3, 7, 2, 6, 4, 8], np.int32)
        assert_equal(x0.flatten(), y0)
        assert_equal(x0.flatten('F'), y0f)
        assert_equal(x0.flatten('F'), x0.T.flatten())
        assert_equal(x1.flatten(), y1)
        assert_equal(x1.flatten('F'), y1f)
        assert_equal(x1.flatten('F'), x1.T.flatten())


    @pytest.mark.parametrize('func', (np.dot, np.matmul))
    def test_arr_mult(self, func):
        a = np.array([[1, 0], [0, 1]])
        b = np.array([[0, 1], [1, 0]])
        c = np.array([[9, 1], [1, -9]])
        d = np.arange(24).reshape(4, 6)
        ddt = np.array(
            [[  55,  145,  235,  325],
             [ 145,  451,  757, 1063],
             [ 235,  757, 1279, 1801],
             [ 325, 1063, 1801, 2539]]
        )
        dtd = np.array(
            [[504, 540, 576, 612, 648, 684],
             [540, 580, 620, 660, 700, 740],
             [576, 620, 664, 708, 752, 796],
             [612, 660, 708, 756, 804, 852],
             [648, 700, 752, 804, 856, 908],
             [684, 740, 796, 852, 908, 964]]
        )


        # gemm vs syrk optimizations
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            eaf = a.astype(et)
            assert_equal(func(eaf, eaf), eaf)
            assert_equal(func(eaf.T, eaf), eaf)
            assert_equal(func(eaf, eaf.T), eaf)
            assert_equal(func(eaf.T, eaf.T), eaf)
            assert_equal(func(eaf.T.copy(), eaf), eaf)
            assert_equal(func(eaf, eaf.T.copy()), eaf)
            assert_equal(func(eaf.T.copy(), eaf.T.copy()), eaf)

        # syrk validations
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            eaf = a.astype(et)
            ebf = b.astype(et)
            assert_equal(func(ebf, ebf), eaf)
            assert_equal(func(ebf.T, ebf), eaf)
            assert_equal(func(ebf, ebf.T), eaf)
            assert_equal(func(ebf.T, ebf.T), eaf)
        # syrk - different shape
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            edf = d.astype(et)
            eddtf = ddt.astype(et)
            edtdf = dtd.astype(et)
            assert_equal(func(edf, edf.T), eddtf)
            assert_equal(func(edf.T, edf), edtdf)

            assert_equal(
                func(edf[:edf.shape[0] // 2, :], edf[::2, :].T),
                func(edf[:edf.shape[0] // 2, :].copy(), edf[::2, :].T.copy())
            )
            assert_equal(
                func(edf[::2, :], edf[:edf.shape[0] // 2, :].T),
                func(edf[::2, :].copy(), edf[:edf.shape[0] // 2, :].T.copy())
            )


    @pytest.mark.skip(reason="dot/matmul with negative strides")
    @pytest.mark.parametrize('func', (np.dot, np.matmul))
    def test_arr_mult_2(self, func):
        # syrk - different shape, stride, and view validations
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            edf = d.astype(et)
            assert_equal(
                func(edf[::-1, :], edf.T),
                func(edf[::-1, :].copy(), edf.T.copy())
            )
            assert_equal(
                func(edf[:, ::-1], edf.T),
                func(edf[:, ::-1].copy(), edf.T.copy())
            )
            assert_equal(
                func(edf, edf[::-1, :].T),
                func(edf, edf[::-1, :].T.copy())
            )
            assert_equal(
                func(edf, edf[:, ::-1].T),
                func(edf, edf[:, ::-1].T.copy())
            )

    @pytest.mark.parametrize('func', (np.dot, np.matmul))
    @pytest.mark.parametrize('dtype', 'ifdFD')
    def test_no_dgemv(self, func, dtype):
        # check vector arg for contiguous before gemv
        # gh-12156
        a = np.arange(8.0, dtype=dtype).reshape(2, 4)
        b = np.broadcast_to(1., (4, 1))
        ret1 = func(a, b)
        ret2 = func(a, b.copy())
        assert_equal(ret1, ret2)

        ret1 = func(b.T, a.T)
        ret2 = func(b.T.copy(), a.T)
        assert_equal(ret1, ret2)


    @pytest.mark.skip(reason="__array_interface__")
    @pytest.mark.parametrize('func', (np.dot, np.matmul))
    @pytest.mark.parametrize('dtype', 'ifdFD')
    def test_no_dgemv_2(self, func, dtype):
        # check for unaligned data
        dt = np.dtype(dtype)
        a = np.zeros(8 * dt.itemsize // 2 + 1, dtype='int16')[1:].view(dtype)
        a = a.reshape(2, 4)
        b = a[0]
        # make sure it is not aligned
        assert_(a.__array_interface__['data'][0] % dt.itemsize != 0)
        ret1 = func(a, b)
        ret2 = func(a.copy(), b.copy())
        assert_equal(ret1, ret2)

        ret1 = func(b.T, a.T)
        ret2 = func(b.T.copy(), a.T.copy())
        assert_equal(ret1, ret2)

    def test_dot(self):
        a = np.array([[1, 0], [0, 1]])
        b = np.array([[0, 1], [1, 0]])
        c = np.array([[9, 1], [1, -9]])
        # function versus methods
        assert_equal(np.dot(a, b), a.dot(b))
        assert_equal(np.dot(np.dot(a, b), c), a.dot(b).dot(c))

        # test passing in an output array
        c = np.zeros_like(a)
        a.dot(b, c)
        assert_equal(c, np.dot(a, b))

        # test keyword args
        c = np.zeros_like(a)
        a.dot(b=b, out=c)
        assert_equal(c, np.dot(a, b))


    @pytest.mark.xfail(reason="_aligned_zeros")
    def test_dot_out_mem_overlap(self):
        np.random.seed(1)

        # Test BLAS and non-BLAS code paths, including all dtypes
        # that dot() supports
        dtypes = [np.dtype(code) for code in np.typecodes['All']
                  if code not in 'USVM']
        for dtype in dtypes:
            a = np.random.rand(3, 3).astype(dtype)

            # Valid dot() output arrays must be aligned
            b = _aligned_zeros((3, 3), dtype=dtype)
            b[...] = np.random.rand(3, 3)

            y = np.dot(a, b)
            x = np.dot(a, b, out=b)
            assert_equal(x, y, err_msg=repr(dtype))

            # Check invalid output array
            assert_raises(ValueError, np.dot, a, b, out=b[::2])
            assert_raises(ValueError, np.dot, a, b, out=b.T)

    @pytest.mark.xfail(reason="TODO: overlapping memor in matmul")
    def test_matmul_out(self):
        # overlapping memory
        a = np.arange(18).reshape(2, 3, 3)
        b = np.matmul(a, a)
        c = np.matmul(a, a, out=a)
        assert_(c is a)
        assert_equal(c, b)
        a = np.arange(18).reshape(2, 3, 3)
        c = np.matmul(a, a, out=a[::-1, ...])
        assert_(c.base is a.base)
        assert_equal(c, b)

    def test_diagonal(self):
        a = np.arange(12).reshape((3, 4))
        assert_equal(a.diagonal(), [0, 5, 10])
        assert_equal(a.diagonal(0), [0, 5, 10])
        assert_equal(a.diagonal(1), [1, 6, 11])
        assert_equal(a.diagonal(-1), [4, 9])
        assert_raises(np.AxisError, a.diagonal, axis1=0, axis2=5)
        assert_raises(np.AxisError, a.diagonal, axis1=5, axis2=0)
        assert_raises(np.AxisError, a.diagonal, axis1=5, axis2=5)
        assert_raises((ValueError, RuntimeError), a.diagonal, axis1=1, axis2=1)

        b = np.arange(8).reshape((2, 2, 2))
        assert_equal(b.diagonal(), [[0, 6], [1, 7]])
        assert_equal(b.diagonal(0), [[0, 6], [1, 7]])
        assert_equal(b.diagonal(1), [[2], [3]])
        assert_equal(b.diagonal(-1), [[4], [5]])
        assert_raises((ValueError, RuntimeError), b.diagonal, axis1=0, axis2=0)
        assert_equal(b.diagonal(0, 1, 2), [[0, 3], [4, 7]])
        assert_equal(b.diagonal(0, 0, 1), [[0, 6], [1, 7]])
        assert_equal(b.diagonal(offset=1, axis1=0, axis2=2), [[1], [3]])
        # Order of axis argument doesn't matter:
        assert_equal(b.diagonal(0, 2, 1), [[0, 3], [4, 7]])

    @pytest.mark.xfail(reason="no readonly views")
    def test_diagonal_view_notwriteable(self):
        a = np.eye(3).diagonal()
        assert_(not a.flags.writeable)
        assert_(not a.flags.owndata)

        a = np.diagonal(np.eye(3))
        assert_(not a.flags.writeable)
        assert_(not a.flags.owndata)

        a = np.diag(np.eye(3))
        assert_(not a.flags.writeable)
        assert_(not a.flags.owndata)

    def test_diagonal_memleak(self):
        # Regression test for a bug that crept in at one point
        a = np.zeros((100, 100))
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(a) < 50)
        for i in range(100):
            a.diagonal()
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(a) < 50)

    def test_size_zero_memleak(self):
        # Regression test for issue 9615
        # Exercises a special-case code path for dot products of length
        # zero in cblasfuncs (making it is specific to floating dtypes).
        a = np.array([], dtype=np.float64)
        x = np.array(2.0)
        for _ in range(100):
            np.dot(a, a, out=x)
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(x) < 50)

    def test_trace(self):
        a = np.arange(12).reshape((3, 4))
        assert_equal(a.trace(), 15)
        assert_equal(a.trace(0), 15)
        assert_equal(a.trace(1), 18)
        assert_equal(a.trace(-1), 13)

        b = np.arange(8).reshape((2, 2, 2))
        assert_equal(b.trace(), [6, 8])
        assert_equal(b.trace(0), [6, 8])
        assert_equal(b.trace(1), [2, 3])
        assert_equal(b.trace(-1), [4, 5])
        assert_equal(b.trace(0, 0, 1), [6, 8])
        assert_equal(b.trace(0, 0, 2), [5, 9])
        assert_equal(b.trace(0, 1, 2), [3, 11])
        assert_equal(b.trace(offset=1, axis1=0, axis2=2), [1, 3])

        out = np.array(1)
        ret = a.trace(out=out)
        assert ret is out

    def test_put(self):
        icodes = np.typecodes['AllInteger']
        fcodes = np.typecodes['AllFloat']
        for dt in icodes + fcodes:
            tgt = np.array([0, 1, 0, 3, 0, 5], dtype=dt)

            # test 1-d
            a = np.zeros(6, dtype=dt)
            a.put([1, 3, 5], [1, 3, 5])
            assert_equal(a, tgt)

            # test 2-d
            a = np.zeros((2, 3), dtype=dt)
            a.put([1, 3, 5], [1, 3, 5])
            assert_equal(a, tgt.reshape(2, 3))

        for dt in '?':
            tgt = np.array([False, True, False, True, False, True], dtype=dt)

            # test 1-d
            a = np.zeros(6, dtype=dt)
            a.put([1, 3, 5], [True]*3)
            assert_equal(a, tgt)

            # test 2-d
            a = np.zeros((2, 3), dtype=dt)
            a.put([1, 3, 5], [True]*3)
            assert_equal(a, tgt.reshape(2, 3))

        # when calling np.put, make sure a
        # TypeError is raised if the object
        # isn't an ndarray
        bad_array = [1, 2, 3]
        assert_raises(TypeError, np.put, bad_array, [0, 2], 5)

    @pytest.mark.xfail(reason="TODO: implement order='F'")
    def test_ravel(self):
        a = np.array([[0, 1], [2, 3]])
        assert_equal(a.ravel(), [0, 1, 2, 3])
        assert_(not a.ravel().flags.owndata)
        assert_equal(a.ravel('F'), [0, 2, 1, 3])
        assert_equal(a.ravel(order='C'), [0, 1, 2, 3])
        assert_equal(a.ravel(order='F'), [0, 2, 1, 3])
        assert_equal(a.ravel(order='A'), [0, 1, 2, 3])
        assert_(not a.ravel(order='A').flags.owndata)
        assert_equal(a.ravel(order='K'), [0, 1, 2, 3])
        assert_(not a.ravel(order='K').flags.owndata)
        assert_equal(a.ravel(), a.reshape(-1))

        a = np.array([[0, 1], [2, 3]], order='F')
        assert_equal(a.ravel(), [0, 1, 2, 3])
        assert_equal(a.ravel(order='A'), [0, 2, 1, 3])
        assert_equal(a.ravel(order='K'), [0, 2, 1, 3])
        assert_(not a.ravel(order='A').flags.owndata)
        assert_(not a.ravel(order='K').flags.owndata)
        assert_equal(a.ravel(), a.reshape(-1))
        assert_equal(a.ravel(order='A'), a.reshape(-1, order='A'))

        a = np.array([[0, 1], [2, 3]])[::-1, :]
        assert_equal(a.ravel(), [2, 3, 0, 1])
        assert_equal(a.ravel(order='C'), [2, 3, 0, 1])
        assert_equal(a.ravel(order='F'), [2, 0, 3, 1])
        assert_equal(a.ravel(order='A'), [2, 3, 0, 1])
        # 'K' doesn't reverse the axes of negative strides
        assert_equal(a.ravel(order='K'), [2, 3, 0, 1])
        assert_(a.ravel(order='K').flags.owndata)

        # Test simple 1-d copy behaviour:
        a = np.arange(10)[::2]
        assert_(a.ravel('K').flags.owndata)
        assert_(a.ravel('C').flags.owndata)
        assert_(a.ravel('F').flags.owndata)

        # Not contiguous and 1-sized axis with non matching stride
        a = np.arange(2**3 * 2)[::2]
        a = a.reshape(2, 1, 2, 2).swapaxes(-1, -2)
        strides = list(a.strides)
        strides[1] = 123
        a.strides = strides
        assert_(a.ravel(order='K').flags.owndata)
        assert_equal(a.ravel('K'), np.arange(0, 15, 2))

        # contiguous and 1-sized axis with non matching stride works:
        a = np.arange(2**3)
        a = a.reshape(2, 1, 2, 2).swapaxes(-1, -2)
        strides = list(a.strides)
        strides[1] = 123
        a.strides = strides
        assert_(np.may_share_memory(a.ravel(order='K'), a))
        assert_equal(a.ravel(order='K'), np.arange(2**3))

        # Test negative strides (not very interesting since non-contiguous):
        a = np.arange(4)[::-1].reshape(2, 2)
        assert_(a.ravel(order='C').flags.owndata)
        assert_(a.ravel(order='K').flags.owndata)
        assert_equal(a.ravel('C'), [3, 2, 1, 0])
        assert_equal(a.ravel('K'), [3, 2, 1, 0])

        # 1-element tidy strides test:
        a = np.array([[1]])
        a.strides = (123, 432)
        # If the following stride is not 8, NPY_RELAXED_STRIDES_DEBUG is
        # messing them up on purpose:
        if np.ones(1).strides == (8,):
            assert_(np.may_share_memory(a.ravel('K'), a))
            assert_equal(a.ravel('K').strides, (a.dtype.itemsize,))

        for order in ('C', 'F', 'A', 'K'):
            # 0-d corner case:
            a = np.array(0)
            assert_equal(a.ravel(order), [0])
            assert_(np.may_share_memory(a.ravel(order), a))

        # Test that certain non-inplace ravels work right (mostly) for 'K':
        b = np.arange(2**4 * 2)[::2].reshape(2, 2, 2, 2)
        a = b[..., ::2]
        assert_equal(a.ravel('K'), [0, 4, 8, 12, 16, 20, 24, 28])
        assert_equal(a.ravel('C'), [0, 4, 8, 12, 16, 20, 24, 28])
        assert_equal(a.ravel('A'), [0, 4, 8, 12, 16, 20, 24, 28])
        assert_equal(a.ravel('F'), [0, 16, 8, 24, 4, 20, 12, 28])

        a = b[::2, ...]
        assert_equal(a.ravel('K'), [0, 2, 4, 6, 8, 10, 12, 14])
        assert_equal(a.ravel('C'), [0, 2, 4, 6, 8, 10, 12, 14])
        assert_equal(a.ravel('A'), [0, 2, 4, 6, 8, 10, 12, 14])
        assert_equal(a.ravel('F'), [0, 8, 4, 12, 2, 10, 6, 14])

    def test_swapaxes(self):
        a = np.arange(1*2*3*4).reshape(1, 2, 3, 4).copy()
        idx = np.indices(a.shape)
        assert_(a.flags['OWNDATA'])
        b = a.copy()
        # check exceptions
        assert_raises(np.AxisError, a.swapaxes, -5, 0)
        assert_raises(np.AxisError, a.swapaxes, 4, 0)
        assert_raises(np.AxisError, a.swapaxes, 0, -5)
        assert_raises(np.AxisError, a.swapaxes, 0, 4)

        for i in range(-4, 4):
            for j in range(-4, 4):
                for k, src in enumerate((a, b)):
                    c = src.swapaxes(i, j)
                    # check shape
                    shape = list(src.shape)
                    shape[i] = src.shape[j]
                    shape[j] = src.shape[i]
                    assert_equal(c.shape, shape, str((i, j, k)))
                    # check array contents
                    i0, i1, i2, i3 = [dim-1 for dim in c.shape]
                    j0, j1, j2, j3 = [dim-1 for dim in src.shape]
                    assert_equal(src[idx[j0], idx[j1], idx[j2], idx[j3]],
                                 c[idx[i0], idx[i1], idx[i2], idx[i3]],
                                 str((i, j, k)))
                    # check a view is always returned, gh-5260
                    assert_(not c.flags['OWNDATA'], str((i, j, k)))
                    # check on non-contiguous input array
                    if k == 1:
                        b = c

    def test_conjugate(self):
        a = np.array([1-1j, 1+1j, 23+23.0j])
        ac = a.conj()
        assert_equal(a.real, ac.real)
        assert_equal(a.imag, -ac.imag)
        assert_equal(ac, a.conjugate())
        assert_equal(ac, np.conjugate(a))

        a = np.array([1-1j, 1+1j, 23+23.0j], 'F')
        ac = a.conj()
        assert_equal(a.real, ac.real)
        assert_equal(a.imag, -ac.imag)
        assert_equal(ac, a.conjugate())
        assert_equal(ac, np.conjugate(a))

        a = np.array([1, 2, 3])
        ac = a.conj()
        assert_equal(a, ac)
        assert_equal(ac, a.conjugate())
        assert_equal(ac, np.conjugate(a))

        a = np.array([1.0, 2.0, 3.0])
        ac = a.conj()
        assert_equal(a, ac)
        assert_equal(ac, a.conjugate())
        assert_equal(ac, np.conjugate(a))


    @pytest.mark.xfail(reason="TODO: ndarray.conjugate with out")
    def test_conjugate_out(self):
        # Minimal test for the out argument being passed on correctly
        # NOTE: The ability to pass `out` is currently undocumented!
        a = np.array([1-1j, 1+1j, 23+23.0j])
        out = np.empty_like(a)
        res = a.conjugate(out)
        assert res is out
        assert_array_equal(out, a.conjugate())

    def test__complex__(self):
        dtypes = ['i1', 'i2', 'i4', 'i8',
                  'u1',
                  'f', 'd', 'F', 'D',
                  '?', ]
        for dt in dtypes:
            a = np.array(7, dtype=dt)
            b = np.array([7], dtype=dt)
            c = np.array([[[[[7]]]]], dtype=dt)

            msg = 'dtype: {0}'.format(dt)
            ap = complex(a)
            assert_equal(ap, a, msg)
            bp = complex(b)
            assert_equal(bp, b, msg)
            cp = complex(c)
            assert_equal(cp, c, msg)

    def test__complex__should_not_work(self):
        dtypes = ['i1', 'i2', 'i4', 'i8',
                  'u1',
                  'f', 'd', 'F', 'D',
                  '?',]
        for dt in dtypes:
            a = np.array([1, 2, 3], dtype=dt)
            assert_raises(TypeError, complex, a)

        c = np.array([(1.0, 3), (2e-3, 7)], dtype=dt)
        assert_raises(TypeError, complex, c)


@pytest.mark.xfail(reason='TODO')
class TestCequenceMethods:
    def test_array_contains(self):
        assert_(4.0 in np.arange(16.).reshape(4,4))
        assert_(20.0 not in np.arange(16.).reshape(4,4))


class TestBinop:
    def test_inplace(self):
        # test refcount 1 inplace conversion
        assert_array_almost_equal(np.array([0.5]) * np.array([1.0, 2.0]),
                                  [0.5, 1.0])

        d = np.array([0.5, 0.5])[::2]
        assert_array_almost_equal(d * (d * np.array([1.0, 2.0])),
                                  [0.25, 0.5])

        a = np.array([0.5])
        b = np.array([0.5])
        c = a + b
        c = a - b
        c = a * b
        c = a / b
        assert_equal(a, b)
        assert_almost_equal(c, 1.)

        c = a + b * 2. / b * a - a / b
        assert_equal(a, b)
        assert_equal(c, 0.5)

        # true divide
        a = np.array([5])
        b = np.array([3])
        c = (a * a) / b

        assert_almost_equal(c, 25 / 3, decimal=5)
        assert_equal(a, 5)
        assert_equal(b, 3)

 

@pytest.mark.skip(reason='segfaults')
class TestTemporaryElide:
    # elision is only triggered on relatively large arrays

    def test_extension_incref_elide(self):
        # test extension (e.g. cython) calling PyNumber_* slots without
        # increasing the reference counts
        #
        # def incref_elide(a):
        #    d = input.copy() # refcount 1
        #    return d, d + d # PyNumber_Add without increasing refcount
        from numpy.core._multiarray_tests import incref_elide
        d = np.ones(100000)
        orig, res = incref_elide(d)
        d + d
        # the return original should not be changed to an inplace operation
        assert_array_equal(orig, d)
        assert_array_equal(res, d + d)

    def test_extension_incref_elide_stack(self):
        # scanning if the refcount == 1 object is on the python stack to check
        # that we are called directly from python is flawed as object may still
        # be above the stack pointer and we have no access to the top of it
        #
        # def incref_elide_l(d):
        #    return l[4] + l[4] # PyNumber_Add without increasing refcount
        from numpy.core._multiarray_tests import incref_elide_l
        # padding with 1 makes sure the object on the stack is not overwritten
        l = [1, 1, 1, 1, np.ones(100000)]
        res = incref_elide_l(l)
        # the return original should not be changed to an inplace operation
        assert_array_equal(l[4], np.ones(100000))
        assert_array_equal(res, l[4] + l[4])

    def test_temporary_with_cast(self):
        # check that we don't elide into a temporary which would need casting
        d = np.ones(200000, dtype=np.int64)
        assert_equal(((d + d) + 2**222).dtype, np.dtype('O'))

        r = ((d + d) / 2)
        assert_equal(r.dtype, np.dtype('f8'))

        r = np.true_divide((d + d), 2)
        assert_equal(r.dtype, np.dtype('f8'))

        r = ((d + d) / 2.)
        assert_equal(r.dtype, np.dtype('f8'))

        r = ((d + d) // 2)
        assert_equal(r.dtype, np.dtype(np.int64))

        # commutative elision into the astype result
        f = np.ones(100000, dtype=np.float32)
        assert_equal(((f + f) + f.astype(np.float64)).dtype, np.dtype('f8'))

        # no elision into lower type
        d = f.astype(np.float64)
        assert_equal(((f + f) + d).dtype, d.dtype)
        l = np.ones(100000, dtype=np.longdouble)
        assert_equal(((d + d) + l).dtype, l.dtype)

        # test unary abs with different output dtype
        for dt in (np.complex64, np.complex128, np.clongdouble):
            c = np.ones(100000, dtype=dt)
            r = abs(c * 2.0)
            assert_equal(r.dtype, np.dtype('f%d' % (c.itemsize // 2)))

    def test_elide_broadcast(self):
        # test no elision on broadcast to higher dimension
        # only triggers elision code path in debug mode as triggering it in
        # normal mode needs 256kb large matching dimension, so a lot of memory
        d = np.ones((2000, 1), dtype=int)
        b = np.ones((2000), dtype=bool)
        r = (1 - d) + b
        assert_equal(r, 1)
        assert_equal(r.shape, (2000, 2000))

    def test_elide_scalar(self):
        # check inplace op does not create ndarray from scalars
        a = np.bool_()
        assert_(type(~(a & a)) is np.bool_)

    def test_elide_scalar_readonly(self):
        # The imaginary part of a real array is readonly. This needs to go
        # through fast_scalar_power which is only called for powers of
        # +1, -1, 0, 0.5, and 2, so use 2. Also need valid refcount for
        # elision which can be gotten for the imaginary part of a real
        # array. Should not error.
        a = np.empty(100000, dtype=np.float64)
        a.imag ** 2

    def test_elide_readonly(self):
        # don't try to elide readonly temporaries
        r = np.asarray(np.broadcast_to(np.zeros(1), 100000).flat) * 0.0
        assert_equal(r, 0)

    def test_elide_updateifcopy(self):
        a = np.ones(2**20)[::2]
        b = a.flat.__array__() + 1
        del b
        assert_equal(a, 1)

@pytest.mark.skip(reason='dont worry about capi')
class TestCAPI:
    def test_IsPythonScalar(self):
        from numpy.core._multiarray_tests import IsPythonScalar
        assert_(IsPythonScalar(b'foobar'))
        assert_(IsPythonScalar(1))
        assert_(IsPythonScalar(2**80))
        assert_(IsPythonScalar(2.))
        assert_(IsPythonScalar("a"))

    @pytest.mark.parametrize("converter",
             [_multiarray_tests.run_scalar_intp_converter,
              _multiarray_tests.run_scalar_intp_from_sequence])
    def test_intp_sequence_converters(self, converter):
        # Test simple values (-1 is special for error return paths)
        assert converter(10) == (10,)
        assert converter(-1) == (-1,)
        # A 0-D array looks a bit like a sequence but must take the integer
        # path:
        assert converter(np.array(123)) == (123,)
        # Test simple sequences (intp_from_sequence only supports length 1):
        assert converter((10,)) == (10,)
        assert converter(np.array([11])) == (11,)

    @pytest.mark.parametrize("converter",
             [_multiarray_tests.run_scalar_intp_converter,
              _multiarray_tests.run_scalar_intp_from_sequence])
    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
            reason="PyPy bug in error formatting")
    def test_intp_sequence_converters_errors(self, converter):
        with pytest.raises(TypeError,
                match="expected a sequence of integers or a single integer, "):
            converter(object())
        with pytest.raises(TypeError,
                match="expected a sequence of integers or a single integer, "
                      "got '32.0'"):
            converter(32.)
        with pytest.raises(TypeError,
                match="'float' object cannot be interpreted as an integer"):
            converter([32.])
        with pytest.raises(ValueError,
                match="Maximum allowed dimension"):
            # These converters currently convert overflows to a ValueError
            converter(2**64)

@pytest.mark.xfail(reason='TODO')
class TestSubscripting:
    def test_test_zero_rank(self):
        x = np.array([1, 2, 3])
        assert_(isinstance(x[0], np.int_))
        assert_(type(x[0, ...]) is np.ndarray)


@pytest.mark.xfail(reason='TODO')
class TestPickling:
    @pytest.mark.skipif(pickle.HIGHEST_PROTOCOL >= 5,
                        reason=('this tests the error messages when trying to'
                                'protocol 5 although it is not available'))
    def test_correct_protocol5_error_message(self):
        array = np.arange(10)

    def test_record_array_with_object_dtype(self):
        my_object = object()

        arr_with_object = np.array(
                [(my_object, 1, 2.0)],
                dtype=[('a', object), ('b', int), ('c', float)])
        arr_without_object = np.array(
                [('xxx', 1, 2.0)],
                dtype=[('a', str), ('b', int), ('c', float)])

        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            depickled_arr_with_object = pickle.loads(
                    pickle.dumps(arr_with_object, protocol=proto))
            depickled_arr_without_object = pickle.loads(
                    pickle.dumps(arr_without_object, protocol=proto))

            assert_equal(arr_with_object.dtype,
                         depickled_arr_with_object.dtype)
            assert_equal(arr_without_object.dtype,
                         depickled_arr_without_object.dtype)

    @pytest.mark.skipif(pickle.HIGHEST_PROTOCOL < 5,
                        reason="requires pickle protocol 5")
    def test_f_contiguous_array(self):
        f_contiguous_array = np.array([[1, 2, 3], [4, 5, 6]], order='F')
        buffers = []

        # When using pickle protocol 5, Fortran-contiguous arrays can be
        # serialized using out-of-band buffers
        bytes_string = pickle.dumps(f_contiguous_array, protocol=5,
                                    buffer_callback=buffers.append)

        assert len(buffers) > 0

        depickled_f_contiguous_array = pickle.loads(bytes_string,
                                                    buffers=buffers)

        assert_equal(f_contiguous_array, depickled_f_contiguous_array)

    def test_non_contiguous_array(self):
        non_contiguous_array = np.arange(12).reshape(3, 4)[:, :2]
        assert not non_contiguous_array.flags.c_contiguous
        assert not non_contiguous_array.flags.f_contiguous

        # make sure non-contiguous arrays can be pickled-depickled
        # using any protocol
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            depickled_non_contiguous_array = pickle.loads(
                    pickle.dumps(non_contiguous_array, protocol=proto))

            assert_equal(non_contiguous_array, depickled_non_contiguous_array)

    def test_roundtrip(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            carray = np.array([[2, 9], [7, 0], [3, 8]])
            DATA = [
                carray,
                np.transpose(carray),
                np.array([('xxx', 1, 2.0)], dtype=[('a', (str, 3)), ('b', int),
                                                   ('c', float)])
            ]

            refs = [weakref.ref(a) for a in DATA]
            for a in DATA:
                assert_equal(
                        a, pickle.loads(pickle.dumps(a, protocol=proto)),
                        err_msg="%r" % a)
            del a, DATA, carray
            break_cycles()
            # check for reference leaks (gh-12793)
            for ref in refs:
                assert ref() is None

    def _loads(self, obj):
        return pickle.loads(obj, encoding='latin1')

    # version 0 pickles, using protocol=2 to pickle
    # version 0 doesn't have a version field
    def test_version0_int8(self):
        s = b'\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x04\x85cnumpy\ndtype\nq\x04U\x02i1K\x00K\x01\x87Rq\x05(U\x01|NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89U\x04\x01\x02\x03\x04tb.'
        a = np.array([1, 2, 3, 4], dtype=np.int8)
        p = self._loads(s)
        assert_equal(a, p)

    def test_version0_float32(self):
        s = b'\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x04\x85cnumpy\ndtype\nq\x04U\x02f4K\x00K\x01\x87Rq\x05(U\x01<NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89U\x10\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@tb.'
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        p = self._loads(s)
        assert_equal(a, p)

    def test_version0_object(self):
        s = b'\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x02\x85cnumpy\ndtype\nq\x04U\x02O8K\x00K\x01\x87Rq\x05(U\x01|NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89]q\x06(}q\x07U\x01aK\x01s}q\x08U\x01bK\x02setb.'
        a = np.array([{'a': 1}, {'b': 2}])
        p = self._loads(s)
        assert_equal(a, p)

    # version 1 pickles, using protocol=2 to pickle
    def test_version1_int8(self):
        s = b'\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x04\x85cnumpy\ndtype\nq\x04U\x02i1K\x00K\x01\x87Rq\x05(K\x01U\x01|NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89U\x04\x01\x02\x03\x04tb.'
        a = np.array([1, 2, 3, 4], dtype=np.int8)
        p = self._loads(s)
        assert_equal(a, p)

    def test_version1_float32(self):
        s = b'\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x04\x85cnumpy\ndtype\nq\x04U\x02f4K\x00K\x01\x87Rq\x05(K\x01U\x01<NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89U\x10\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@tb.'
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        p = self._loads(s)
        assert_equal(a, p)

    def test_version1_object(self):
        s = b'\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x02\x85cnumpy\ndtype\nq\x04U\x02O8K\x00K\x01\x87Rq\x05(K\x01U\x01|NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89]q\x06(}q\x07U\x01aK\x01s}q\x08U\x01bK\x02setb.'
        a = np.array([{'a': 1}, {'b': 2}])
        p = self._loads(s)
        assert_equal(a, p)

    def test_subarray_int_shape(self):
        s = b"cnumpy.core.multiarray\n_reconstruct\np0\n(cnumpy\nndarray\np1\n(I0\ntp2\nS'b'\np3\ntp4\nRp5\n(I1\n(I1\ntp6\ncnumpy\ndtype\np7\n(S'V6'\np8\nI0\nI1\ntp9\nRp10\n(I3\nS'|'\np11\nN(S'a'\np12\ng3\ntp13\n(dp14\ng12\n(g7\n(S'V4'\np15\nI0\nI1\ntp16\nRp17\n(I3\nS'|'\np18\n(g7\n(S'i1'\np19\nI0\nI1\ntp20\nRp21\n(I3\nS'|'\np22\nNNNI-1\nI-1\nI0\ntp23\nb(I2\nI2\ntp24\ntp25\nNNI4\nI1\nI0\ntp26\nbI0\ntp27\nsg3\n(g7\n(S'V2'\np28\nI0\nI1\ntp29\nRp30\n(I3\nS'|'\np31\n(g21\nI2\ntp32\nNNI2\nI1\nI0\ntp33\nbI4\ntp34\nsI6\nI1\nI0\ntp35\nbI00\nS'\\x01\\x01\\x01\\x01\\x01\\x02'\np36\ntp37\nb."
        a = np.array([(1, (1, 2))], dtype=[('a', 'i1', (2, 2)), ('b', 'i1', 2)])
        p = self._loads(s)
        assert_equal(a, p)


class TestFancyIndexing:
    def test_list(self):
        x = np.ones((1, 1))
        x[:, [0]] = 2.0
        assert_array_equal(x, np.array([[2.0]]))

        x = np.ones((1, 1, 1))
        x[:, :, [0]] = 2.0
        assert_array_equal(x, np.array([[[2.0]]]))

    def test_tuple(self):
        x = np.ones((1, 1))
        x[:, (0,)] = 2.0
        assert_array_equal(x, np.array([[2.0]]))
        x = np.ones((1, 1, 1))
        x[:, :, (0,)] = 2.0
        assert_array_equal(x, np.array([[[2.0]]]))

    def test_mask(self):
        x = np.array([1, 2, 3, 4])
        m = np.array([0, 1, 0, 0], bool)
        assert_array_equal(x[m], np.array([2]))

    def test_mask2(self):
        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        m = np.array([0, 1], bool)
        m2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], bool)
        m3 = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], bool)
        assert_array_equal(x[m], np.array([[5, 6, 7, 8]]))
        assert_array_equal(x[m2], np.array([2, 5]))
        assert_array_equal(x[m3], np.array([2]))

    def test_assign_mask(self):
        x = np.array([1, 2, 3, 4])
        m = np.array([0, 1, 0, 0], bool)
        x[m] = 5
        assert_array_equal(x, np.array([1, 5, 3, 4]))

    def test_assign_mask2(self):
        xorig = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        m = np.array([0, 1], bool)
        m2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], bool)
        m3 = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], bool)
        x = xorig.copy()
        x[m] = 10
        assert_array_equal(x, np.array([[1, 2, 3, 4], [10, 10, 10, 10]]))
        x = xorig.copy()
        x[m2] = 10
        assert_array_equal(x, np.array([[1, 10, 3, 4], [10, 6, 7, 8]]))
        x = xorig.copy()
        x[m3] = 10
        assert_array_equal(x, np.array([[1, 10, 3, 4], [5, 6, 7, 8]]))


class TestArgmaxArgminCommon:

    sizes = [(), (3,), (3, 2), (2, 3),
             (3, 3), (2, 3, 4), (4, 3, 2),
             (1, 2, 3, 4), (2, 3, 4, 1),
             (3, 4, 1, 2), (4, 1, 2, 3),
             (64,), (128,), (256,)]

    @pytest.mark.parametrize("size, axis", itertools.chain(*[[(size, axis)
        for axis in list(range(-len(size), len(size))) + [None]]
        for size in sizes]))
    @pytest.mark.parametrize('method', [np.argmax, np.argmin])
    def test_np_argmin_argmax_keepdims(self, size, axis, method):

        arr = np.random.normal(size=size)

        # contiguous arrays
        if axis is None:
            new_shape = [1 for _ in range(len(size))]
        else:
            new_shape = list(size)
            new_shape[axis] = 1
        new_shape = tuple(new_shape)

        _res_orig = method(arr, axis=axis)
        res_orig = _res_orig.reshape(new_shape)
        res = method(arr, axis=axis, keepdims=True)
        assert_equal(res, res_orig)
        assert_(res.shape == new_shape)
        outarray = np.empty(res.shape, dtype=res.dtype)
        res1 = method(arr, axis=axis, out=outarray,
                            keepdims=True)
        assert_(res1 is outarray)
        assert_equal(res, outarray)

        if len(size) > 0:
            wrong_shape = list(new_shape)
            if axis is not None:
                wrong_shape[axis] = 2
            else:
                wrong_shape[0] = 2
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            with pytest.raises(ValueError):
                method(arr.T, axis=axis,
                        out=wrong_outarray, keepdims=True)

        # non-contiguous arrays
        if axis is None:
            new_shape = [1 for _ in range(len(size))]
        else:
            new_shape = list(size)[::-1]
            new_shape[axis] = 1
        new_shape = tuple(new_shape)

        _res_orig = method(arr.T, axis=axis)
        res_orig = _res_orig.reshape(new_shape)
        res = method(arr.T, axis=axis, keepdims=True)
        assert_equal(res, res_orig)
        assert_(res.shape == new_shape)
        outarray = np.empty(new_shape[::-1], dtype=res.dtype)
        outarray = outarray.T
        res1 = method(arr.T, axis=axis, out=outarray,
                            keepdims=True)
        assert_(res1 is outarray)
        assert_equal(res, outarray)

        if len(size) > 0:
            # one dimension lesser for non-zero sized
            # array should raise an error
            with pytest.raises(ValueError):
                method(arr[0], axis=axis,
                        out=outarray, keepdims=True)

        if len(size) > 0:
            wrong_shape = list(new_shape)
            if axis is not None:
                wrong_shape[axis] = 2
            else:
                wrong_shape[0] = 2
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            with pytest.raises(ValueError):
                method(arr.T, axis=axis,
                        out=wrong_outarray, keepdims=True)

    @pytest.mark.xfail(reason="TODO: implement choose")
    @pytest.mark.parametrize('method', ['max', 'min'])
    def test_all(self, method):
        a = np.random.normal(0, 1, (4, 5, 6, 7, 8))
        arg_method = getattr(a, 'arg' + method)
        val_method = getattr(a, method)
        for i in range(a.ndim):
            a_maxmin = val_method(i)
            aarg_maxmin = arg_method(i)
            axes = list(range(a.ndim))
            axes.remove(i)
            assert_(np.all(a_maxmin == aarg_maxmin.choose(
                                        *a.transpose(i, *axes))))

    @pytest.mark.parametrize('method', ['argmax', 'argmin'])
    def test_output_shape(self, method):
        # see also gh-616
        a = np.ones((10, 5))
        arg_method = getattr(a, method)
        # Check some simple shape mismatches
        out = np.ones(11, dtype=np.int_)
        assert_raises(ValueError, arg_method, -1, out)

        out = np.ones((2, 5), dtype=np.int_)
        assert_raises(ValueError, arg_method, -1, out)

        # these could be relaxed possibly (used to allow even the previous)
        out = np.ones((1, 10), dtype=np.int_)
        assert_raises(ValueError, arg_method, -1, out)

        out = np.ones(10, dtype=np.int_)
        arg_method(-1, out=out)
        assert_equal(out, arg_method(-1))

    @pytest.mark.parametrize('ndim', [0, 1])
    @pytest.mark.parametrize('method', ['argmax', 'argmin'])
    def test_ret_is_out(self, ndim, method):
        a = np.ones((4,) + (256,)*ndim)
        arg_method = getattr(a, method)
        out = np.empty((256,)*ndim, dtype=np.intp)
        ret = arg_method(axis=0, out=out)
        assert ret is out

    @pytest.mark.parametrize('arr_method, np_method',
        [('argmax', np.argmax),
         ('argmin', np.argmin)])
    def test_np_vs_ndarray(self, arr_method, np_method):
        # make sure both ndarray.argmax/argmin and
        # numpy.argmax/argmin support out/axis args
        a = np.random.normal(size=(2, 3))
        arg_method = getattr(a, arr_method)

        # check positional args
        out1 = np.zeros(2, dtype=int)
        out2 = np.zeros(2, dtype=int)
        assert_equal(arg_method(1, out1), np_method(a, 1, out2))
        assert_equal(out1, out2)

        # check keyword args
        out1 = np.zeros(3, dtype=int)
        out2 = np.zeros(3, dtype=int)
        assert_equal(arg_method(out=out1, axis=0),
                     np_method(a, out=out2, axis=0))
        assert_equal(out1, out2)


class TestArgmax:
    usg_data = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 0),
        ([3, 3, 3, 3,  2,  2,  2,  2], 0),
        ([0, 1, 2, 3,  4,  5,  6,  7], 7),
        ([7, 6, 5, 4,  3,  2,  1,  0], 0)
    ]
    sg_data = usg_data + [
        ([1, 2, 3, 4, -4, -3, -2, -1], 3),
        ([1, 2, 3, 4, -1, -2, -3, -4], 3)
    ]
    darr = [(np.array(d[0], dtype=t), d[1]) for d, t in (
        itertools.product(usg_data, (
            np.uint8,
        ))
    )]
    darr = darr + [(np.array(d[0], dtype=t), d[1]) for d, t in (
        itertools.product(sg_data, (
            np.int8, np.int16, np.int32, np.int64, np.float32, np.float64
        ))
    )]
    darr = darr + [(np.array(d[0], dtype=t), d[1]) for d, t in (
        itertools.product((
            ([0, 1, 2, 3, np.nan], 4),
            ([0, 1, 2, np.nan, 3], 3),
            ([np.nan, 0, 1, 2, 3], 0),
            ([np.nan, 0, np.nan, 2, 3], 0),
            # To hit the tail of SIMD multi-level(x4, x1) inner loops
            # on variant SIMD widthes
            ([1] * (2*5-1) + [np.nan], 2*5-1),
            ([1] * (4*5-1) + [np.nan], 4*5-1),
            ([1] * (8*5-1) + [np.nan], 8*5-1),
            ([1] * (16*5-1) + [np.nan], 16*5-1),
            ([1] * (32*5-1) + [np.nan], 32*5-1)
        ), (
            np.float32, np.float64
        ))
    )]
    nan_arr = darr + [
  # RuntimeError: "max_values_cpu" not implemented for 'ComplexDouble'
  #      ([0, 1, 2, 3, complex(0, np.nan)], 4),
  #      ([0, 1, 2, 3, complex(np.nan, 0)], 4),
  #      ([0, 1, 2, complex(np.nan, 0), 3], 3),
  #      ([0, 1, 2, complex(0, np.nan), 3], 3),
  #      ([complex(0, np.nan), 0, 1, 2, 3], 0),
  #      ([complex(np.nan, np.nan), 0, 1, 2, 3], 0),
  #      ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, 1)], 0),
  #      ([complex(np.nan, np.nan), complex(np.nan, 2), complex(np.nan, 1)], 0),
  #      ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, np.nan)], 0),

  #      ([complex(0, 0), complex(0, 2), complex(0, 1)], 1),
  #      ([complex(1, 0), complex(0, 2), complex(0, 1)], 0),
  #      ([complex(1, 0), complex(0, 2), complex(1, 1)], 2),

        ([False, False, False, False, True], 4),
        ([False, False, False, True, False], 3),
        ([True, False, False, False, False], 0),
        ([True, False, True, False, False], 0),
    ]

    @pytest.mark.parametrize('data', nan_arr)
    def test_combinations(self, data):
        arr, pos = data
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning,
                        "invalid value encountered in reduce")
            val = np.max(arr)

        assert_equal(np.argmax(arr), pos, err_msg="%r" % arr)
        assert_equal(arr[np.argmax(arr)], val, err_msg="%r" % arr)

        # add padding to test SIMD loops
        rarr = np.repeat(arr, 129)
        rpos = pos * 129
        assert_equal(np.argmax(rarr), rpos, err_msg="%r" % rarr)
        assert_equal(rarr[np.argmax(rarr)], val, err_msg="%r" % rarr)

        padd = np.repeat(np.min(arr), 513)
        rarr = np.concatenate((arr, padd))
        rpos = pos
        assert_equal(np.argmax(rarr), rpos, err_msg="%r" % rarr)
        assert_equal(rarr[np.argmax(rarr)], val, err_msg="%r" % rarr)


    def test_maximum_signed_integers(self):

        a = np.array([1, 2**7 - 1, -2**7], dtype=np.int8)
        assert_equal(np.argmax(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)

        a = np.array([1, 2**15 - 1, -2**15], dtype=np.int16)
        assert_equal(np.argmax(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)

        a = np.array([1, 2**31 - 1, -2**31], dtype=np.int32)
        assert_equal(np.argmax(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)

        a = np.array([1, 2**63 - 1, -2**63], dtype=np.int64)
        assert_equal(np.argmax(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)


class TestArgmin:
    usg_data = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 8),
        ([3, 3, 3, 3,  2,  2,  2,  2], 4),
        ([0, 1, 2, 3,  4,  5,  6,  7], 0),
        ([7, 6, 5, 4,  3,  2,  1,  0], 7)
    ]
    sg_data = usg_data + [
        ([1, 2, 3, 4, -4, -3, -2, -1], 4),
        ([1, 2, 3, 4, -1, -2, -3, -4], 7)
    ]
    darr = [(np.array(d[0], dtype=t), d[1]) for d, t in (
        itertools.product(usg_data, (
            np.uint8,
        ))
    )]
    darr = darr + [(np.array(d[0], dtype=t), d[1]) for d, t in (
        itertools.product(sg_data, (
            np.int8, np.int16, np.int32, np.int64, np.float32, np.float64
        ))
    )]
    darr = darr + [(np.array(d[0], dtype=t), d[1]) for d, t in (
        itertools.product((
            ([0, 1, 2, 3, np.nan], 4),
            ([0, 1, 2, np.nan, 3], 3),
            ([np.nan, 0, 1, 2, 3], 0),
            ([np.nan, 0, np.nan, 2, 3], 0),
            # To hit the tail of SIMD multi-level(x4, x1) inner loops
            # on variant SIMD widthes
            ([1] * (2*5-1) + [np.nan], 2*5-1),
            ([1] * (4*5-1) + [np.nan], 4*5-1),
            ([1] * (8*5-1) + [np.nan], 8*5-1),
            ([1] * (16*5-1) + [np.nan], 16*5-1),
            ([1] * (32*5-1) + [np.nan], 32*5-1)
        ), (
            np.float32, np.float64
        ))
    )]
    nan_arr = darr + [
    # RuntimeError: "min_values_cpu" not implemented for 'ComplexDouble'
    #    ([0, 1, 2, 3, complex(0, np.nan)], 4),
    #    ([0, 1, 2, 3, complex(np.nan, 0)], 4),
    #    ([0, 1, 2, complex(np.nan, 0), 3], 3),
    #    ([0, 1, 2, complex(0, np.nan), 3], 3),
    #    ([complex(0, np.nan), 0, 1, 2, 3], 0),
    #    ([complex(np.nan, np.nan), 0, 1, 2, 3], 0),
    #    ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, 1)], 0),
    #    ([complex(np.nan, np.nan), complex(np.nan, 2), complex(np.nan, 1)], 0),
    #    ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, np.nan)], 0),

    #    ([complex(0, 0), complex(0, 2), complex(0, 1)], 0),
    #    ([complex(1, 0), complex(0, 2), complex(0, 1)], 2),
    #    ([complex(1, 0), complex(0, 2), complex(1, 1)], 1),

        ([True, True, True, True, False], 4),
        ([True, True, True, False, True], 3),
        ([False, True, True, True, True], 0),
        ([False, True, False, True, True], 0),
    ]

    @pytest.mark.parametrize('data', nan_arr)
    def test_combinations(self, data):
        arr, pos = data
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning,
                       "invalid value encountered in reduce")
            min_val = np.min(arr)

        assert_equal(np.argmin(arr), pos, err_msg="%r" % arr)
        assert_equal(arr[np.argmin(arr)], min_val, err_msg="%r" % arr)

        # add padding to test SIMD loops
        rarr = np.repeat(arr, 129)
        rpos = pos * 129
        assert_equal(np.argmin(rarr), rpos, err_msg="%r" % rarr)
        assert_equal(rarr[np.argmin(rarr)], min_val, err_msg="%r" % rarr)

        padd = np.repeat(np.max(arr), 513)
        rarr = np.concatenate((arr, padd))
        rpos = pos
        assert_equal(np.argmin(rarr), rpos, err_msg="%r" % rarr)
        assert_equal(rarr[np.argmin(rarr)], min_val, err_msg="%r" % rarr)

    def test_minimum_signed_integers(self):

        a = np.array([1, -2**7, -2**7 + 1, 2**7 - 1], dtype=np.int8)
        assert_equal(np.argmin(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmin(a), 129)

        a = np.array([1, -2**15, -2**15 + 1, 2**15 - 1], dtype=np.int16)
        assert_equal(np.argmin(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmin(a), 129)

        a = np.array([1, -2**31, -2**31 + 1, 2**31 - 1], dtype=np.int32)
        assert_equal(np.argmin(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmin(a), 129)

        a = np.array([1, -2**63, -2**63 + 1, 2**63 - 1], dtype=np.int64)
        assert_equal(np.argmin(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmin(a), 129)


class TestMinMax:

    def test_scalar(self):
        assert_raises(np.AxisError, np.amax, 1, 1)
        assert_raises(np.AxisError, np.amin, 1, 1)

        pytest.xfail(reason='min/max/argmin/argmax on 0D arrays & axis')
        assert_equal(np.amax(1, axis=0), 1)
        assert_equal(np.amin(1, axis=0), 1)
        assert_equal(np.amax(1, axis=None), 1)
        assert_equal(np.amin(1, axis=None), 1)

    def test_axis(self):
        assert_raises(np.AxisError, np.amax, [1, 2, 3], 1000)
        assert_equal(np.amax([[1, 2, 3]], axis=1), 3)


class TestNewaxis:
    def test_basic(self):
        sk = np.array([0, -0.1, 0.1])
        res = 250*sk[:, np.newaxis]
        assert_almost_equal(res.ravel(), 250*sk)


class TestClip:
    def _check_range(self, x, cmin, cmax):
        assert_(np.all(x >= cmin))
        assert_(np.all(x <= cmax))

    def _clip_type(self, type_group, array_max,
                   clip_min, clip_max, inplace=False,
                   expected_min=None, expected_max=None):
        if expected_min is None:
            expected_min = clip_min
        if expected_max is None:
            expected_max = clip_max

        for T in np.sctypes[type_group]:
            if sys.byteorder == 'little':
                byte_orders = ['=', '>']
            else:
                byte_orders = ['<', '=']

            for byteorder in byte_orders:
                dtype = np.dtype(T).newbyteorder(byteorder)

                x = (np.random.random(1000) * array_max).astype(dtype)
                if inplace:
                    # The tests that call us pass clip_min and clip_max that
                    # might not fit in the destination dtype. They were written
                    # assuming the previous unsafe casting, which now must be
                    # passed explicitly to avoid a warning.
                    x.clip(clip_min, clip_max, x, casting='unsafe')
                else:
                    x = x.clip(clip_min, clip_max)
                    byteorder = '='

                if x.dtype.byteorder == '|':
                    byteorder = '|'
                assert_equal(x.dtype.byteorder, byteorder)
                self._check_range(x, expected_min, expected_max)
        return x

    @pytest.mark.skip(reason="endianness")
    def test_basic(self):
        for inplace in [False, True]:
            self._clip_type(
                'float', 1024, -12.8, 100.2, inplace=inplace)
            self._clip_type(
                'float', 1024, 0, 0, inplace=inplace)

            self._clip_type(
                'int', 1024, -120, 100, inplace=inplace)
            self._clip_type(
                'int', 1024, 0, 0, inplace=inplace)

            self._clip_type(
                'uint', 1024, 0, 0, inplace=inplace)
            self._clip_type(
                'uint', 1024, -120, 100, inplace=inplace, expected_min=0)

    def test_max_or_min(self):
        val = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        x = val.clip(3)
        assert_(np.all(x >= 3))
        x = val.clip(min=3)
        assert_(np.all(x >= 3))
        x = val.clip(max=4)
        assert_(np.all(x <= 4))

    def test_nan(self):
        input_arr = np.array([-2., np.nan, 0.5, 3., 0.25, np.nan])
        result = input_arr.clip(-1, 1)
        expected = np.array([-1., np.nan, 0.5, 1., 0.25, np.nan])
        assert_array_equal(result, expected)


@pytest.mark.xfail(reason='TODO')
class TestCompress:
    def test_axis(self):
        tgt = [[5, 6, 7, 8, 9]]
        arr = np.arange(10).reshape(2, 5)
        out = np.compress([0, 1], arr, axis=0)
        assert_equal(out, tgt)

        tgt = [[1, 3], [6, 8]]
        out = np.compress([0, 1, 0, 1, 0], arr, axis=1)
        assert_equal(out, tgt)

    def test_truncate(self):
        tgt = [[1], [6]]
        arr = np.arange(10).reshape(2, 5)
        out = np.compress([0, 1], arr, axis=1)
        assert_equal(out, tgt)

    def test_flatten(self):
        arr = np.arange(10).reshape(2, 5)
        out = np.compress([0, 1], arr)
        assert_equal(out, 1)


@pytest.mark.xfail(reason='TODO')
class TestPutmask:
    def tst_basic(self, x, T, mask, val):
        np.putmask(x, mask, val)
        assert_equal(x[mask], np.array(val, T))

    def test_ip_types(self):
        unchecked_types = [bytes, str, np.void]

        x = np.random.random(1000)*100
        mask = x < 40

        for val in [-100, 0, 15]:
            for types in np.sctypes.values():
                for T in types:
                    if T not in unchecked_types:
                        if val < 0 and np.dtype(T).kind == "u":
                            val = np.iinfo(T).max - 99
                        self.tst_basic(x.copy().astype(T), T, mask, val)

            # Also test string of a length which uses an untypical length
            dt = np.dtype("S3")
            self.tst_basic(x.astype(dt), dt.type, mask, dt.type(val)[:3])

    def test_mask_size(self):
        assert_raises(ValueError, np.putmask, np.array([1, 2, 3]), [True], 5)

    @pytest.mark.parametrize('dtype', ('>i4', '<i4'))
    def test_byteorder(self, dtype):
        x = np.array([1, 2, 3], dtype)
        np.putmask(x, [True, False, True], -1)
        assert_array_equal(x, [-1, 2, -1])

    def test_record_array(self):
        # Note mixed byteorder.
        rec = np.array([(-5, 2.0, 3.0), (5.0, 4.0, 3.0)],
                      dtype=[('x', '<f8'), ('y', '>f8'), ('z', '<f8')])
        np.putmask(rec['x'], [True, False], 10)
        assert_array_equal(rec['x'], [10, 5])
        assert_array_equal(rec['y'], [2, 4])
        assert_array_equal(rec['z'], [3, 3])
        np.putmask(rec['y'], [True, False], 11)
        assert_array_equal(rec['x'], [10, 5])
        assert_array_equal(rec['y'], [11, 4])
        assert_array_equal(rec['z'], [3, 3])

    def test_overlaps(self):
        # gh-6272 check overlap
        x = np.array([True, False, True, False])
        np.putmask(x[1:4], [True, True, True], x[:3])
        assert_equal(x, np.array([True, True, False, True]))

        x = np.array([True, False, True, False])
        np.putmask(x[1:4], x[:3], [True, False, True])
        assert_equal(x, np.array([True, True, True, True]))

    def test_writeable(self):
        a = np.arange(5)
        a.flags.writeable = False

        with pytest.raises(ValueError):
            np.putmask(a, a >= 2, 3)

    def test_kwargs(self):
        x = np.array([0, 0])
        np.putmask(x, [0, 1], [-1, -2])
        assert_array_equal(x, [0, -2])

        x = np.array([0, 0])
        np.putmask(x, mask=[0, 1], values=[-1, -2])
        assert_array_equal(x, [0, -2])

        x = np.array([0, 0])
        np.putmask(x, values=[-1, -2],  mask=[0, 1])
        assert_array_equal(x, [0, -2])

        with pytest.raises(TypeError):
            np.putmask(a=x, values=[-1, -2],  mask=[0, 1])


class TestTake:
    def tst_basic(self, x):
        ind = list(range(x.shape[0]))
        assert_array_equal(np.take(x, ind, axis=0), x)

    def test_ip_types(self):
        x = np.random.random(24)*100
        x = np.reshape(x, (2, 3, 4))
        for types in np.sctypes.values():
            for T in types:
                self.tst_basic(x.copy().astype(T))

    def test_raise(self):
        x = np.random.random(24)*100
        x = np.reshape(x, (2, 3, 4))
        assert_raises(IndexError, np.take, x, [0, 1, 2], axis=0)
        assert_raises(IndexError, np.take, x, [-3], axis=0)
        assert_array_equal(np.take(x, [-1], axis=0)[0], x[1])

    @pytest.mark.xfail(reason="XXX: take(..., mode='clip')")
    def test_clip(self):
        x = np.random.random(24)*100
        x = np.reshape(x, (2, 3, 4))
        assert_array_equal(np.take(x, [-1], axis=0, mode='clip')[0], x[0])
        assert_array_equal(np.take(x, [2], axis=0, mode='clip')[0], x[1])

    @pytest.mark.xfail(reason="XXX: take(..., mode='wrap')")
    def test_wrap(self):
        x = np.random.random(24)*100
        x = np.reshape(x, (2, 3, 4))
        assert_array_equal(np.take(x, [-1], axis=0, mode='wrap')[0], x[1])
        assert_array_equal(np.take(x, [2], axis=0, mode='wrap')[0], x[0])
        assert_array_equal(np.take(x, [3], axis=0, mode='wrap')[0], x[1])

    @pytest.mark.xfail(reason="XXX: take(mode='wrap')")
    def test_out_overlap(self):
        # gh-6272 check overlap on out
        x = np.arange(5)
        y = np.take(x, [1, 2, 3], out=x[2:5], mode='wrap')
        assert_equal(y, np.array([1, 2, 3]))

    @pytest.mark.parametrize('shape', [(1, 2), (1,), ()])
    def test_ret_is_out(self, shape):
        # 0d arrays should not be an exception to this rule
        x = np.arange(5)
        inds = np.zeros(shape, dtype=np.intp)
        out = np.zeros(shape, dtype=x.dtype)
        ret = np.take(x, inds, out=out)
        assert ret is out


@pytest.mark.xfail(reason='TODO')
class TestLexsort:
    @pytest.mark.parametrize('dtype',[
        np.uint8,
        np.int8, np.int16, np.int32, np.int64,
        np.float16, np.float32, np.float64
    ])
    def test_basic(self, dtype):
        a = np.array([1, 2, 1, 3, 1, 5], dtype=dtype)
        b = np.array([0, 4, 5, 6, 2, 3], dtype=dtype)
        idx = np.lexsort((b, a))
        expected_idx = np.array([0, 4, 2, 1, 3, 5])
        assert_array_equal(idx, expected_idx)
        assert_array_equal(a[idx], np.sort(a))

    def test_mixed(self):
        a = np.array([1, 2, 1, 3, 1, 5])
        b = np.array([0, 4, 5, 6, 2, 3], dtype='datetime64[D]')

        idx = np.lexsort((b, a))
        expected_idx = np.array([0, 4, 2, 1, 3, 5])
        assert_array_equal(idx, expected_idx)

    def test_datetime(self):
        a = np.array([0,0,0], dtype='datetime64[D]')
        b = np.array([2,1,0], dtype='datetime64[D]')
        idx = np.lexsort((b, a))
        expected_idx = np.array([2, 1, 0])
        assert_array_equal(idx, expected_idx)

        a = np.array([0,0,0], dtype='timedelta64[D]')
        b = np.array([2,1,0], dtype='timedelta64[D]')
        idx = np.lexsort((b, a))
        expected_idx = np.array([2, 1, 0])
        assert_array_equal(idx, expected_idx)

    def test_object(self):  # gh-6312
        a = np.random.choice(10, 1000)
        b = np.random.choice(['abc', 'xy', 'wz', 'efghi', 'qwst', 'x'], 1000)

        for u in a, b:
            left = np.lexsort((u.astype('O'),))
            right = np.argsort(u, kind='mergesort')
            assert_array_equal(left, right)

        for u, v in (a, b), (b, a):
            idx = np.lexsort((u, v))
            assert_array_equal(idx, np.lexsort((u.astype('O'), v)))
            assert_array_equal(idx, np.lexsort((u, v.astype('O'))))
            u, v = np.array(u, dtype='object'), np.array(v, dtype='object')
            assert_array_equal(idx, np.lexsort((u, v)))

    def test_invalid_axis(self): # gh-7528
        x = np.linspace(0., 1., 42*3).reshape(42, 3)
        assert_raises(np.AxisError, np.lexsort, x, axis=2)


@pytest.mark.skip(reason='dont worry about IO')
class TestIO:
    """Test tofile, fromfile, tobytes, and fromstring"""

    @pytest.fixture()
    def x(self):
        shape = (2, 4, 3)
        rand = np.random.random
        x = rand(shape) + rand(shape).astype(complex) * 1j
        x[0, :, 1] = [np.nan, np.inf, -np.inf, np.nan]
        return x

    @pytest.fixture(params=["string", "path_obj"])
    def tmp_filename(self, tmp_path, request):
        # This fixture covers two cases:
        # one where the filename is a string and
        # another where it is a pathlib object
        filename = tmp_path / "file"
        if request.param == "string":
            filename = str(filename)
        yield filename

    def test_nofile(self):
        # this should probably be supported as a file
        # but for now test for proper errors
        b = io.BytesIO()
        assert_raises(OSError, np.fromfile, b, np.uint8, 80)
        d = np.ones(7)
        assert_raises(OSError, lambda x: x.tofile(b), d)

    def test_bool_fromstring(self):
        v = np.array([True, False, True, False], dtype=np.bool_)
        y = np.fromstring('1 0 -2.3 0.0', sep=' ', dtype=np.bool_)
        assert_array_equal(v, y)

    def test_uint64_fromstring(self):
        d = np.fromstring("9923372036854775807 104783749223640",
                          dtype=np.uint64, sep=' ')
        e = np.array([9923372036854775807, 104783749223640], dtype=np.uint64)
        assert_array_equal(d, e)

    def test_int64_fromstring(self):
        d = np.fromstring("-25041670086757 104783749223640",
                          dtype=np.int64, sep=' ')
        e = np.array([-25041670086757, 104783749223640], dtype=np.int64)
        assert_array_equal(d, e)

    def test_fromstring_count0(self):
        d = np.fromstring("1,2", sep=",", dtype=np.int64, count=0)
        assert d.shape == (0,)

    def test_empty_files_text(self, tmp_filename):
        with open(tmp_filename, 'w') as f:
            pass
        y = np.fromfile(tmp_filename)
        assert_(y.size == 0, "Array not empty")

    def test_empty_files_binary(self, tmp_filename):
        with open(tmp_filename, 'wb') as f:
            pass
        y = np.fromfile(tmp_filename, sep=" ")
        assert_(y.size == 0, "Array not empty")

    def test_roundtrip_file(self, x, tmp_filename):
        with open(tmp_filename, 'wb') as f:
            x.tofile(f)
        # NB. doesn't work with flush+seek, due to use of C stdio
        with open(tmp_filename, 'rb') as f:
            y = np.fromfile(f, dtype=x.dtype)
        assert_array_equal(y, x.flat)

    def test_roundtrip(self, x, tmp_filename):
        x.tofile(tmp_filename)
        y = np.fromfile(tmp_filename, dtype=x.dtype)
        assert_array_equal(y, x.flat)

    def test_roundtrip_dump_pathlib(self, x, tmp_filename):
        p = pathlib.Path(tmp_filename)
        x.dump(p)
        y = np.load(p, allow_pickle=True)
        assert_array_equal(y, x)

    def test_roundtrip_binary_str(self, x):
        s = x.tobytes()
        y = np.frombuffer(s, dtype=x.dtype)
        assert_array_equal(y, x.flat)

        s = x.tobytes('F')
        y = np.frombuffer(s, dtype=x.dtype)
        assert_array_equal(y, x.flatten('F'))

    def test_roundtrip_str(self, x):
        x = x.real.ravel()
        s = "@".join(map(str, x))
        y = np.fromstring(s, sep="@")
        # NB. str imbues less precision
        nan_mask = ~np.isfinite(x)
        assert_array_equal(x[nan_mask], y[nan_mask])
        assert_array_almost_equal(x[~nan_mask], y[~nan_mask], decimal=5)

    def test_roundtrip_repr(self, x):
        x = x.real.ravel()
        s = "@".join(map(repr, x))
        y = np.fromstring(s, sep="@")
        assert_array_equal(x, y)

    def test_unseekable_fromfile(self, x, tmp_filename):
        # gh-6246
        x.tofile(tmp_filename)

        def fail(*args, **kwargs):
            raise OSError('Can not tell or seek')

        with io.open(tmp_filename, 'rb', buffering=0) as f:
            f.seek = fail
            f.tell = fail
            assert_raises(OSError, np.fromfile, f, dtype=x.dtype)

    def test_io_open_unbuffered_fromfile(self, x, tmp_filename):
        # gh-6632
        x.tofile(tmp_filename)
        with io.open(tmp_filename, 'rb', buffering=0) as f:
            y = np.fromfile(f, dtype=x.dtype)
            assert_array_equal(y, x.flat)

    def test_largish_file(self, tmp_filename):
        # check the fallocate path on files > 16MB
        d = np.zeros(4 * 1024 ** 2)
        d.tofile(tmp_filename)
        assert_equal(os.path.getsize(tmp_filename), d.nbytes)
        assert_array_equal(d, np.fromfile(tmp_filename))
        # check offset
        with open(tmp_filename, "r+b") as f:
            f.seek(d.nbytes)
            d.tofile(f)
            assert_equal(os.path.getsize(tmp_filename), d.nbytes * 2)
        # check append mode (gh-8329)
        open(tmp_filename, "w").close()  # delete file contents
        with open(tmp_filename, "ab") as f:
            d.tofile(f)
        assert_array_equal(d, np.fromfile(tmp_filename))
        with open(tmp_filename, "ab") as f:
            d.tofile(f)
        assert_equal(os.path.getsize(tmp_filename), d.nbytes * 2)

    def test_io_open_buffered_fromfile(self, x, tmp_filename):
        # gh-6632
        x.tofile(tmp_filename)
        with io.open(tmp_filename, 'rb', buffering=-1) as f:
            y = np.fromfile(f, dtype=x.dtype)
        assert_array_equal(y, x.flat)

    def test_file_position_after_fromfile(self, tmp_filename):
        # gh-4118
        sizes = [io.DEFAULT_BUFFER_SIZE//8,
                 io.DEFAULT_BUFFER_SIZE,
                 io.DEFAULT_BUFFER_SIZE*8]

        for size in sizes:
            with open(tmp_filename, 'wb') as f:
                f.seek(size-1)
                f.write(b'\0')

            for mode in ['rb', 'r+b']:
                err_msg = "%d %s" % (size, mode)

                with open(tmp_filename, mode) as f:
                    f.read(2)
                    np.fromfile(f, dtype=np.float64, count=1)
                    pos = f.tell()
                assert_equal(pos, 10, err_msg=err_msg)

    def test_file_position_after_tofile(self, tmp_filename):
        # gh-4118
        sizes = [io.DEFAULT_BUFFER_SIZE//8,
                 io.DEFAULT_BUFFER_SIZE,
                 io.DEFAULT_BUFFER_SIZE*8]

        for size in sizes:
            err_msg = "%d" % (size,)

            with open(tmp_filename, 'wb') as f:
                f.seek(size-1)
                f.write(b'\0')
                f.seek(10)
                f.write(b'12')
                np.array([0], dtype=np.float64).tofile(f)
                pos = f.tell()
            assert_equal(pos, 10 + 2 + 8, err_msg=err_msg)

            with open(tmp_filename, 'r+b') as f:
                f.read(2)
                f.seek(0, 1)  # seek between read&write required by ANSI C
                np.array([0], dtype=np.float64).tofile(f)
                pos = f.tell()
            assert_equal(pos, 10, err_msg=err_msg)

    def test_load_object_array_fromfile(self, tmp_filename):
        # gh-12300
        with open(tmp_filename, 'w') as f:
            # Ensure we have a file with consistent contents
            pass

        with open(tmp_filename, 'rb') as f:
            assert_raises_regex(ValueError, "Cannot read into object array",
                                np.fromfile, f, dtype=object)

        assert_raises_regex(ValueError, "Cannot read into object array",
                            np.fromfile, tmp_filename, dtype=object)

    def test_fromfile_offset(self, x, tmp_filename):
        with open(tmp_filename, 'wb') as f:
            x.tofile(f)

        with open(tmp_filename, 'rb') as f:
            y = np.fromfile(f, dtype=x.dtype, offset=0)
            assert_array_equal(y, x.flat)

        with open(tmp_filename, 'rb') as f:
            count_items = len(x.flat) // 8
            offset_items = len(x.flat) // 4
            offset_bytes = x.dtype.itemsize * offset_items
            y = np.fromfile(
                f, dtype=x.dtype, count=count_items, offset=offset_bytes
            )
            assert_array_equal(
                y, x.flat[offset_items:offset_items+count_items]
            )

            # subsequent seeks should stack
            offset_bytes = x.dtype.itemsize
            z = np.fromfile(f, dtype=x.dtype, offset=offset_bytes)
            assert_array_equal(z, x.flat[offset_items+count_items+1:])

        with open(tmp_filename, 'wb') as f:
            x.tofile(f, sep=",")

        with open(tmp_filename, 'rb') as f:
            assert_raises_regex(
                    TypeError,
                    "'offset' argument only permitted for binary files",
                    np.fromfile, tmp_filename, dtype=x.dtype,
                    sep=",", offset=1)

    @pytest.mark.skipif(IS_PYPY, reason="bug in PyPy's PyNumber_AsSsize_t")
    def test_fromfile_bad_dup(self, x, tmp_filename):
        def dup_str(fd):
            return 'abc'

        def dup_bigint(fd):
            return 2**68

        old_dup = os.dup
        try:
            with open(tmp_filename, 'wb') as f:
                x.tofile(f)
                for dup, exc in ((dup_str, TypeError), (dup_bigint, OSError)):
                    os.dup = dup
                    assert_raises(exc, np.fromfile, f)
        finally:
            os.dup = old_dup

    def _check_from(self, s, value, filename, **kw):
        if 'sep' not in kw:
            y = np.frombuffer(s, **kw)
        else:
            y = np.fromstring(s, **kw)
        assert_array_equal(y, value)

        with open(filename, 'wb') as f:
            f.write(s)
        y = np.fromfile(filename, **kw)
        assert_array_equal(y, value)

    @pytest.fixture(params=["period", "comma"])
    def decimal_sep_localization(self, request):
        """
        Including this fixture in a test will automatically
        execute it with both types of decimal separator.

        So::

            def test_decimal(decimal_sep_localization):
                pass

        is equivalent to the following two tests::

            def test_decimal_period_separator():
                pass

            def test_decimal_comma_separator():
                with CommaDecimalPointLocale():
                    pass
        """
        if request.param == "period":
            yield
        elif request.param == "comma":
            with CommaDecimalPointLocale():
                yield
        else:
            assert False, request.param

    def test_nan(self, tmp_filename, decimal_sep_localization):
        self._check_from(
            b"nan +nan -nan NaN nan(foo) +NaN(BAR) -NAN(q_u_u_x_)",
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            tmp_filename,
            sep=' ')

    def test_inf(self, tmp_filename, decimal_sep_localization):
        self._check_from(
            b"inf +inf -inf infinity -Infinity iNfInItY -inF",
            [np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf],
            tmp_filename,
            sep=' ')

    def test_numbers(self, tmp_filename, decimal_sep_localization):
        self._check_from(
            b"1.234 -1.234 .3 .3e55 -123133.1231e+133",
            [1.234, -1.234, .3, .3e55, -123133.1231e+133],
            tmp_filename,
            sep=' ')

    def test_binary(self, tmp_filename):
        self._check_from(
            b'\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@',
            np.array([1, 2, 3, 4]),
            tmp_filename,
            dtype='<f4')

    @pytest.mark.slow  # takes > 1 minute on mechanical hard drive
    def test_big_binary(self):
        """Test workarounds for 32-bit limit for MSVC fwrite, fseek, and ftell

        These normally would hang doing something like this.
        See : https://github.com/numpy/numpy/issues/2256
        """
        if sys.platform != 'win32' or '[GCC ' in sys.version:
            return
        try:
            # before workarounds, only up to 2**32-1 worked
            fourgbplus = 2**32 + 2**16
            testbytes = np.arange(8, dtype=np.int8)
            n = len(testbytes)
            flike = tempfile.NamedTemporaryFile()
            f = flike.file
            np.tile(testbytes, fourgbplus // testbytes.nbytes).tofile(f)
            flike.seek(0)
            a = np.fromfile(f, dtype=np.int8)
            flike.close()
            assert_(len(a) == fourgbplus)
            # check only start and end for speed:
            assert_((a[:n] == testbytes).all())
            assert_((a[-n:] == testbytes).all())
        except (MemoryError, ValueError):
            pass

    def test_string(self, tmp_filename):
        self._check_from(b'1,2,3,4', [1., 2., 3., 4.], tmp_filename, sep=',')

    def test_counted_string(self, tmp_filename, decimal_sep_localization):
        self._check_from(
            b'1,2,3,4', [1., 2., 3., 4.], tmp_filename, count=4, sep=',')
        self._check_from(
            b'1,2,3,4', [1., 2., 3.], tmp_filename, count=3, sep=',')
        self._check_from(
            b'1,2,3,4', [1., 2., 3., 4.], tmp_filename, count=-1, sep=',')

    def test_string_with_ws(self, tmp_filename):
        self._check_from(
            b'1 2  3     4   ', [1, 2, 3, 4], tmp_filename, dtype=int, sep=' ')

    def test_counted_string_with_ws(self, tmp_filename):
        self._check_from(
            b'1 2  3     4   ', [1, 2, 3], tmp_filename, count=3, dtype=int,
            sep=' ')

    def test_ascii(self, tmp_filename, decimal_sep_localization):
        self._check_from(
            b'1 , 2 , 3 , 4', [1., 2., 3., 4.], tmp_filename, sep=',')
        self._check_from(
            b'1,2,3,4', [1., 2., 3., 4.], tmp_filename, dtype=float, sep=',')

    def test_malformed(self, tmp_filename, decimal_sep_localization):
        with assert_warns(DeprecationWarning):
            self._check_from(
                b'1.234 1,234', [1.234, 1.], tmp_filename, sep=' ')

    def test_long_sep(self, tmp_filename):
        self._check_from(
            b'1_x_3_x_4_x_5', [1, 3, 4, 5], tmp_filename, sep='_x_')

    def test_dtype(self, tmp_filename):
        v = np.array([1, 2, 3, 4], dtype=np.int_)
        self._check_from(b'1,2,3,4', v, tmp_filename, sep=',', dtype=np.int_)

    def test_dtype_bool(self, tmp_filename):
        # can't use _check_from because fromstring can't handle True/False
        v = np.array([True, False, True, False], dtype=np.bool_)
        s = b'1,0,-2.3,0'
        with open(tmp_filename, 'wb') as f:
            f.write(s)
        y = np.fromfile(tmp_filename, sep=',', dtype=np.bool_)
        assert_(y.dtype == '?')
        assert_array_equal(y, v)

    def test_tofile_sep(self, tmp_filename, decimal_sep_localization):
        x = np.array([1.51, 2, 3.51, 4], dtype=float)
        with open(tmp_filename, 'w') as f:
            x.tofile(f, sep=',')
        with open(tmp_filename, 'r') as f:
            s = f.read()
        #assert_equal(s, '1.51,2.0,3.51,4.0')
        y = np.array([float(p) for p in s.split(',')])
        assert_array_equal(x,y)

    def test_tofile_format(self, tmp_filename, decimal_sep_localization):
        x = np.array([1.51, 2, 3.51, 4], dtype=float)
        with open(tmp_filename, 'w') as f:
            x.tofile(f, sep=',', format='%.2f')
        with open(tmp_filename, 'r') as f:
            s = f.read()
        assert_equal(s, '1.51,2.00,3.51,4.00')

    def test_tofile_cleanup(self, tmp_filename):
        x = np.zeros((10), dtype=object)
        with open(tmp_filename, 'wb') as f:
            assert_raises(OSError, lambda: x.tofile(f, sep=''))
        # Dup-ed file handle should be closed or remove will fail on Windows OS
        os.remove(tmp_filename)

        # Also make sure that we close the Python handle
        assert_raises(OSError, lambda: x.tofile(tmp_filename))
        os.remove(tmp_filename)

    def test_fromfile_subarray_binary(self, tmp_filename):
        # Test subarray dtypes which are absorbed into the shape
        x = np.arange(24, dtype="i4").reshape(2, 3, 4)
        x.tofile(tmp_filename)
        res = np.fromfile(tmp_filename, dtype="(3,4)i4")
        assert_array_equal(x, res)

        x_str = x.tobytes()
        with assert_warns(DeprecationWarning):
            # binary fromstring is deprecated
            res = np.fromstring(x_str, dtype="(3,4)i4")
            assert_array_equal(x, res)

    def test_parsing_subarray_unsupported(self, tmp_filename):
        # We currently do not support parsing subarray dtypes
        data = "12,42,13," * 50
        with pytest.raises(ValueError):
            expected = np.fromstring(data, dtype="(3,)i", sep=",")

        with open(tmp_filename, "w") as f:
            f.write(data)

        with pytest.raises(ValueError):
            np.fromfile(tmp_filename, dtype="(3,)i", sep=",")

    def test_read_shorter_than_count_subarray(self, tmp_filename):
        # Test that requesting more values does not cause any problems
        # in conjunction with subarray dimensions being absorbed into the
        # array dimension.
        expected = np.arange(511 * 10, dtype="i").reshape(-1, 10)

        binary = expected.tobytes()
        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                np.fromstring(binary, dtype="(10,)i", count=10000)

        expected.tofile(tmp_filename)
        res = np.fromfile(tmp_filename, dtype="(10,)i", count=10000)
        assert_array_equal(res, expected)


@pytest.mark.xfail(reason='TODO')
class TestFromBuffer:
    @pytest.mark.parametrize('byteorder', ['<', '>'])
    @pytest.mark.parametrize('dtype', [float, int, complex])
    def test_basic(self, byteorder, dtype):
        dt = np.dtype(dtype).newbyteorder(byteorder)
        x = (np.random.random((4, 7)) * 5).astype(dt)
        buf = x.tobytes()
        assert_array_equal(np.frombuffer(buf, dtype=dt), x.flat)

    @pytest.mark.parametrize("obj", [np.arange(10), b"12345678"])
    def test_array_base(self, obj):
        # Objects (including NumPy arrays), which do not use the
        # `release_buffer` slot should be directly used as a base object.
        # See also gh-21612
        new = np.frombuffer(obj)
        assert new.base is obj

    def test_empty(self):
        assert_array_equal(np.frombuffer(b''), np.array([]))

    @pytest.mark.skipif(IS_PYPY,
            reason="PyPy's memoryview currently does not track exports. See: "
                   "https://foss.heptapod.net/pypy/pypy/-/issues/3724")
    def test_mmap_close(self):
        # The old buffer protocol was not safe for some things that the new
        # one is.  But `frombuffer` always used the old one for a long time.
        # Checks that it is safe with the new one (using memoryviews)
        with tempfile.TemporaryFile(mode='wb') as tmp:
            tmp.write(b"asdf")
            tmp.flush()
            mm = mmap.mmap(tmp.fileno(), 0)
            arr = np.frombuffer(mm, dtype=np.uint8)
            with pytest.raises(BufferError):
                mm.close()  # cannot close while array uses the buffer
            del arr
            mm.close()


@pytest.mark.xfail(reason='TODO')
class TestFlat:
    def setup_method(self):
        a0 = np.arange(20.0)
        a = a0.reshape(4, 5)
        a0.shape = (4, 5)
        a.flags.writeable = False
        self.a = a
        self.b = a[::2, ::2]
        self.a0 = a0
        self.b0 = a0[::2, ::2]

    def test_contiguous(self):
        testpassed = False
        try:
            self.a.flat[12] = 100.0
        except ValueError:
            testpassed = True
        assert_(testpassed)
        assert_(self.a.flat[12] == 12.0)

    def test_discontiguous(self):
        testpassed = False
        try:
            self.b.flat[4] = 100.0
        except ValueError:
            testpassed = True
        assert_(testpassed)
        assert_(self.b.flat[4] == 12.0)

    def test___array__(self):
        c = self.a.flat.__array__()
        d = self.b.flat.__array__()
        e = self.a0.flat.__array__()
        f = self.b0.flat.__array__()

        assert_(c.flags.writeable is False)
        assert_(d.flags.writeable is False)
        assert_(e.flags.writeable is True)
        assert_(f.flags.writeable is False)
        assert_(c.flags.writebackifcopy is False)
        assert_(d.flags.writebackifcopy is False)
        assert_(e.flags.writebackifcopy is False)
        assert_(f.flags.writebackifcopy is False)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_refcount(self):
        # includes regression test for reference count error gh-13165
        inds = [np.intp(0), np.array([True]*self.a.size), np.array([0]), None]
        indtype = np.dtype(np.intp)
        rc_indtype = sys.getrefcount(indtype)
        for ind in inds:
            rc_ind = sys.getrefcount(ind)
            for _ in range(100):
                try:
                    self.a.flat[ind]
                except IndexError:
                    pass
            assert_(abs(sys.getrefcount(ind) - rc_ind) < 50)
            assert_(abs(sys.getrefcount(indtype) - rc_indtype) < 50)

    def test_index_getset(self):
        it = np.arange(10).reshape(2, 1, 5).flat
        with pytest.raises(AttributeError):
            it.index = 10

        for _ in it:
            pass
        # Check the value of `.index` is updated correctly (see also gh-19153)
        # If the type was incorrect, this would show up on big-endian machines
        assert it.index == it.base.size


class TestResize:

    @_no_tracing
    def test_basic(self):
        x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if IS_PYPY:
            x.resize((5, 5), refcheck=False)
        else:
            x.resize((5, 5))
        assert_array_equal(x.ravel()[:9],
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).ravel())
        assert_array_equal(x[9:].ravel(), 0)

    @pytest.mark.skip(reason="how to find if someone is refencing an array")
    def test_check_reference(self):
        x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y = x
        assert_raises(ValueError, x.resize, (5, 1))
        del y  # avoid pyflakes unused variable warning.

    @_no_tracing
    def test_int_shape(self):
        x = np.eye(3)
        if IS_PYPY:
            x.resize(3, refcheck=False)
        else:
            x.resize(3)
        assert_array_equal(x, np.eye(3)[0,:])

    def test_none_shape(self):
        x = np.eye(3)
        x.resize(None)
        assert_array_equal(x, np.eye(3))
        x.resize()
        assert_array_equal(x, np.eye(3))

    def test_0d_shape(self):
        # to it multiple times to test it does not break alloc cache gh-9216
        for i in range(10):
            x = np.empty((1,))
            x.resize(())
            assert_equal(x.shape, ())
            assert_equal(x.size, 1)
            x = np.empty(())
            x.resize((1,))
            assert_equal(x.shape, (1,))
            assert_equal(x.size, 1)

    def test_invalid_arguments(self):
        assert_raises(TypeError, np.eye(3).resize, 'hi')
        assert_raises(ValueError, np.eye(3).resize, -1)
        assert_raises(TypeError, np.eye(3).resize, order=1)
        assert_raises((NotImplementedError, TypeError), np.eye(3).resize, refcheck='hi')

    @_no_tracing
    def test_freeform_shape(self):
        x = np.eye(3)
        if IS_PYPY:
            x.resize(3, 2, 1, refcheck=False)
        else:
            x.resize(3, 2, 1)
        assert_(x.shape == (3, 2, 1))

    @_no_tracing
    def test_zeros_appended(self):
        x = np.eye(3)
        if IS_PYPY:
            x.resize(2, 3, 3, refcheck=False)
        else:
            x.resize(2, 3, 3)
        assert_array_equal(x[0], np.eye(3))
        assert_array_equal(x[1], np.zeros((3, 3)))

    def test_empty_view(self):
        # check that sizes containing a zero don't trigger a reallocate for
        # already empty arrays
        x = np.zeros((10, 0), int)
        x_view = x[...]
        x_view.resize((0, 10))
        x_view.resize((0, 100))

    @pytest.mark.skip(reason="ignore weakrefs for ndarray.resize")
    def test_check_weakref(self):
        x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        xref = weakref.ref(x)
        assert_raises(ValueError, x.resize, (5, 1))
        del xref  # avoid pyflakes unused variable warning.


def _mean(a, **args):
    return a.mean(**args)


def _var(a, **args):
    return a.var(**args)


def _std(a, **args):
    return a.std(**args)


class TestStats:

    funcs = [_mean, _var, _std]

    def setup_method(self):
        np.random.seed(3)
        self.rmat = np.random.random((4, 5))
        self.cmat = self.rmat + 1j * self.rmat

    def test_python_type(self):
        for x in (np.float16(1.), 1, 1., 1+0j):
            assert_equal(np.mean([x]), 1.)
            assert_equal(np.std([x]), 0.)
            assert_equal(np.var([x]), 0.)

    def test_keepdims(self):
        mat = np.eye(3)
        for f in self.funcs:
            for axis in [0, 1]:
                res = f(mat, axis=axis, keepdims=True)
                assert_(res.ndim == mat.ndim)
                assert_(res.shape[axis] == 1)
            for axis in [None]:
                res = f(mat, axis=axis, keepdims=True)
                assert_(res.shape == (1, 1))

    def test_out(self):
        mat = np.eye(3)
        for f in self.funcs:
            out = np.zeros(3)
            tgt = f(mat, axis=1)
            res = f(mat, axis=1, out=out)
            assert_almost_equal(res, out)
            assert_almost_equal(res, tgt)
        out = np.empty(2)
        assert_raises(ValueError, f, mat, axis=1, out=out)
        out = np.empty((2, 2))
        assert_raises(ValueError, f, mat, axis=1, out=out)

    def test_dtype_from_input(self):

        icodes = np.typecodes['AllInteger']
        fcodes = np.typecodes['AllFloat']

        # integer types
        for f in self.funcs:
            for c in icodes:
                mat = np.eye(3, dtype=c)
                tgt = np.float64
                res = f(mat, axis=1).dtype.type
                assert_(res is tgt)
                # scalar case
                res = f(mat, axis=None).dtype.type
                assert_(res is tgt)

        # mean for float types
        for f in [_mean]:
            for c in fcodes:
                mat = np.eye(3, dtype=c)
                tgt = mat.dtype.type
                res = f(mat, axis=1).dtype.type
                assert_(res is tgt)
                # scalar case
                res = f(mat, axis=None).dtype.type
                assert_(res is tgt)

        # var, std for float types
        for f in [_var, _std]:
            for c in fcodes:
                mat = np.eye(3, dtype=c)
                # deal with complex types
                tgt = mat.real.dtype.type
                res = f(mat, axis=1).dtype.type
                assert_(res is tgt)
                # scalar case
                res = f(mat, axis=None).dtype.type
                assert_(res is tgt)

    def test_dtype_from_dtype(self):
        mat = np.eye(3)

        # stats for integer types
        # FIXME:
        # this needs definition as there are lots places along the line
        # where type casting may take place.

        # for f in self.funcs:
        #    for c in np.typecodes['AllInteger']:
        #        tgt = np.dtype(c).type
        #        res = f(mat, axis=1, dtype=c).dtype.type
        #        assert_(res is tgt)
        #        # scalar case
        #        res = f(mat, axis=None, dtype=c).dtype.type
        #        assert_(res is tgt)

        # stats for float types
        for f in self.funcs:
            for c in np.typecodes['AllFloat']:
                tgt = np.dtype(c).type
                res = f(mat, axis=1, dtype=c).dtype.type
                assert_(res is tgt)
                # scalar case
                res = f(mat, axis=None, dtype=c).dtype.type
                assert_(res is tgt)

    def test_ddof(self):
        for f in [_var]:
            for ddof in range(3):
                dim = self.rmat.shape[1]
                tgt = f(self.rmat, axis=1) * dim
                res = f(self.rmat, axis=1, ddof=ddof) * (dim - ddof)
        for f in [_std]:
            for ddof in range(3):
                dim = self.rmat.shape[1]
                tgt = f(self.rmat, axis=1) * np.sqrt(dim)
                res = f(self.rmat, axis=1, ddof=ddof) * np.sqrt(dim - ddof)
                assert_almost_equal(res, tgt)
                assert_almost_equal(res, tgt)

    def test_ddof_too_big(self):
        dim = self.rmat.shape[1]
        for f in [_var, _std]:
            for ddof in range(dim, dim + 2):
       #         with warnings.catch_warnings(record=True) as w:
       #             warnings.simplefilter('always')
                    res = f(self.rmat, axis=1, ddof=ddof)
                    assert_(not (res < 0).any())
        #            assert_(len(w) > 0)
        #            assert_(issubclass(w[0].category, RuntimeWarning))

    def test_empty(self):
        A = np.zeros((0, 3))
        for f in self.funcs:
            for axis in [0, None]:
          #      with warnings.catch_warnings(record=True) as w:
          #          warnings.simplefilter('always')
                    assert_(np.isnan(f(A, axis=axis)).all())
          #          assert_(len(w) > 0)
          #          assert_(issubclass(w[0].category, RuntimeWarning))
            for axis in [1]:
          #      with warnings.catch_warnings(record=True) as w:
          #          warnings.simplefilter('always')
                    assert_equal(f(A, axis=axis), np.zeros([]))

    def test_mean_values(self):
        for mat in [self.rmat, self.cmat]:
            for axis in [0, 1]:
                tgt = mat.sum(axis=axis)
                res = _mean(mat, axis=axis) * mat.shape[axis]
                assert_almost_equal(res, tgt)
            for axis in [None]:
                tgt = mat.sum(axis=axis)
                res = _mean(mat, axis=axis) * np.prod(mat.shape)
                assert_almost_equal(res, tgt)

    def test_mean_float16(self):
        # This fail if the sum inside mean is done in float16 instead
        # of float32.
        assert_(_mean(np.ones(100000, dtype='float16')) == 1)

    def test_mean_axis_error(self):
        # Ensure that AxisError is raised instead of IndexError when axis is
        # out of bounds, see gh-15817.
        with assert_raises(np.AxisError):
            np.arange(10).mean(axis=2)

    @pytest.mark.xfail(reason='implement mean(..., where=...)')
    def test_mean_where(self):
        a = np.arange(16).reshape((4, 4))
        wh_full = np.array([[False, True, False, True],
                            [True, False, True, False],
                            [True, True, False, False],
                            [False, False, True, True]])
        wh_partial = np.array([[False],
                               [True],
                               [True],
                               [False]])
        _cases = [(1, True, [1.5, 5.5, 9.5, 13.5]),
                  (0, wh_full, [6., 5., 10., 9.]),
                  (1, wh_full, [2., 5., 8.5, 14.5]),
                  (0, wh_partial, [6., 7., 8., 9.])]
        for _ax, _wh, _res in _cases:
            assert_allclose(a.mean(axis=_ax, where=_wh),
                            np.array(_res))
            assert_allclose(np.mean(a, axis=_ax, where=_wh),
                            np.array(_res))

        a3d = np.arange(16).reshape((2, 2, 4))
        _wh_partial = np.array([False, True, True, False])
        _res = [[1.5, 5.5], [9.5, 13.5]]
        assert_allclose(a3d.mean(axis=2, where=_wh_partial),
                        np.array(_res))
        assert_allclose(np.mean(a3d, axis=2, where=_wh_partial),
                        np.array(_res))

        with pytest.warns(RuntimeWarning) as w:
            assert_allclose(a.mean(axis=1, where=wh_partial),
                            np.array([np.nan, 5.5, 9.5, np.nan]))
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(a.mean(where=False), np.nan)
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(np.mean(a, where=False), np.nan)

    def test_var_values(self):
        for mat in [self.rmat, self.cmat]:
            for axis in [0, 1, None]:
                msqr = _mean(mat * mat.conj(), axis=axis)
                mean = _mean(mat, axis=axis)
                tgt = msqr - mean * mean.conjugate()
                res = _var(mat, axis=axis)
                assert_almost_equal(res, tgt)

    @pytest.mark.parametrize(('complex_dtype', 'ndec'), (
        ('complex64', 6),
        ('complex128', 7),
    ))
    def test_var_complex_values(self, complex_dtype, ndec):
        # Test fast-paths for every builtin complex type
        for axis in [0, 1, None]:
            mat = self.cmat.copy().astype(complex_dtype)
            msqr = _mean(mat * mat.conj(), axis=axis)
            mean = _mean(mat, axis=axis)
            tgt = msqr - mean * mean.conjugate()
            res = _var(mat, axis=axis)
            assert_almost_equal(res, tgt, decimal=ndec)

    def test_var_dimensions(self):
        # _var paths for complex number introduce additions on views that
        # increase dimensions. Ensure this generalizes to higher dims
        mat = np.stack([self.cmat]*3)
        for axis in [0, 1, 2, -1, None]:
            msqr = _mean(mat * mat.conj(), axis=axis)
            mean = _mean(mat, axis=axis)
            tgt = msqr - mean * mean.conjugate()
            res = _var(mat, axis=axis)
            assert_almost_equal(res, tgt)

    @pytest.mark.skip(reason='endianness')
    def test_var_complex_byteorder(self):
        # Test that var fast-path does not cause failures for complex arrays
        # with non-native byteorder
        cmat = self.cmat.copy().astype('complex128')
        cmat_swapped = cmat.astype(cmat.dtype.newbyteorder())
        assert_almost_equal(cmat.var(), cmat_swapped.var())

    def test_var_axis_error(self):
        # Ensure that AxisError is raised instead of IndexError when axis is
        # out of bounds, see gh-15817.
        with assert_raises(np.AxisError):
            np.arange(10).var(axis=2)

    @pytest.mark.xfail(reason="implement var(..., where=...)")
    def test_var_where(self):
        a = np.arange(25).reshape((5, 5))
        wh_full = np.array([[False, True, False, True, True],
                            [True, False, True, True, False],
                            [True, True, False, False, True],
                            [False, True, True, False, True],
                            [True, False, True, True, False]])
        wh_partial = np.array([[False],
                               [True],
                               [True],
                               [False],
                               [True]])
        _cases = [(0, True, [50., 50., 50., 50., 50.]),
                  (1, True, [2., 2., 2., 2., 2.])]
        for _ax, _wh, _res in _cases:
            assert_allclose(a.var(axis=_ax, where=_wh),
                            np.array(_res))
            assert_allclose(np.var(a, axis=_ax, where=_wh),
                            np.array(_res))

        a3d = np.arange(16).reshape((2, 2, 4))
        _wh_partial = np.array([False, True, True, False])
        _res = [[0.25, 0.25], [0.25, 0.25]]
        assert_allclose(a3d.var(axis=2, where=_wh_partial),
                        np.array(_res))
        assert_allclose(np.var(a3d, axis=2, where=_wh_partial),
                        np.array(_res))

        assert_allclose(np.var(a, axis=1, where=wh_full),
                        np.var(a[wh_full].reshape((5, 3)), axis=1))
        assert_allclose(np.var(a, axis=0, where=wh_partial),
                        np.var(a[wh_partial[:,0]], axis=0))
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(a.var(where=False), np.nan)
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(np.var(a, where=False), np.nan)

    def test_std_values(self):
        for mat in [self.rmat, self.cmat]:
            for axis in [0, 1, None]:
                tgt = np.sqrt(_var(mat, axis=axis))
                res = _std(mat, axis=axis)
                assert_almost_equal(res, tgt)

    @pytest.mark.xfail(reason="implement std(..., where=...)")
    def test_std_where(self):
        a = np.arange(25).reshape((5,5))[::-1]
        whf = np.array([[False, True, False, True, True],
                        [True, False, True, False, True],
                        [True, True, False, True, False],
                        [True, False, True, True, False],
                        [False, True, False, True, True]])
        whp = np.array([[False],
                        [False],
                        [True],
                        [True],
                        [False]])
        _cases = [
            (0, True, 7.07106781*np.ones((5))),
            (1, True, 1.41421356*np.ones((5))),
            (0, whf,
             np.array([4.0824829 , 8.16496581, 5., 7.39509973, 8.49836586])),
            (0, whp, 2.5*np.ones((5)))
        ]
        for _ax, _wh, _res in _cases:
            assert_allclose(a.std(axis=_ax, where=_wh), _res)
            assert_allclose(np.std(a, axis=_ax, where=_wh), _res)

        a3d = np.arange(16).reshape((2, 2, 4))
        _wh_partial = np.array([False, True, True, False])
        _res = [[0.5, 0.5], [0.5, 0.5]]
        assert_allclose(a3d.std(axis=2, where=_wh_partial),
                        np.array(_res))
        assert_allclose(np.std(a3d, axis=2, where=_wh_partial),
                        np.array(_res))

        assert_allclose(a.std(axis=1, where=whf),
                        np.std(a[whf].reshape((5,3)), axis=1))
        assert_allclose(np.std(a, axis=1, where=whf),
                        (a[whf].reshape((5,3))).std(axis=1))
        assert_allclose(a.std(axis=0, where=whp),
                        np.std(a[whp[:,0]], axis=0))
        assert_allclose(np.std(a, axis=0, where=whp),
                        (a[whp[:,0]]).std(axis=0))
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(a.std(where=False), np.nan)
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(np.std(a, where=False), np.nan)


class TestVdot:
    def test_basic(self):
        dt_numeric = np.typecodes['AllFloat'] + np.typecodes['AllInteger']
        dt_complex = np.typecodes['Complex']

        # test real
        a = np.eye(3)
        for dt in dt_numeric:
            b = a.astype(dt)
            res = np.vdot(b, b)
            assert_(np.isscalar(res))
            assert_equal(np.vdot(b, b), 3)

        # test complex
        a = np.eye(3) * 1j
        for dt in dt_complex:
            b = a.astype(dt)
            res = np.vdot(b, b)
            assert_(np.isscalar(res))
            assert_equal(np.vdot(b, b), 3)

        # test boolean
        b = np.eye(3, dtype=bool)
        res = np.vdot(b, b)
        assert_(np.isscalar(res))
        assert_equal(np.vdot(b, b), True)

    @pytest.mark.xfail(reason="implement order='F'")
    def test_vdot_array_order(self):
        a = np.array([[1, 2], [3, 4]], order='C')
        b = np.array([[1, 2], [3, 4]], order='F')
        res = np.vdot(a, a)

        # integer arrays are exact
        assert_equal(np.vdot(a, b), res)
        assert_equal(np.vdot(b, a), res)
        assert_equal(np.vdot(b, b), res)

    def test_vdot_uncontiguous(self):
        for size in [2, 1000]:
            # Different sizes match different branches in vdot.
            a = np.zeros((size, 2, 2))
            b = np.zeros((size, 2, 2))
            a[:, 0, 0] = np.arange(size)
            b[:, 0, 0] = np.arange(size) + 1
            # Make a and b uncontiguous:
            a = a[..., 0]
            b = b[..., 0]

            assert_equal(np.vdot(a, b),
                         np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a, b.copy()),
                         np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a.copy(), b),
                         np.vdot(a.flatten(), b.flatten()))

    @pytest.mark.xfail(reason="implement order='F'")
    def test_vdot_uncontiguous_2(self):
        # test order='F' separately
        for size in [2, 1000]:
            # Different sizes match different branches in vdot.
            a = np.zeros((size, 2, 2))
            b = np.zeros((size, 2, 2))
            a[:, 0, 0] = np.arange(size)
            b[:, 0, 0] = np.arange(size) + 1
            # Make a and b uncontiguous:
            a = a[..., 0]
            b = b[..., 0]

            assert_equal(np.vdot(a.copy('F'), b),
                         np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a, b.copy('F')),
                         np.vdot(a.flatten(), b.flatten()))


class TestDot:
    def setup_method(self):
        np.random.seed(128)

        # Numpy and pytorch random streams differ, so inline the
        # values from numpy 1.24.1
        # self.A = np.random.rand(4, 2)
        self.A = np.array([[0.86663704, 0.26314485],
               [0.13140848, 0.04159344],
               [0.23892433, 0.6454746 ],
               [0.79059935, 0.60144244]])

        # self.b1 = np.random.rand(2, 1)
        self.b1 = np.array([[0.33429937], [0.11942846]])

        # self.b2 = np.random.rand(2)
        self.b2 = np.array([0.30913305, 0.10972379])

        # self.b3 = np.random.rand(1, 2)
        self.b3 = np.array([[0.60211331, 0.25128496]])
        
        # self.b4 = np.random.rand(4)
        self.b4 = np.array([0.29968129, 0.517116, 0.71520252, 0.9314494])

        self.N = 7

    def test_dotmatmat(self):
        A = self.A
        res = np.dot(A.transpose(), A)
        tgt = np.array([[1.45046013, 0.86323640],
                        [0.86323640, 0.84934569]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotmatvec(self):
        A, b1 = self.A, self.b1
        res = np.dot(A, b1)
        tgt = np.array([[0.32114320], [0.04889721],
                        [0.15696029], [0.33612621]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotmatvec2(self):
        A, b2 = self.A, self.b2
        res = np.dot(A, b2)
        tgt = np.array([0.29677940, 0.04518649, 0.14468333, 0.31039293])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecmat(self):
        A, b4 = self.A, self.b4
        res = np.dot(b4, A)
        tgt = np.array([1.23495091, 1.12222648])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecmat2(self):
        b3, A = self.b3, self.A
        res = np.dot(b3, A.transpose())
        tgt = np.array([[0.58793804, 0.08957460, 0.30605758, 0.62716383]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecmat3(self):
        A, b4 = self.A, self.b4
        res = np.dot(A.transpose(), b4)
        tgt = np.array([1.23495091, 1.12222648])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecvecouter(self):
        b1, b3 = self.b1, self.b3
        res = np.dot(b1, b3)
        tgt = np.array([[0.20128610, 0.08400440], [0.07190947, 0.03001058]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecvecinner(self):
        b1, b3 = self.b1, self.b3
        res = np.dot(b3, b1)
        tgt = np.array([[ 0.23129668]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotcolumnvect1(self):
        b1 = np.ones((3, 1))
        b2 = [5.3]
        res = np.dot(b1, b2)
        tgt = np.array([5.3, 5.3, 5.3])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotcolumnvect2(self):
        b1 = np.ones((3, 1)).transpose()
        b2 = [6.2]
        res = np.dot(b2, b1)
        tgt = np.array([6.2, 6.2, 6.2])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecscalar(self):
        np.random.seed(100)
        # Numpy guarantees the random stream, and we don't. So inline the
        # values from numpy 1.24.1
        # b1 = np.random.rand(1, 1)
        b1 = np.array([[0.54340494]])

        # b2 = np.random.rand(1, 4)
        b2 = np.array([[0.27836939, 0.42451759, 0.84477613, 0.00471886]])

        res = np.dot(b1, b2)
        tgt = np.array([[0.15126730, 0.23068496, 0.45905553, 0.00256425]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecscalar2(self):
        np.random.seed(100)
        # b1 = np.random.rand(4, 1)
        b1 = np.array([[0.54340494], [0.27836939], [0.42451759], [0.84477613]])

        # b2 = np.random.rand(1, 1)
        b2 = np.array([[0.00471886]])

        res = np.dot(b1, b2)
        tgt = np.array([[0.00256425],[0.00131359],[0.00200324],[ 0.00398638]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_all(self):
        dims = [(), (1,), (1, 1)]
        dout = [(), (1,), (1, 1), (1,), (), (1,), (1, 1), (1,), (1, 1)]
        for dim, (dim1, dim2) in zip(dout, itertools.product(dims, dims)):
            b1 = np.zeros(dim1)
            b2 = np.zeros(dim2)
            res = np.dot(b1, b2)
            tgt = np.zeros(dim)
            assert_(res.shape == tgt.shape)
            assert_almost_equal(res, tgt, decimal=self.N)

    @pytest.mark.skip(reason='numpy internals')
    def test_dot_2args(self):
        from numpy.core.multiarray import dot

        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([[1, 0], [1, 1]], dtype=float)
        c = np.array([[3, 2], [7, 4]], dtype=float)

        d = dot(a, b)
        assert_allclose(c, d)

    @pytest.mark.skip(reason='numpy internals')
    def test_dot_3args(self):
        from numpy.core.multiarray import dot

        np.random.seed(22)
        f = np.random.random_sample((1024, 16))
        v = np.random.random_sample((16, 32))

        r = np.empty((1024, 32))
        for i in range(12):
            dot(f, v, r)
        if HAS_REFCOUNT:
            assert_equal(sys.getrefcount(r), 2)
        r2 = dot(f, v, out=None)
        assert_array_equal(r2, r)
        assert_(r is dot(f, v, out=r))

        v = v[:, 0].copy()  # v.shape == (16,)
        r = r[:, 0].copy()  # r.shape == (1024,)
        r2 = dot(f, v)
        assert_(r is dot(f, v, r))
        assert_array_equal(r2, r)

    @pytest.mark.skip(reason='numpy internals')
    def test_dot_3args_errors(self):
        from numpy.core.multiarray import dot

        np.random.seed(22)
        f = np.random.random_sample((1024, 16))
        v = np.random.random_sample((16, 32))

        r = np.empty((1024, 31))
        assert_raises(ValueError, dot, f, v, r)

        r = np.empty((1024,))
        assert_raises(ValueError, dot, f, v, r)

        r = np.empty((32,))
        assert_raises(ValueError, dot, f, v, r)

        r = np.empty((32, 1024))
        assert_raises(ValueError, dot, f, v, r)
        assert_raises(ValueError, dot, f, v, r.T)

        r = np.empty((1024, 64))
        assert_raises(ValueError, dot, f, v, r[:, ::2])
        assert_raises(ValueError, dot, f, v, r[:, :32])

        r = np.empty((1024, 32), dtype=np.float32)
        assert_raises(ValueError, dot, f, v, r)

        r = np.empty((1024, 32), dtype=int)
        assert_raises(ValueError, dot, f, v, r)

    @pytest.mark.xfail(reason="TODO order='F'")
    def test_dot_array_order(self):
        a = np.array([[1, 2], [3, 4]], order='C')
        b = np.array([[1, 2], [3, 4]], order='F')
        res = np.dot(a, a)

        # integer arrays are exact
        assert_equal(np.dot(a, b), res)
        assert_equal(np.dot(b, a), res)
        assert_equal(np.dot(b, b), res)

    @pytest.mark.skip(reason='TODO: nbytes, view, __array_interface__')
    def test_accelerate_framework_sgemv_fix(self):

        def aligned_array(shape, align, dtype, order='C'):
            d = dtype(0)
            N = np.prod(shape)
            tmp = np.zeros(N * d.nbytes + align, dtype=np.uint8)
            address = tmp.__array_interface__["data"][0]
            for offset in range(align):
                if (address + offset) % align == 0:
                    break
            tmp = tmp[offset:offset+N*d.nbytes].view(dtype=dtype)
            return tmp.reshape(shape, order=order)

        def as_aligned(arr, align, dtype, order='C'):
            aligned = aligned_array(arr.shape, align, dtype, order)
            aligned[:] = arr[:]
            return aligned

        def assert_dot_close(A, X, desired):
            assert_allclose(np.dot(A, X), desired, rtol=1e-5, atol=1e-7)

        m = aligned_array(100, 15, np.float32)
        s = aligned_array((100, 100), 15, np.float32)
        np.dot(s, m)  # this will always segfault if the bug is present

        testdata = itertools.product((15, 32), (10000,), (200, 89), ('C', 'F'))
        for align, m, n, a_order in testdata:
            # Calculation in double precision
            A_d = np.random.rand(m, n)
            X_d = np.random.rand(n)
            desired = np.dot(A_d, X_d)
            # Calculation with aligned single precision
            A_f = as_aligned(A_d, align, np.float32, order=a_order)
            X_f = as_aligned(X_d, align, np.float32)
            assert_dot_close(A_f, X_f, desired)
            # Strided A rows
            A_d_2 = A_d[::2]
            desired = np.dot(A_d_2, X_d)
            A_f_2 = A_f[::2]
            assert_dot_close(A_f_2, X_f, desired)
            # Strided A columns, strided X vector
            A_d_22 = A_d_2[:, ::2]
            X_d_2 = X_d[::2]
            desired = np.dot(A_d_22, X_d_2)
            A_f_22 = A_f_2[:, ::2]
            X_f_2 = X_f[::2]
            assert_dot_close(A_f_22, X_f_2, desired)
            # Check the strides are as expected
            if a_order == 'F':
                assert_equal(A_f_22.strides, (8, 8 * m))
            else:
                assert_equal(A_f_22.strides, (8 * n, 8))
            assert_equal(X_f_2.strides, (8,))
            # Strides in A rows + cols only
            X_f_2c = as_aligned(X_f_2, align, np.float32)
            assert_dot_close(A_f_22, X_f_2c, desired)
            # Strides just in A cols
            A_d_12 = A_d[:, ::2]
            desired = np.dot(A_d_12, X_d_2)
            A_f_12 = A_f[:, ::2]
            assert_dot_close(A_f_12, X_f_2c, desired)
            # Strides in A cols and X
            assert_dot_close(A_f_12, X_f_2, desired)

    @pytest.mark.slow
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @requires_memory(free_bytes=18e9)  # complex case needs 18GiB+
    def test_huge_vectordot(self, dtype):
        # Large vector multiplications are chunked with 32bit BLAS
        # Test that the chunking does the right thing, see also gh-22262
        data = np.ones(2**30+100, dtype=dtype)
        res = np.dot(data, data)
        assert res == 2**30+100


class MatmulCommon:
    """Common tests for '@' operator and numpy.matmul.

    """
    # Should work with these types. Will want to add
    # "O" at some point
    types = "?bhilBefdFD"

    def test_exceptions(self):
        dims = [
            ((1,), (2,)),            # mismatched vector vector
            ((2, 1,), (2,)),         # mismatched matrix vector
            ((2,), (1, 2)),          # mismatched vector matrix
            ((1, 2), (3, 1)),        # mismatched matrix matrix
            ((1,), ()),              # vector scalar
            ((), (1)),               # scalar vector
            ((1, 1), ()),            # matrix scalar
            ((), (1, 1)),            # scalar matrix
            ((2, 2, 1), (3, 1, 2)),  # cannot broadcast
            ]

        for dt, (dm1, dm2) in itertools.product(self.types, dims):
            a = np.ones(dm1, dtype=dt)
            b = np.ones(dm2, dtype=dt)
            assert_raises((RuntimeError, ValueError), self.matmul, a, b)

    def test_shapes(self):
        dims = [
            ((1, 1), (2, 1, 1)),     # broadcast first argument
            ((2, 1, 1), (1, 1)),     # broadcast second argument
            ((2, 1, 1), (2, 1, 1)),  # matrix stack sizes match
            ]

        for dt, (dm1, dm2) in itertools.product(self.types, dims):
            a = np.ones(dm1, dtype=dt)
            b = np.ones(dm2, dtype=dt)
            res = self.matmul(a, b)
            assert_(res.shape == (2, 1, 1))

        # vector vector returns scalars.
        for dt in self.types:
            a = np.ones((2,), dtype=dt)
            b = np.ones((2,), dtype=dt)
            c = self.matmul(a, b)
            assert_(np.array(c).shape == ())

    def test_result_types(self):
        mat = np.ones((1,1))
        vec = np.ones((1,))
        for dt in self.types:
            m = mat.astype(dt)
            v = vec.astype(dt)
            for arg in [(m, v), (v, m), (m, m)]:
                res = self.matmul(*arg)
                assert_(res.dtype == dt)

    @pytest.mark.xfail(reason='no scalars')
    def test_result_types_2(self):
        # in numpy, vector vector returns scalars
        # we return a 0D array instead

        for dt in self.types:
            v = np.ones((1,)).astype(dt)
            if dt != "O":
                res = self.matmul(v, v)
                assert_(type(res) is np.dtype(dt).type)

    def test_scalar_output(self):
        vec1 = np.array([2])
        vec2 = np.array([3, 4]).reshape(1, -1)
        tgt = np.array([6, 8])
        for dt in self.types[1:]:
            v1 = vec1.astype(dt)
            v2 = vec2.astype(dt)
            res = self.matmul(v1, v2)
            assert_equal(res, tgt)
            res = self.matmul(v2.T, v1)
            assert_equal(res, tgt)

        # boolean type
        vec = np.array([True, True], dtype='?').reshape(1, -1)
        res = self.matmul(vec[:, 0], vec)
        assert_equal(res, True)

    def test_vector_vector_values(self):
        vec1 = np.array([1, 2])
        vec2 = np.array([3, 4]).reshape(-1, 1)
        tgt1 = np.array([11])
        tgt2 = np.array([[3, 6], [4, 8]])
        for dt in self.types[1:]:
            v1 = vec1.astype(dt)
            v2 = vec2.astype(dt)
            res = self.matmul(v1, v2)
            assert_equal(res, tgt1)
            # no broadcast, we must make v1 into a 2d ndarray
            res = self.matmul(v2, v1.reshape(1, -1))
            assert_equal(res, tgt2)

        # boolean type
        vec = np.array([True, True], dtype='?')
        res = self.matmul(vec, vec)
        assert_equal(res, True)

    def test_vector_matrix_values(self):
        vec = np.array([1, 2])
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.stack([mat1]*2, axis=0)
        tgt1 = np.array([7, 10])
        tgt2 = np.stack([tgt1]*2, axis=0)
        for dt in self.types[1:]:
            v = vec.astype(dt)
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            res = self.matmul(v, m1)
            assert_equal(res, tgt1)
            res = self.matmul(v, m2)
            assert_equal(res, tgt2)

        # boolean type
        vec = np.array([True, False])
        mat1 = np.array([[True, False], [False, True]])
        mat2 = np.stack([mat1]*2, axis=0)
        tgt1 = np.array([True, False])
        tgt2 = np.stack([tgt1]*2, axis=0)

        res = self.matmul(vec, mat1)
        assert_equal(res, tgt1)
        res = self.matmul(vec, mat2)
        assert_equal(res, tgt2)

    def test_matrix_vector_values(self):
        vec = np.array([1, 2])
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.stack([mat1]*2, axis=0)
        tgt1 = np.array([5, 11])
        tgt2 = np.stack([tgt1]*2, axis=0)
        for dt in self.types[1:]:
            v = vec.astype(dt)
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            res = self.matmul(m1, v)
            assert_equal(res, tgt1)
            res = self.matmul(m2, v)
            assert_equal(res, tgt2)

        # boolean type
        vec = np.array([True, False])
        mat1 = np.array([[True, False], [False, True]])
        mat2 = np.stack([mat1]*2, axis=0)
        tgt1 = np.array([True, False])
        tgt2 = np.stack([tgt1]*2, axis=0)

        res = self.matmul(vec, mat1)
        assert_equal(res, tgt1)
        res = self.matmul(vec, mat2)
        assert_equal(res, tgt2)

    def test_matrix_matrix_values(self):
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.array([[1, 0], [1, 1]])
        mat12 = np.stack([mat1, mat2], axis=0)
        mat21 = np.stack([mat2, mat1], axis=0)
        tgt11 = np.array([[7, 10], [15, 22]])
        tgt12 = np.array([[3, 2], [7, 4]])
        tgt21 = np.array([[1, 2], [4, 6]])
        tgt12_21 = np.stack([tgt12, tgt21], axis=0)
        tgt11_12 = np.stack((tgt11, tgt12), axis=0)
        tgt11_21 = np.stack((tgt11, tgt21), axis=0)
        for dt in self.types[1:]:
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            m12 = mat12.astype(dt)
            m21 = mat21.astype(dt)

            # matrix @ matrix
            res = self.matmul(m1, m2)
            assert_equal(res, tgt12)
            res = self.matmul(m2, m1)
            assert_equal(res, tgt21)

            # stacked @ matrix
            res = self.matmul(m12, m1)
            assert_equal(res, tgt11_21)

            # matrix @ stacked
            res = self.matmul(m1, m12)
            assert_equal(res, tgt11_12)

            # stacked @ stacked
            res = self.matmul(m12, m21)
            assert_equal(res, tgt12_21)

        # boolean type
        m1 = np.array([[1, 1], [0, 0]], dtype=np.bool_)
        m2 = np.array([[1, 0], [1, 1]], dtype=np.bool_)
        m12 = np.stack([m1, m2], axis=0)
        m21 = np.stack([m2, m1], axis=0)
        tgt11 = m1
        tgt12 = m1
        tgt21 = np.array([[1, 1], [1, 1]], dtype=np.bool_)
        tgt12_21 = np.stack([tgt12, tgt21], axis=0)
        tgt11_12 = np.stack((tgt11, tgt12), axis=0)
        tgt11_21 = np.stack((tgt11, tgt21), axis=0)

        # matrix @ matrix
        res = self.matmul(m1, m2)
        assert_equal(res, tgt12)
        res = self.matmul(m2, m1)
        assert_equal(res, tgt21)

        # stacked @ matrix
        res = self.matmul(m12, m1)
        assert_equal(res, tgt11_21)

        # matrix @ stacked
        res = self.matmul(m1, m12)
        assert_equal(res, tgt11_12)

        # stacked @ stacked
        res = self.matmul(m12, m21)
        assert_equal(res, tgt12_21)


class TestMatmul(MatmulCommon):

    def setup_method(self):
        self.matmul = np.matmul

    def test_out_arg(self):
        a = np.ones((5, 2), dtype=float)
        b = np.array([[1, 3], [5, 7]], dtype=float)
        tgt = np.dot(a, b)

        # test as positional argument
        msg = "out positional argument"
        out = np.zeros((5, 2), dtype=float)
        self.matmul(a, b, out)
        assert_array_equal(out, tgt, err_msg=msg)

        # test as keyword argument
        msg = "out keyword argument"
        out = np.zeros((5, 2), dtype=float)
        self.matmul(a, b, out=out)
        assert_array_equal(out, tgt, err_msg=msg)

        # test out with not allowed type cast (safe casting)
        msg = "Cannot cast"
        out = np.zeros((5, 2), dtype=np.int32)
        assert_raises_regex(TypeError, msg, self.matmul, a, b, out=out)

        # test out with type upcast to complex
        out = np.zeros((5, 2), dtype=np.complex128)
        c = self.matmul(a, b, out=out)
        assert_(c is out)
  #      with suppress_warnings() as sup:
  #          sup.filter(np.ComplexWarning, '')
        c = c.astype(tgt.dtype)
        assert_array_equal(c, tgt)

    def test_empty_out(self):
        # Check that the output cannot be broadcast, so that it cannot be
        # size zero when the outer dimensions (iterator size) has size zero.
        arr = np.ones((0, 1, 1))
        out = np.ones((1, 1, 1))
        assert self.matmul(arr, arr).shape == (0, 1, 1)

        with pytest.raises((RuntimeError, ValueError)):
            self.matmul(arr, arr, out=out)

    def test_out_contiguous(self):
        a = np.ones((5, 2), dtype=float)
        b = np.array([[1, 3], [5, 7]], dtype=float)
        v = np.array([1, 3], dtype=float)
        tgt = np.dot(a, b)
        tgt_mv = np.dot(a, v)

        # test out non-contiguous
        out = np.ones((5, 2, 2), dtype=float)
        c = self.matmul(a, b, out=out[..., 0])
        assert c.tensor._base is out.tensor
        assert_array_equal(c, tgt)
        c = self.matmul(a, v, out=out[:, 0, 0])
        assert_array_equal(c, tgt_mv)
        c = self.matmul(v, a.T, out=out[:, 0, 0])
        assert_array_equal(c, tgt_mv)

        # test out contiguous in only last dim
        out = np.ones((10, 2), dtype=float)
        c = self.matmul(a, b, out=out[::2, :])
        assert_array_equal(c, tgt)

        # test transposes of out, args
        out = np.ones((5, 2), dtype=float)
        c = self.matmul(b.T, a.T, out=out.T)
        assert_array_equal(out, tgt)

    m1 = np.arange(15.).reshape(5, 3)
    m2 = np.arange(21.).reshape(3, 7)
    m3 = np.arange(30.).reshape(5, 6)[:, ::2]  # non-contiguous
    vc = np.arange(10.)
    vr = np.arange(6.)
    m0 = np.zeros((3, 0))
    @pytest.mark.parametrize('args', (
            # matrix-matrix
            (m1, m2), (m2.T, m1.T), (m2.T.copy(), m1.T), (m2.T, m1.T.copy()),
            # matrix-matrix-transpose, contiguous and non
            (m1, m1.T), (m1.T, m1), (m1, m3.T), (m3, m1.T),
            (m3, m3.T), (m3.T, m3),
            # matrix-matrix non-contiguous
            (m3, m2), (m2.T, m3.T), (m2.T.copy(), m3.T),
            # vector-matrix, matrix-vector, contiguous
            (m1, vr[:3]), (vc[:5], m1), (m1.T, vc[:5]), (vr[:3], m1.T),
            # vector-matrix, matrix-vector, vector non-contiguous
            (m1, vr[::2]), (vc[::2], m1), (m1.T, vc[::2]), (vr[::2], m1.T),
            # vector-matrix, matrix-vector, matrix non-contiguous
            (m3, vr[:3]), (vc[:5], m3), (m3.T, vc[:5]), (vr[:3], m3.T),
            # vector-matrix, matrix-vector, both non-contiguous
            (m3, vr[::2]), (vc[::2], m3), (m3.T, vc[::2]), (vr[::2], m3.T),
            # size == 0
            (m0, m0.T), (m0.T, m0), (m1, m0), (m0.T, m1.T),
        ))
    def test_dot_equivalent(self, args):
        r1 = np.matmul(*args)
        r2 = np.dot(*args)
        assert_equal(r1, r2)

        r3 = np.matmul(args[0].copy(), args[1].copy())
        assert_equal(r1, r3)


    @pytest.mark.skip(reason='object arrays')
    def test_matmul_exception_multiply(self):
        # test that matmul fails if `__mul__` is missing
        class add_not_multiply():
            def __add__(self, other):
                return self
        a = np.full((3,3), add_not_multiply())
        with assert_raises(TypeError):
            b = np.matmul(a, a)

    @pytest.mark.skip(reason='object arrays')
    def test_matmul_exception_add(self):
        # test that matmul fails if `__add__` is missing
        class multiply_not_add():
            def __mul__(self, other):
                return self
        a = np.full((3,3), multiply_not_add())
        with assert_raises(TypeError):
            b = np.matmul(a, a)

    @pytest.mark.xfail(reason="TODO: implement .view")
    def test_matmul_bool(self):
        # gh-14439
        a = np.array([[1, 0],[1, 1]], dtype=bool)
        assert np.max(a.view(np.uint8)) == 1
        b = np.matmul(a, a)
        # matmul with boolean output should always be 0, 1
        assert np.max(b.view(np.uint8)) == 1

        rg = np.random.default_rng(np.random.PCG64(43))
        d = rg.integers(2, size=4*5, dtype=np.int8)
        d = d.reshape(4, 5) > 0
        out1 = np.matmul(d, d.reshape(5, 4))
        out2 = np.dot(d, d.reshape(5, 4))
        assert_equal(out1, out2)

        c = np.matmul(np.zeros((2, 0), dtype=bool), np.zeros(0, dtype=bool))
        assert not np.any(c)


class TestMatmulOperator(MatmulCommon):
    import operator
    matmul = operator.matmul

    @pytest.mark.skip(reason="no __array_priority__")
    def test_array_priority_override(self):

        class A:
            __array_priority__ = 1000

            def __matmul__(self, other):
                return "A"

            def __rmatmul__(self, other):
                return "A"

        a = A()
        b = np.ones(2)
        assert_equal(self.matmul(a, b), "A")
        assert_equal(self.matmul(b, a), "A")

    def test_matmul_raises(self):
        assert_raises((RuntimeError, TypeError), self.matmul, np.int8(5), np.int8(5))


@pytest.mark.xfail(reason="torch supports inplace matmul, and so do we")
def test_matmul_inplace():
    # It would be nice to support in-place matmul eventually, but for now
    # we don't have a working implementation, so better just to error out
    # and nudge people to writing "a = a @ b".
    a = np.eye(3)
    b = np.eye(3)
    assert_raises(TypeError, a.__imatmul__, b)
    import operator
    assert_raises(TypeError, operator.imatmul, a, b)
    assert_raises(TypeError, exec, "a @= b", globals(), locals())


@pytest.mark.xfail(reason='matmul_axes')
def test_matmul_axes():
    a = np.arange(3*4*5).reshape(3, 4, 5)
    c = np.matmul(a, a, axes=[(-2, -1), (-1, -2), (1, 2)])
    assert c.shape == (3, 4, 4)
    d = np.matmul(a, a, axes=[(-2, -1), (-1, -2), (0, 1)])
    assert d.shape == (4, 4, 3)
    e = np.swapaxes(d, 0, 2)
    assert_array_equal(e, c)
    f = np.matmul(a, np.arange(3), axes=[(1, 0), (0), (0)])
    assert f.shape == (4, 5)


class TestInner:

    def test_inner_scalar_and_vector(self):
        for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
            sca = np.array(3, dtype=dt)[()]
            vec = np.array([1, 2], dtype=dt)
            desired = np.array([3, 6], dtype=dt)
            assert_equal(np.inner(vec, sca), desired)
            assert_equal(np.inner(sca, vec), desired)

    def test_vecself(self):
        # Ticket 844.
        # Inner product of a vector with itself segfaults or give
        # meaningless result
        a = np.zeros(shape=(1, 80), dtype=np.float64)
        p = np.inner(a, a)
        assert_almost_equal(p, 0, decimal=14)

    def test_inner_product_with_various_contiguities(self):
        # github issue 6532
        for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
            # check an inner product involving a matrix transpose
            A = np.array([[1, 2], [3, 4]], dtype=dt)
            B = np.array([[1, 3], [2, 4]], dtype=dt)
            C = np.array([1, 1], dtype=dt)
            desired = np.array([4, 6], dtype=dt)
            assert_equal(np.inner(A.T, C), desired)
            assert_equal(np.inner(C, A.T), desired)
            assert_equal(np.inner(B, C), desired)
            assert_equal(np.inner(C, B), desired)
            # check a matrix product
            desired = np.array([[7, 10], [15, 22]], dtype=dt)
            assert_equal(np.inner(A, B), desired)
            # check the syrk vs. gemm paths
            desired = np.array([[5, 11], [11, 25]], dtype=dt)
            assert_equal(np.inner(A, A), desired)
            assert_equal(np.inner(A, A.copy()), desired)

    @pytest.mark.skip(reason="[::-1] not supported")
    def test_inner_product_reversed_view(self):
        for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
            # check an inner product involving an aliased and reversed view
            a = np.arange(5).astype(dt)
            b = a[::-1]
            desired = np.array(10, dtype=dt).item()
            assert_equal(np.inner(b, a), desired)

    def test_3d_tensor(self):
        for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
            a = np.arange(24).reshape(2,3,4).astype(dt)
            b = np.arange(24, 48).reshape(2,3,4).astype(dt)
            desired = np.array(
                [[[[ 158,  182,  206],
                   [ 230,  254,  278]],

                  [[ 566,  654,  742],
                   [ 830,  918, 1006]],

                  [[ 974, 1126, 1278],
                   [1430, 1582, 1734]]],

                 [[[1382, 1598, 1814],
                   [2030, 2246, 2462]],

                  [[1790, 2070, 2350],
                   [2630, 2910, 3190]],

                  [[2198, 2542, 2886],
                   [3230, 3574, 3918]]]]
            ).astype(dt)
            assert_equal(np.inner(a, b), desired)
            assert_equal(np.inner(b, a).transpose(2,3,0,1), desired)


class TestChoose:
    def setup_method(self):
        self.x = 2*np.ones((3,), dtype=int)
        self.y = 3*np.ones((3,), dtype=int)
        self.x2 = 2*np.ones((2, 3), dtype=int)
        self.y2 = 3*np.ones((2, 3), dtype=int)
        self.ind = [0, 0, 1]

    def test_basic(self):
        A = np.choose(self.ind, (self.x, self.y))
        assert_equal(A, [2, 2, 3])

    def test_broadcast1(self):
        A = np.choose(self.ind, (self.x2, self.y2))
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])

    def test_broadcast2(self):
        A = np.choose(self.ind, (self.x, self.y2))
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])

    @pytest.mark.parametrize("ops",
        [(1000, np.array([1], dtype=np.uint8)),
         (-1, np.array([1], dtype=np.uint8)),
         (1., np.float32(3)),
         (1., np.array([3], dtype=np.float32))],)
    def test_output_dtype(self, ops):
        expected_dt = np.result_type(*ops)
        assert(np.choose([0], ops).dtype == expected_dt)

    def test_docstring_1(self):
        # examples from the docstring,
        # https://numpy.org/doc/1.23/reference/generated/numpy.choose.html
        choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                   [20, 21, 22, 23], [30, 31, 32, 33]]
        A = np.choose([2, 3, 1, 0], choices)

        assert_equal(A, [20, 31, 12,  3])

    def test_docstring_2(self):
        a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        choices = [-10, 10]
        A = np.choose(a, choices)
        assert_equal(A, [[ 10, -10,  10],
                         [-10,  10, -10],
                         [ 10, -10,  10]])

    def test_docstring_3(self):
        a = np.array([0, 1]).reshape((2, 1, 1))
        c1 = np.array([1, 2, 3]).reshape((1, 3, 1))
        c2 = np.array([-1, -2, -3, -4, -5]).reshape((1, 1, 5))
        A = np.choose(a, (c1, c2)) # result is 2x3x5, res[0,:,:]=c1, res[1,:,:]=c2
        expected = np.array([[[ 1,  1,  1,  1,  1],
                              [ 2,  2,  2,  2,  2],
                              [ 3,  3,  3,  3,  3]],
                            [[-1, -2, -3, -4, -5],
                             [-1, -2, -3, -4, -5],
                             [-1, -2, -3, -4, -5]]])
        assert_equal(A, expected)


class TestRepeat:
    def setup_method(self):
        self.m = np.array([1, 2, 3, 4, 5, 6])
        self.m_rect = self.m.reshape((2, 3))

    def test_basic(self):
        A = np.repeat(self.m, [1, 3, 2, 1, 1, 2])
        assert_equal(A, [1, 2, 2, 2, 3,
                         3, 4, 5, 6, 6])

    def test_broadcast1(self):
        A = np.repeat(self.m, 2)
        assert_equal(A, [1, 1, 2, 2, 3, 3,
                         4, 4, 5, 5, 6, 6])

    def test_axis_spec(self):
        A = np.repeat(self.m_rect, [2, 1], axis=0)
        assert_equal(A, [[1, 2, 3],
                         [1, 2, 3],
                         [4, 5, 6]])

        A = np.repeat(self.m_rect, [1, 3, 2], axis=1)
        assert_equal(A, [[1, 2, 2, 2, 3, 3],
                         [4, 5, 5, 5, 6, 6]])

    def test_broadcast2(self):
        A = np.repeat(self.m_rect, 2, axis=0)
        assert_equal(A, [[1, 2, 3],
                         [1, 2, 3],
                         [4, 5, 6],
                         [4, 5, 6]])

        A = np.repeat(self.m_rect, 2, axis=1)
        assert_equal(A, [[1, 1, 2, 2, 3, 3],
                         [4, 4, 5, 5, 6, 6]])


# TODO: test for multidimensional
NEIGH_MODE = {'zero': 0, 'one': 1, 'constant': 2, 'circular': 3, 'mirror': 4}


@pytest.mark.xfail(reason='TODO')
@pytest.mark.parametrize('dt', [float, Decimal], ids=['float', 'object'])
class TestNeighborhoodIter:
    # Simple, 2d tests
    def test_simple2d(self, dt):
        # Test zero and one padding for simple data type
        x = np.array([[0, 1], [2, 3]], dtype=dt)
        r = [np.array([[0, 0, 0], [0, 0, 1]], dtype=dt),
             np.array([[0, 0, 0], [0, 1, 0]], dtype=dt),
             np.array([[0, 0, 1], [0, 2, 3]], dtype=dt),
             np.array([[0, 1, 0], [2, 3, 0]], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 0, -1, 1], x[0], NEIGH_MODE['zero'])
        assert_array_equal(l, r)

        r = [np.array([[1, 1, 1], [1, 0, 1]], dtype=dt),
             np.array([[1, 1, 1], [0, 1, 1]], dtype=dt),
             np.array([[1, 0, 1], [1, 2, 3]], dtype=dt),
             np.array([[0, 1, 1], [2, 3, 1]], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 0, -1, 1], x[0], NEIGH_MODE['one'])
        assert_array_equal(l, r)

        r = [np.array([[4, 4, 4], [4, 0, 1]], dtype=dt),
             np.array([[4, 4, 4], [0, 1, 4]], dtype=dt),
             np.array([[4, 0, 1], [4, 2, 3]], dtype=dt),
             np.array([[0, 1, 4], [2, 3, 4]], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 0, -1, 1], 4, NEIGH_MODE['constant'])
        assert_array_equal(l, r)

        # Test with start in the middle
        r = [np.array([[4, 0, 1], [4, 2, 3]], dtype=dt),
             np.array([[0, 1, 4], [2, 3, 4]], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 0, -1, 1], 4, NEIGH_MODE['constant'], 2)
        assert_array_equal(l, r)

    def test_mirror2d(self, dt):
        x = np.array([[0, 1], [2, 3]], dtype=dt)
        r = [np.array([[0, 0, 1], [0, 0, 1]], dtype=dt),
             np.array([[0, 1, 1], [0, 1, 1]], dtype=dt),
             np.array([[0, 0, 1], [2, 2, 3]], dtype=dt),
             np.array([[0, 1, 1], [2, 3, 3]], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 0, -1, 1], x[0], NEIGH_MODE['mirror'])
        assert_array_equal(l, r)

    # Simple, 1d tests
    def test_simple(self, dt):
        # Test padding with constant values
        x = np.linspace(1, 5, 5).astype(dt)
        r = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0]]
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 1], x[0], NEIGH_MODE['zero'])
        assert_array_equal(l, r)

        r = [[1, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 1]]
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 1], x[0], NEIGH_MODE['one'])
        assert_array_equal(l, r)

        r = [[x[4], 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, x[4]]]
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 1], x[4], NEIGH_MODE['constant'])
        assert_array_equal(l, r)

    # Test mirror modes
    def test_mirror(self, dt):
        x = np.linspace(1, 5, 5).astype(dt)
        r = np.array([[2, 1, 1, 2, 3], [1, 1, 2, 3, 4], [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 5], [3, 4, 5, 5, 4]], dtype=dt)
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-2, 2], x[1], NEIGH_MODE['mirror'])
        assert_([i.dtype == dt for i in l])
        assert_array_equal(l, r)

    # Circular mode
    def test_circular(self, dt):
        x = np.linspace(1, 5, 5).astype(dt)
        r = np.array([[4, 5, 1, 2, 3], [5, 1, 2, 3, 4], [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 1], [3, 4, 5, 1, 2]], dtype=dt)
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-2, 2], x[0], NEIGH_MODE['circular'])
        assert_array_equal(l, r)


# Test stacking neighborhood iterators
@pytest.mark.xfail(reason='TODO')
class TestStackedNeighborhoodIter:
    # Simple, 1d test: stacking 2 constant-padded neigh iterators
    def test_simple_const(self):
        dt = np.float64
        # Test zero and one padding for simple data type
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([0], dtype=dt),
             np.array([0], dtype=dt),
             np.array([1], dtype=dt),
             np.array([2], dtype=dt),
             np.array([3], dtype=dt),
             np.array([0], dtype=dt),
             np.array([0], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-2, 4], NEIGH_MODE['zero'], [0, 0], NEIGH_MODE['zero'])
        assert_array_equal(l, r)

        r = [np.array([1, 0, 1], dtype=dt),
             np.array([0, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 0], dtype=dt),
             np.array([3, 0, 1], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [-1, 1], NEIGH_MODE['one'])
        assert_array_equal(l, r)

    # 2nd simple, 1d test: stacking 2 neigh iterators, mixing const padding and
    # mirror padding
    def test_simple_mirror(self):
        dt = np.float64
        # Stacking zero on top of mirror
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([0, 1, 1], dtype=dt),
             np.array([1, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 3], dtype=dt),
             np.array([3, 3, 0], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['mirror'], [-1, 1], NEIGH_MODE['zero'])
        assert_array_equal(l, r)

        # Stacking mirror on top of zero
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([1, 0, 0], dtype=dt),
             np.array([0, 0, 1], dtype=dt),
             np.array([0, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 0], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [-2, 0], NEIGH_MODE['mirror'])
        assert_array_equal(l, r)

        # Stacking mirror on top of zero: 2nd
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([0, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 0], dtype=dt),
             np.array([3, 0, 0], dtype=dt),
             np.array([0, 0, 3], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [0, 2], NEIGH_MODE['mirror'])
        assert_array_equal(l, r)

        # Stacking mirror on top of zero: 3rd
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([1, 0, 0, 1, 2], dtype=dt),
             np.array([0, 0, 1, 2, 3], dtype=dt),
             np.array([0, 1, 2, 3, 0], dtype=dt),
             np.array([1, 2, 3, 0, 0], dtype=dt),
             np.array([2, 3, 0, 0, 3], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [-2, 2], NEIGH_MODE['mirror'])
        assert_array_equal(l, r)

    # 3rd simple, 1d test: stacking 2 neigh iterators, mixing const padding and
    # circular padding
    def test_simple_circular(self):
        dt = np.float64
        # Stacking zero on top of mirror
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([0, 3, 1], dtype=dt),
             np.array([3, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 1], dtype=dt),
             np.array([3, 1, 0], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['circular'], [-1, 1], NEIGH_MODE['zero'])
        assert_array_equal(l, r)

        # Stacking mirror on top of zero
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([3, 0, 0], dtype=dt),
             np.array([0, 0, 1], dtype=dt),
             np.array([0, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 0], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [-2, 0], NEIGH_MODE['circular'])
        assert_array_equal(l, r)

        # Stacking mirror on top of zero: 2nd
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([0, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 0], dtype=dt),
             np.array([3, 0, 0], dtype=dt),
             np.array([0, 0, 1], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [0, 2], NEIGH_MODE['circular'])
        assert_array_equal(l, r)

        # Stacking mirror on top of zero: 3rd
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([3, 0, 0, 1, 2], dtype=dt),
             np.array([0, 0, 1, 2, 3], dtype=dt),
             np.array([0, 1, 2, 3, 0], dtype=dt),
             np.array([1, 2, 3, 0, 0], dtype=dt),
             np.array([2, 3, 0, 0, 1], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [-2, 2], NEIGH_MODE['circular'])
        assert_array_equal(l, r)

    # 4th simple, 1d test: stacking 2 neigh iterators, but with lower iterator
    # being strictly within the array
    def test_simple_strict_within(self):
        dt = np.float64
        # Stacking zero on top of zero, first neighborhood strictly inside the
        # array
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([1, 2, 3, 0], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [1, 1], NEIGH_MODE['zero'], [-1, 2], NEIGH_MODE['zero'])
        assert_array_equal(l, r)

        # Stacking mirror on top of zero, first neighborhood strictly inside the
        # array
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([1, 2, 3, 3], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [1, 1], NEIGH_MODE['zero'], [-1, 2], NEIGH_MODE['mirror'])
        assert_array_equal(l, r)

        # Stacking mirror on top of zero, first neighborhood strictly inside the
        # array
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([1, 2, 3, 1], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [1, 1], NEIGH_MODE['zero'], [-1, 2], NEIGH_MODE['circular'])
        assert_array_equal(l, r)

@pytest.mark.xfail(reason='TODO')
class TestWarnings:

    def test_complex_warning(self):
        x = np.array([1, 2])
        y = np.array([1-2j, 1+2j])

        with warnings.catch_warnings():
            warnings.simplefilter("error", np.ComplexWarning)
            assert_raises(np.ComplexWarning, x.__setitem__, slice(None), y)
            assert_equal(x, [1, 2])


class TestMinScalarType:

    def test_usigned_shortshort(self):
        dt = np.min_scalar_type(2**8-1)
        wanted = np.dtype('uint8')
        assert_equal(wanted, dt)

    # three tests below are added based on what numpy does
    def test_complex(self):
        dt = np.min_scalar_type(0+0j)
        assert dt == np.dtype('complex64')

    def test_float(self):
        dt = np.min_scalar_type(0.1)
        assert dt == np.dtype('float16')

    def test_nonscalar(self):
        dt = np.min_scalar_type([0, 1, 2])
        assert dt == np.dtype('int64')


from numpy.core._internal import _dtype_from_pep3118

@pytest.mark.skip(reason='dont worry about buffer protocol')
class TestPEP3118Dtype:
    def _check(self, spec, wanted):
        dt = np.dtype(wanted)
        actual = _dtype_from_pep3118(spec)
        assert_equal(actual, dt,
                     err_msg="spec %r != dtype %r" % (spec, wanted))

    def test_native_padding(self):
        align = np.dtype('i').alignment
        for j in range(8):
            if j == 0:
                s = 'bi'
            else:
                s = 'b%dxi' % j
            self._check('@'+s, {'f0': ('i1', 0),
                                'f1': ('i', align*(1 + j//align))})
            self._check('='+s, {'f0': ('i1', 0),
                                'f1': ('i', 1+j)})

    def test_native_padding_2(self):
        # Native padding should work also for structs and sub-arrays
        self._check('x3T{xi}', {'f0': (({'f0': ('i', 4)}, (3,)), 4)})
        self._check('^x3T{xi}', {'f0': (({'f0': ('i', 1)}, (3,)), 1)})

    def test_trailing_padding(self):
        # Trailing padding should be included, *and*, the item size
        # should match the alignment if in aligned mode
        align = np.dtype('i').alignment
        size = np.dtype('i').itemsize

        def aligned(n):
            return align*(1 + (n-1)//align)

        base = dict(formats=['i'], names=['f0'])

        self._check('ix',    dict(itemsize=aligned(size + 1), **base))
        self._check('ixx',   dict(itemsize=aligned(size + 2), **base))
        self._check('ixxx',  dict(itemsize=aligned(size + 3), **base))
        self._check('ixxxx', dict(itemsize=aligned(size + 4), **base))
        self._check('i7x',   dict(itemsize=aligned(size + 7), **base))

        self._check('^ix',    dict(itemsize=size + 1, **base))
        self._check('^ixx',   dict(itemsize=size + 2, **base))
        self._check('^ixxx',  dict(itemsize=size + 3, **base))
        self._check('^ixxxx', dict(itemsize=size + 4, **base))
        self._check('^i7x',   dict(itemsize=size + 7, **base))

    def test_native_padding_3(self):
        dt = np.dtype(
                [('a', 'b'), ('b', 'i'),
                    ('sub', np.dtype('b,i')), ('c', 'i')],
                align=True)
        self._check("T{b:a:xxxi:b:T{b:f0:=i:f1:}:sub:xxxi:c:}", dt)

        dt = np.dtype(
                [('a', 'b'), ('b', 'i'), ('c', 'b'), ('d', 'b'),
                    ('e', 'b'), ('sub', np.dtype('b,i', align=True))])
        self._check("T{b:a:=i:b:b:c:b:d:b:e:T{b:f0:xxxi:f1:}:sub:}", dt)

    def test_padding_with_array_inside_struct(self):
        dt = np.dtype(
                [('a', 'b'), ('b', 'i'), ('c', 'b', (3,)),
                    ('d', 'i')],
                align=True)
        self._check("T{b:a:xxxi:b:3b:c:xi:d:}", dt)

    def test_byteorder_inside_struct(self):
        # The byte order after @T{=i} should be '=', not '@'.
        # Check this by noting the absence of native alignment.
        self._check('@T{^i}xi', {'f0': ({'f0': ('i', 0)}, 0),
                                 'f1': ('i', 5)})

    def test_intra_padding(self):
        # Natively aligned sub-arrays may require some internal padding
        align = np.dtype('i').alignment
        size = np.dtype('i').itemsize

        def aligned(n):
            return (align*(1 + (n-1)//align))

        self._check('(3)T{ix}', (dict(
            names=['f0'],
            formats=['i'],
            offsets=[0],
            itemsize=aligned(size + 1)
        ), (3,)))

    def test_char_vs_string(self):
        dt = np.dtype('c')
        self._check('c', dt)

        dt = np.dtype([('f0', 'S1', (4,)), ('f1', 'S4')])
        self._check('4c4s', dt)

    def test_field_order(self):
        # gh-9053 - previously, we relied on dictionary key order
        self._check("(0)I:a:f:b:", [('a', 'I', (0,)), ('b', 'f')])
        self._check("(0)I:b:f:a:", [('b', 'I', (0,)), ('a', 'f')])

    def test_unnamed_fields(self):
        self._check('ii',     [('f0', 'i'), ('f1', 'i')])
        self._check('ii:f0:', [('f1', 'i'), ('f0', 'i')])

        self._check('i', 'i')
        self._check('i:f0:', [('f0', 'i')])


@pytest.mark.skip(reason='dont worry about buffer protocol')
class TestNewBufferProtocol:
    """ Test PEP3118 buffers """

    def _check_roundtrip(self, obj):
        obj = np.asarray(obj)
        x = memoryview(obj)
        y = np.asarray(x)
        y2 = np.array(x)
        assert_(not y.flags.owndata)
        assert_(y2.flags.owndata)

        assert_equal(y.dtype, obj.dtype)
        assert_equal(y.shape, obj.shape)
        assert_array_equal(obj, y)

        assert_equal(y2.dtype, obj.dtype)
        assert_equal(y2.shape, obj.shape)
        assert_array_equal(obj, y2)

    def test_roundtrip(self):
        x = np.array([1, 2, 3, 4, 5], dtype='i4')
        self._check_roundtrip(x)

        x = np.array([[1, 2], [3, 4]], dtype=np.float64)
        self._check_roundtrip(x)

        x = np.zeros((3, 3, 3), dtype=np.float32)[:, 0,:]
        self._check_roundtrip(x)

        dt = [('a', 'b'),
              ('b', 'h'),
              ('c', 'i'),
              ('d', 'l'),
              ('dx', 'q'),
              ('e', 'B'),
              ('f', 'H'),
              ('g', 'I'),
              ('h', 'L'),
              ('hx', 'Q'),
              ('i', np.single),
              ('j', np.double),
              ('k', np.longdouble),
              ('ix', np.csingle),
              ('jx', np.cdouble),
              ('kx', np.clongdouble),
              ('l', 'S4'),
              ('m', 'U4'),
              ('n', 'V3'),
              ('o', '?'),
              ('p', np.half),
              ]
        x = np.array(
                [(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    b'aaaa', 'bbbb', b'xxx', True, 1.0)],
                dtype=dt)
        self._check_roundtrip(x)

        x = np.array(([[1, 2], [3, 4]],), dtype=[('a', (int, (2, 2)))])
        self._check_roundtrip(x)

        x = np.array([1, 2, 3], dtype='>i2')
        self._check_roundtrip(x)

        x = np.array([1, 2, 3], dtype='<i2')
        self._check_roundtrip(x)

        x = np.array([1, 2, 3], dtype='>i4')
        self._check_roundtrip(x)

        x = np.array([1, 2, 3], dtype='<i4')
        self._check_roundtrip(x)

        # check long long can be represented as non-native
        x = np.array([1, 2, 3], dtype='>q')
        self._check_roundtrip(x)

        # Native-only data types can be passed through the buffer interface
        # only in native byte order
        if sys.byteorder == 'little':
            x = np.array([1, 2, 3], dtype='>g')
            assert_raises(ValueError, self._check_roundtrip, x)
            x = np.array([1, 2, 3], dtype='<g')
            self._check_roundtrip(x)
        else:
            x = np.array([1, 2, 3], dtype='>g')
            self._check_roundtrip(x)
            x = np.array([1, 2, 3], dtype='<g')
            assert_raises(ValueError, self._check_roundtrip, x)

    def test_roundtrip_half(self):
        half_list = [
            1.0,
            -2.0,
            6.5504 * 10**4,  # (max half precision)
            2**-14,  # ~= 6.10352 * 10**-5 (minimum positive normal)
            2**-24,  # ~= 5.96046 * 10**-8 (minimum strictly positive subnormal)
            0.0,
            -0.0,
            float('+inf'),
            float('-inf'),
            0.333251953125,  # ~= 1/3
        ]

        x = np.array(half_list, dtype='>e')
        self._check_roundtrip(x)
        x = np.array(half_list, dtype='<e')
        self._check_roundtrip(x)

    def test_roundtrip_single_types(self):
        for typ in np.sctypeDict.values():
            dtype = np.dtype(typ)

            if dtype.char in 'Mm':
                # datetimes cannot be used in buffers
                continue
            if dtype.char == 'V':
                # skip void
                continue

            x = np.zeros(4, dtype=dtype)
            self._check_roundtrip(x)

            if dtype.char not in 'qQgG':
                dt = dtype.newbyteorder('<')
                x = np.zeros(4, dtype=dt)
                self._check_roundtrip(x)

                dt = dtype.newbyteorder('>')
                x = np.zeros(4, dtype=dt)
                self._check_roundtrip(x)

    def test_roundtrip_scalar(self):
        # Issue #4015.
        self._check_roundtrip(0)

    def test_export_simple_1d(self):
        x = np.array([1, 2, 3, 4, 5], dtype='i')
        y = memoryview(x)
        assert_equal(y.format, 'i')
        assert_equal(y.shape, (5,))
        assert_equal(y.ndim, 1)
        assert_equal(y.strides, (4,))
        assert_equal(y.suboffsets, ())
        assert_equal(y.itemsize, 4)

    def test_export_simple_nd(self):
        x = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = memoryview(x)
        assert_equal(y.format, 'd')
        assert_equal(y.shape, (2, 2))
        assert_equal(y.ndim, 2)
        assert_equal(y.strides, (16, 8))
        assert_equal(y.suboffsets, ())
        assert_equal(y.itemsize, 8)

    def test_export_discontiguous(self):
        x = np.zeros((3, 3, 3), dtype=np.float32)[:, 0,:]
        y = memoryview(x)
        assert_equal(y.format, 'f')
        assert_equal(y.shape, (3, 3))
        assert_equal(y.ndim, 2)
        assert_equal(y.strides, (36, 4))
        assert_equal(y.suboffsets, ())
        assert_equal(y.itemsize, 4)

    def test_export_record(self):
        dt = [('a', 'b'),
              ('b', 'h'),
              ('c', 'i'),
              ('d', 'l'),
              ('dx', 'q'),
              ('e', 'B'),
              ('f', 'H'),
              ('g', 'I'),
              ('h', 'L'),
              ('hx', 'Q'),
              ('i', np.single),
              ('j', np.double),
              ('k', np.longdouble),
              ('ix', np.csingle),
              ('jx', np.cdouble),
              ('kx', np.clongdouble),
              ('l', 'S4'),
              ('m', 'U4'),
              ('n', 'V3'),
              ('o', '?'),
              ('p', np.half),
              ]
        x = np.array(
                [(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    b'aaaa', 'bbbb', b'   ', True, 1.0)],
                dtype=dt)
        y = memoryview(x)
        assert_equal(y.shape, (1,))
        assert_equal(y.ndim, 1)
        assert_equal(y.suboffsets, ())

        sz = sum([np.dtype(b).itemsize for a, b in dt])
        if np.dtype('l').itemsize == 4:
            assert_equal(y.format, 'T{b:a:=h:b:i:c:l:d:q:dx:B:e:@H:f:=I:g:L:h:Q:hx:f:i:d:j:^g:k:=Zf:ix:Zd:jx:^Zg:kx:4s:l:=4w:m:3x:n:?:o:@e:p:}')
        else:
            assert_equal(y.format, 'T{b:a:=h:b:i:c:q:d:q:dx:B:e:@H:f:=I:g:Q:h:Q:hx:f:i:d:j:^g:k:=Zf:ix:Zd:jx:^Zg:kx:4s:l:=4w:m:3x:n:?:o:@e:p:}')
        # Cannot test if NPY_RELAXED_STRIDES_DEBUG changes the strides
        if not (np.ones(1).strides[0] == np.iinfo(np.intp).max):
            assert_equal(y.strides, (sz,))
        assert_equal(y.itemsize, sz)

    def test_export_subarray(self):
        x = np.array(([[1, 2], [3, 4]],), dtype=[('a', ('i', (2, 2)))])
        y = memoryview(x)
        assert_equal(y.format, 'T{(2,2)i:a:}')
        assert_equal(y.shape, ())
        assert_equal(y.ndim, 0)
        assert_equal(y.strides, ())
        assert_equal(y.suboffsets, ())
        assert_equal(y.itemsize, 16)

    def test_export_endian(self):
        x = np.array([1, 2, 3], dtype='>i')
        y = memoryview(x)
        if sys.byteorder == 'little':
            assert_equal(y.format, '>i')
        else:
            assert_equal(y.format, 'i')

        x = np.array([1, 2, 3], dtype='<i')
        y = memoryview(x)
        if sys.byteorder == 'little':
            assert_equal(y.format, 'i')
        else:
            assert_equal(y.format, '<i')

    def test_export_flags(self):
        # Check SIMPLE flag, see also gh-3613 (exception should be BufferError)
        assert_raises(ValueError,
                      _multiarray_tests.get_buffer_info,
                       np.arange(5)[::2], ('SIMPLE',))

    def test_padding(self):
        for j in range(8):
            x = np.array([(1,), (2,)], dtype={'f0': (int, j)})
            self._check_roundtrip(x)

    def test_reference_leak(self):
        if HAS_REFCOUNT:
            count_1 = sys.getrefcount(np.core._internal)
        a = np.zeros(4)
        b = memoryview(a)
        c = np.asarray(b)
        if HAS_REFCOUNT:
            count_2 = sys.getrefcount(np.core._internal)
            assert_equal(count_1, count_2)
        del c  # avoid pyflakes unused variable warning.

    def test_padded_struct_array(self):
        dt1 = np.dtype(
                [('a', 'b'), ('b', 'i'), ('sub', np.dtype('b,i')), ('c', 'i')],
                align=True)
        x1 = np.arange(dt1.itemsize, dtype=np.int8).view(dt1)
        self._check_roundtrip(x1)

        dt2 = np.dtype(
                [('a', 'b'), ('b', 'i'), ('c', 'b', (3,)), ('d', 'i')],
                align=True)
        x2 = np.arange(dt2.itemsize, dtype=np.int8).view(dt2)
        self._check_roundtrip(x2)

        dt3 = np.dtype(
                [('a', 'b'), ('b', 'i'), ('c', 'b'), ('d', 'b'),
                    ('e', 'b'), ('sub', np.dtype('b,i', align=True))])
        x3 = np.arange(dt3.itemsize, dtype=np.int8).view(dt3)
        self._check_roundtrip(x3)

    @pytest.mark.valgrind_error(reason="leaks buffer info cache temporarily.")
    def test_relaxed_strides(self, c=np.ones((1, 10, 10), dtype='i8')):
        # Note: c defined as parameter so that it is persistent and leak
        # checks will notice gh-16934 (buffer info cache leak).
        c.strides = (-1, 80, 8)  # strides need to be fixed at export

        assert_(memoryview(c).strides == (800, 80, 8))

        # Writing C-contiguous data to a BytesIO buffer should work
        fd = io.BytesIO()
        fd.write(c.data)

        fortran = c.T
        assert_(memoryview(fortran).strides == (8, 80, 800))

        arr = np.ones((1, 10))
        if arr.flags.f_contiguous:
            shape, strides = _multiarray_tests.get_buffer_info(
                    arr, ['F_CONTIGUOUS'])
            assert_(strides[0] == 8)
            arr = np.ones((10, 1), order='F')
            shape, strides = _multiarray_tests.get_buffer_info(
                    arr, ['C_CONTIGUOUS'])
            assert_(strides[-1] == 8)

    @pytest.mark.valgrind_error(reason="leaks buffer info cache temporarily.")
    @pytest.mark.skipif(not np.ones((10, 1), order="C").flags.f_contiguous,
            reason="Test is unnecessary (but fails) without relaxed strides.")
    def test_relaxed_strides_buffer_info_leak(self, arr=np.ones((1, 10))):
        """Test that alternating export of C- and F-order buffers from
        an array which is both C- and F-order when relaxed strides is
        active works.
        This test defines array in the signature to ensure leaking more
        references every time the test is run (catching the leak with
        pytest-leaks).
        """
        for i in range(10):
            _, s = _multiarray_tests.get_buffer_info(arr, ['F_CONTIGUOUS'])
            assert s == (8, 8)
            _, s = _multiarray_tests.get_buffer_info(arr, ['C_CONTIGUOUS'])
            assert s == (80, 8)

    def test_out_of_order_fields(self):
        dt = np.dtype(dict(
            formats=['<i4', '<i4'],
            names=['one', 'two'],
            offsets=[4, 0],
            itemsize=8
        ))

        # overlapping fields cannot be represented by PEP3118
        arr = np.empty(1, dt)
        with assert_raises(ValueError):
            memoryview(arr)

    def test_max_dims(self):
        a = np.ones((1,) * 32)
        self._check_roundtrip(a)

    @pytest.mark.slow
    def test_error_too_many_dims(self):
        def make_ctype(shape, scalar_type):
            t = scalar_type
            for dim in shape[::-1]:
                t = dim * t
            return t

        # construct a memoryview with 33 dimensions
        c_u8_33d = make_ctype((1,)*33, ctypes.c_uint8)
        m = memoryview(c_u8_33d())
        assert_equal(m.ndim, 33)

        assert_raises_regex(
            RuntimeError, "ndim",
            np.array, m)

        # The above seems to create some deep cycles, clean them up for
        # easier reference count debugging:
        del c_u8_33d, m
        for i in range(33):
            if gc.collect() == 0:
                break

    def test_error_pointer_type(self):
        # gh-6741
        m = memoryview(ctypes.pointer(ctypes.c_uint8()))
        assert_('&' in m.format)

        assert_raises_regex(
            ValueError, "format string",
            np.array, m)

    def test_error_message_unsupported(self):
        # wchar has no corresponding numpy type - if this changes in future, we
        # need a better way to construct an invalid memoryview format.
        t = ctypes.c_wchar * 4
        with assert_raises(ValueError) as cm:
            np.array(t())

        exc = cm.exception
        with assert_raises_regex(
            NotImplementedError,
            r"Unrepresentable .* 'u' \(UCS-2 strings\)"
        ):
            raise exc.__cause__

    def test_ctypes_integer_via_memoryview(self):
        # gh-11150, due to bpo-10746
        for c_integer in {ctypes.c_int, ctypes.c_long, ctypes.c_longlong}:
            value = c_integer(42)
            with warnings.catch_warnings(record=True):
                warnings.filterwarnings('always', r'.*\bctypes\b', RuntimeWarning)
                np.asarray(value)

    def test_ctypes_struct_via_memoryview(self):
        # gh-10528
        class foo(ctypes.Structure):
            _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint32)]
        f = foo(a=1, b=2)

        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('always', r'.*\bctypes\b', RuntimeWarning)
            arr = np.asarray(f)

        assert_equal(arr['a'], 1)
        assert_equal(arr['b'], 2)
        f.a = 3
        assert_equal(arr['a'], 3)

    @pytest.mark.parametrize("obj", [np.ones(3)[()]])
    def test_error_if_stored_buffer_info_is_corrupted(self, obj):
        """
        If a user extends a NumPy array before 1.20 and then runs it
        on NumPy 1.20+. A C-subclassed array might in theory modify
        the new buffer-info field. This checks that an error is raised
        if this happens (for buffer export), an error is written on delete.
        This is a sanity check to help users transition to safe code, it
        may be deleted at any point.
        """
        # corrupt buffer info:
        _multiarray_tests.corrupt_or_fix_bufferinfo(obj)
        name = type(obj)
        with pytest.raises(RuntimeError,
                    match=f".*{name} appears to be C subclassed"):
            memoryview(obj)
        # Fix buffer info again before we delete (or we lose the memory)
        _multiarray_tests.corrupt_or_fix_bufferinfo(obj)

    def test_no_suboffsets(self):
        try:
            import _testbuffer
        except ImportError:
            raise pytest.skip("_testbuffer is not available")

        for shape in [(2, 3), (2, 3, 4)]:
            data = list(range(np.prod(shape)))
            buffer = _testbuffer.ndarray(data, shape, format='i',
                                         flags=_testbuffer.ND_PIL)
            msg = "NumPy currently does not support.*suboffsets"
            with pytest.raises(BufferError, match=msg):
                np.asarray(buffer)
            with pytest.raises(BufferError, match=msg):
                np.asarray([buffer])

            # Also check (unrelated and more limited but similar) frombuffer:
            with pytest.raises(BufferError):
                np.frombuffer(buffer)


@pytest.mark.xfail(reason='TODO')
class TestArrayCreationCopyArgument(object):

    class RaiseOnBool:

        def __bool__(self):
            raise ValueError

    # true_vals = [True, np._CopyMode.ALWAYS, np.True_]
    # false_vals = [False, np._CopyMode.IF_NEEDED, np.False_]
    true_vals = [True, 1, np.True_]
    false_vals = [False, 0, np.False_]


    def test_scalars(self):
        # Test both numpy and python scalars
        for dtype in np.typecodes["All"]:
            arr = np.zeros((), dtype=dtype)
            scalar = arr[()]
            pyscalar = arr.item(0)

            # Test never-copy raises error:
            assert_raises(ValueError, np.array, scalar,
                            copy=np._CopyMode.NEVER)
            assert_raises(ValueError, np.array, pyscalar,
                            copy=np._CopyMode.NEVER)
            assert_raises(ValueError, np.array, pyscalar,
                            copy=self.RaiseOnBool())
            assert_raises(ValueError, _multiarray_tests.npy_ensurenocopy,
                            [1])
            # Casting with a dtype (to unsigned integers) can be special:
            with pytest.raises(ValueError):
                np.array(pyscalar, dtype=np.int64, copy=np._CopyMode.NEVER)

    def test_compatible_cast(self):

        # Some types are compatible even though they are different, no
        # copy is necessary for them. This is mostly true for some integers
        def int_types(byteswap=False):
            int_types = (np.typecodes["Integer"] +
                         np.typecodes["UnsignedInteger"])
            for int_type in int_types:
                yield np.dtype(int_type)
                if byteswap:
                    yield np.dtype(int_type).newbyteorder()

        for int1 in int_types():
            for int2 in int_types(True):
                arr = np.arange(10, dtype=int1)

                for copy in self.true_vals:
                    res = np.array(arr, copy=copy, dtype=int2)
                    assert res is not arr and res.flags.owndata
                    assert_array_equal(res, arr)

                if int1 == int2:
                    # Casting is not necessary, base check is sufficient here
                    for copy in self.false_vals:
                        res = np.array(arr, copy=copy, dtype=int2)
                        assert res is arr or res.base is arr

                    res = np.array(arr,
                                   copy=np._CopyMode.NEVER,
                                   dtype=int2)
                    assert res is arr or res.base is arr

                else:
                    # Casting is necessary, assert copy works:
                    for copy in self.false_vals:
                        res = np.array(arr, copy=copy, dtype=int2)
                        assert res is not arr and res.flags.owndata
                        assert_array_equal(res, arr)

                    assert_raises(ValueError, np.array,
                                  arr, copy=np._CopyMode.NEVER,
                                  dtype=int2)
                    assert_raises(ValueError, np.array,
                                  arr, copy=None,
                                  dtype=int2)

    def test_buffer_interface(self):

        # Buffer interface gives direct memory access (no copy)
        arr = np.arange(10)
        view = memoryview(arr)

        # Checking bases is a bit tricky since numpy creates another
        # memoryview, so use may_share_memory.
        for copy in self.true_vals:
            res = np.array(view, copy=copy)
            assert not np.may_share_memory(arr, res)
        for copy in self.false_vals:
            res = np.array(view, copy=copy)
            assert np.may_share_memory(arr, res)
        res = np.array(view, copy=np._CopyMode.NEVER)
        assert np.may_share_memory(arr, res)

    def test_array_interfaces(self):
        # Array interface gives direct memory access (much like a memoryview)
        base_arr = np.arange(10)

        class ArrayLike:
            __array_interface__ = base_arr.__array_interface__

        arr = ArrayLike()

        for copy, val in [(True, None), (np._CopyMode.ALWAYS, None),
                          (False, arr), (np._CopyMode.IF_NEEDED, arr),
                          (np._CopyMode.NEVER, arr)]:
            res = np.array(arr, copy=copy)
            assert res.base is val

    def test___array__(self):
        base_arr = np.arange(10)

        class ArrayLike:
            def __array__(self):
                # __array__ should return a copy, numpy cannot know this
                # however.
                return base_arr

        arr = ArrayLike()

        for copy in self.true_vals:
            res = np.array(arr, copy=copy)
            assert_array_equal(res, base_arr)
            # An additional copy is currently forced by numpy in this case,
            # you could argue, numpy does not trust the ArrayLike. This
            # may be open for change:
            assert res is not base_arr

        for copy in self.false_vals:
            res = np.array(arr, copy=False)
            assert_array_equal(res, base_arr)
            assert res is base_arr  # numpy trusts the ArrayLike

        with pytest.raises(ValueError):
            np.array(arr, copy=np._CopyMode.NEVER)

    @pytest.mark.parametrize(
            "arr", [np.ones(()), np.arange(81).reshape((9, 9))])
    @pytest.mark.parametrize("order1", ["C", "F", None])
    @pytest.mark.parametrize("order2", ["C", "F", "A", "K"])
    def test_order_mismatch(self, arr, order1, order2):
        # The order is the main (python side) reason that can cause
        # a never-copy to fail.
        # Prepare C-order, F-order and non-contiguous arrays:
        arr = arr.copy(order1)
        if order1 == "C":
            assert arr.flags.c_contiguous
        elif order1 == "F":
            assert arr.flags.f_contiguous
        elif arr.ndim != 0:
            # Make array non-contiguous
            arr = arr[::2, ::2]
            assert not arr.flags.forc

        # Whether a copy is necessary depends on the order of arr:
        if order2 == "C":
            no_copy_necessary = arr.flags.c_contiguous
        elif order2 == "F":
            no_copy_necessary = arr.flags.f_contiguous
        else:
            # Keeporder and Anyorder are OK with non-contiguous output.
            # This is not consistent with the `astype` behaviour which
            # enforces contiguity for "A". It is probably historic from when
            # "K" did not exist.
            no_copy_necessary = True

        # Test it for both the array and a memoryview
        for view in [arr, memoryview(arr)]:
            for copy in self.true_vals:
                res = np.array(view, copy=copy, order=order2)
                assert res is not arr and res.flags.owndata
                assert_array_equal(arr, res)

            if no_copy_necessary:
                for copy in self.false_vals:
                    res = np.array(view, copy=copy, order=order2)
                    # res.base.obj refers to the memoryview
                    if not IS_PYPY:
                        assert res is arr or res.base.obj is arr

                res = np.array(view, copy=np._CopyMode.NEVER,
                               order=order2)
                if not IS_PYPY:
                    assert res is arr or res.base.obj is arr
            else:
                for copy in self.false_vals:
                    res = np.array(arr, copy=copy, order=order2)
                    assert_array_equal(arr, res)
                assert_raises(ValueError, np.array,
                              view, copy=np._CopyMode.NEVER,
                              order=order2)
                assert_raises(ValueError, np.array,
                              view, copy=None,
                              order=order2)

    def test_striding_not_ok(self):
        arr = np.array([[1, 2, 4], [3, 4, 5]])
        assert_raises(ValueError, np.array,
                      arr.T, copy=np._CopyMode.NEVER,
                      order='C')
        assert_raises(ValueError, np.array,
                      arr.T, copy=np._CopyMode.NEVER,
                      order='C', dtype=np.int64)
        assert_raises(ValueError, np.array,
                      arr, copy=np._CopyMode.NEVER,
                      order='F')
        assert_raises(ValueError, np.array,
                      arr, copy=np._CopyMode.NEVER,
                      order='F', dtype=np.int64)


@pytest.mark.xfail(reason='TODO')
class TestArrayAttributeDeletion:

    def test_multiarray_writable_attributes_deletion(self):
        # ticket #2046, should not seqfault, raise AttributeError
        a = np.ones(2)
        attr = ['shape', 'strides', 'data', 'dtype', 'real', 'imag', 'flat']
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Assigning the 'data' attribute")
            for s in attr:
                assert_raises(AttributeError, delattr, a, s)

    def test_multiarray_not_writable_attributes_deletion(self):
        a = np.ones(2)
        attr = ["ndim", "flags", "itemsize", "size", "nbytes", "base",
                "ctypes", "T", "__array_interface__", "__array_struct__",
                "__array_priority__", "__array_finalize__"]
        for s in attr:
            assert_raises(AttributeError, delattr, a, s)

    def test_multiarray_flags_writable_attribute_deletion(self):
        a = np.ones(2).flags
        attr = ['writebackifcopy', 'updateifcopy', 'aligned', 'writeable']
        for s in attr:
            assert_raises(AttributeError, delattr, a, s)

    def test_multiarray_flags_not_writable_attribute_deletion(self):
        a = np.ones(2).flags
        attr = ["contiguous", "c_contiguous", "f_contiguous", "fortran",
                "owndata", "fnc", "forc", "behaved", "carray", "farray",
                "num"]
        for s in attr:
            assert_raises(AttributeError, delattr, a, s)


@pytest.mark.xfail(reason='TODO')
class TestArrayInterface():
    class Foo:
        def __init__(self, value):
            self.value = value
            self.iface = {'typestr': 'f8'}

        def __float__(self):
            return float(self.value)

        @property
        def __array_interface__(self):
            return self.iface


    f = Foo(0.5)

    @pytest.mark.parametrize('val, iface, expected', [
        (f, {}, 0.5),
        ([f], {}, [0.5]),
        ([f, f], {}, [0.5, 0.5]),
        (f, {'shape': ()}, 0.5),
        (f, {'shape': None}, TypeError),
        (f, {'shape': (1, 1)}, [[0.5]]),
        (f, {'shape': (2,)}, ValueError),
        (f, {'strides': ()}, 0.5),
        (f, {'strides': (2,)}, ValueError),
        (f, {'strides': 16}, TypeError),
        ])
    def test_scalar_interface(self, val, iface, expected):
        # Test scalar coercion within the array interface
        self.f.iface = {'typestr': 'f8'}
        self.f.iface.update(iface)
        if HAS_REFCOUNT:
            pre_cnt = sys.getrefcount(np.dtype('f8'))
        if isinstance(expected, type):
            assert_raises(expected, np.array, val)
        else:
            result = np.array(val)
            assert_equal(np.array(val), expected)
            assert result.dtype == 'f8'
            del result
        if HAS_REFCOUNT:
            post_cnt = sys.getrefcount(np.dtype('f8'))
            assert_equal(pre_cnt, post_cnt)


@pytest.mark.xfail(reason='TODO')
def test_flat_element_deletion():
    it = np.ones(3).flat
    try:
        del it[1]
        del it[1:2]
    except TypeError:
        pass
    except Exception:
        raise AssertionError


@pytest.mark.xfail(reason='TODO')
class TestMapIter:
    def test_mapiter(self):
        # The actual tests are within the C code in
        # multiarray/_multiarray_tests.c.src

        a = np.arange(12).reshape((3, 4)).astype(float)
        index = ([1, 1, 2, 0],
                 [0, 0, 2, 3])
        vals = [50, 50, 30, 16]

        _multiarray_tests.test_inplace_increment(a, index, vals)
        assert_equal(a, [[0.00, 1., 2.0, 19.],
                         [104., 5., 6.0, 7.0],
                         [8.00, 9., 40., 11.]])

        b = np.arange(6).astype(float)
        index = (np.array([1, 2, 0]),)
        vals = [50, 4, 100.1]
        _multiarray_tests.test_inplace_increment(b, index, vals)
        assert_equal(b, [100.1,  51.,   6.,   3.,   4.,   5.])


@pytest.mark.xfail(reason='TODO')
class TestAsCArray:
    def test_1darray(self):
        array = np.arange(24, dtype=np.double)
        from_c = _multiarray_tests.test_as_c_array(array, 3)
        assert_equal(array[3], from_c)

    def test_2darray(self):
        array = np.arange(24, dtype=np.double).reshape(3, 8)
        from_c = _multiarray_tests.test_as_c_array(array, 2, 4)
        assert_equal(array[2, 4], from_c)

    def test_3darray(self):
        array = np.arange(24, dtype=np.double).reshape(2, 3, 4)
        from_c = _multiarray_tests.test_as_c_array(array, 1, 2, 3)
        assert_equal(array[1, 2, 3], from_c)


class TestConversion:
    def test_array_scalar_relational_operation(self):
        # All integer
        for dt1 in np.typecodes['AllInteger']:
            assert_(1 > np.array(0, dtype=dt1), "type %s failed" % (dt1,))
            assert_(not 1 < np.array(0, dtype=dt1), "type %s failed" % (dt1,))

            for dt2 in np.typecodes['AllInteger']:
                assert_(np.array(1, dtype=dt1) > np.array(0, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))
                assert_(not np.array(1, dtype=dt1) < np.array(0, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))

        # Unsigned integers
        for dt1 in 'B':
            assert_(-1 < np.array(1, dtype=dt1), "type %s failed" % (dt1,))
            assert_(not -1 > np.array(1, dtype=dt1), "type %s failed" % (dt1,))
            assert_(-1 != np.array(1, dtype=dt1), "type %s failed" % (dt1,))

            # Unsigned vs signed
            for dt2 in 'bhil':
                assert_(np.array(1, dtype=dt1) > np.array(-1, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))
                assert_(not np.array(1, dtype=dt1) < np.array(-1, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))
                assert_(np.array(1, dtype=dt1) != np.array(-1, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))

        # Signed integers and floats
        for dt1 in 'bhl' + np.typecodes['Float']:
            assert_(1 > np.array(-1, dtype=dt1), "type %s failed" % (dt1,))
            assert_(not 1 < np.array(-1, dtype=dt1), "type %s failed" % (dt1,))
            assert_(-1 == np.array(-1, dtype=dt1), "type %s failed" % (dt1,))

            for dt2 in 'bhl' + np.typecodes['Float']:
                assert_(np.array(1, dtype=dt1) > np.array(-1, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))
                assert_(not np.array(1, dtype=dt1) < np.array(-1, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))
                assert_(np.array(-1, dtype=dt1) == np.array(-1, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))

    @pytest.mark.skip(reason="object arrays")
    def test_to_bool_scalar(self):
        assert_equal(bool(np.array([False])), False)
        assert_equal(bool(np.array([True])), True)
        assert_equal(bool(np.array([[42]])), True)
        assert_raises(ValueError, bool, np.array([1, 2]))

        class NotConvertible:
            def __bool__(self):
                raise NotImplementedError

        assert_raises(NotImplementedError, bool, np.array(NotConvertible()))
        assert_raises(NotImplementedError, bool, np.array([NotConvertible()]))
        if IS_PYSTON:
            pytest.skip("Pyston disables recursion checking")

        self_containing = np.array([None])
        self_containing[0] = self_containing

        Error = RecursionError

        assert_raises(Error, bool, self_containing)  # previously stack overflow
        self_containing[0] = None  # resolve circular reference

    def test_to_int_scalar(self):
        # gh-9972 means that these aren't always the same
        int_funcs = (int, lambda x: x.__int__())
        for int_func in int_funcs:
            assert_equal(int_func(np.array(0)), 0)
            assert_equal(int_func(np.array([1])), 1)
            assert_equal(int_func(np.array([[42]])), 42)
            assert_raises((ValueError, TypeError), int_func, np.array([1, 2]))

    @pytest.mark.skip(reason="object arrays")
    def test_to_int_scalar_2(self):
        int_funcs = (int, lambda x: x.__int__())
        for int_func in int_funcs:
            # gh-9972
            assert_equal(4, int_func(np.array('4')))
            assert_equal(5, int_func(np.bytes_(b'5')))
            assert_equal(6, int_func(np.unicode_('6')))

            # The delegation of int() to __trunc__ was deprecated in
            # Python 3.11.
            if sys.version_info < (3, 11):
                class HasTrunc:
                    def __trunc__(self):
                        return 3
                assert_equal(3, int_func(np.array(HasTrunc())))
                assert_equal(3, int_func(np.array([HasTrunc()])))
            else:
                pass

            class NotConvertible:
                def __int__(self):
                    raise NotImplementedError
            assert_raises(NotImplementedError,
                int_func, np.array(NotConvertible()))
            assert_raises(NotImplementedError,
                int_func, np.array([NotConvertible()]))


class TestWhere:
    def test_basic(self):
        dts = [bool, np.int16, np.int32, np.int64, np.double, np.complex128]
        for dt in dts:
            c = np.ones(53, dtype=bool)
            assert_equal(np.where( c, dt(0), dt(1)), dt(0))
            assert_equal(np.where(~c, dt(0), dt(1)), dt(1))
            assert_equal(np.where(True, dt(0), dt(1)), dt(0))
            assert_equal(np.where(False, dt(0), dt(1)), dt(1))
            d = np.ones_like(c).astype(dt)
            e = np.zeros_like(d)
            r = d.astype(dt)
            c[7] = False
            r[7] = e[7]
            assert_equal(np.where(c, e, e), e)
            assert_equal(np.where(c, d, e), r)
            assert_equal(np.where(c, d, e[0]), r)
            assert_equal(np.where(c, d[0], e), r)
            assert_equal(np.where(c[::2], d[::2], e[::2]), r[::2])
            assert_equal(np.where(c[1::2], d[1::2], e[1::2]), r[1::2])
            assert_equal(np.where(c[::3], d[::3], e[::3]), r[::3])
            assert_equal(np.where(c[1::3], d[1::3], e[1::3]), r[1::3])
          #  assert_equal(np.where(c[::-2], d[::-2], e[::-2]), r[::-2])
          #  assert_equal(np.where(c[::-3], d[::-3], e[::-3]), r[::-3])
          #  assert_equal(np.where(c[1::-3], d[1::-3], e[1::-3]), r[1::-3])

    def test_exotic(self):
        # zero sized
        m = np.array([], dtype=bool).reshape(0, 3)
        b = np.array([], dtype=np.float64).reshape(0, 3)
        assert_array_equal(np.where(m, 0, b), np.array([]).reshape(0, 3))

    @pytest.mark.skip(reason='object arrays')
    def test_exotic_2(self):
        # object cast
        d = np.array([-1.34, -0.16, -0.54, -0.31, -0.08, -0.95, 0.000, 0.313,
                      0.547, -0.18, 0.876, 0.236, 1.969, 0.310, 0.699, 1.013,
                      1.267, 0.229, -1.39, 0.487])
        nan = float('NaN')
        e = np.array(['5z', '0l', nan, 'Wz', nan, nan, 'Xq', 'cs', nan, nan,
                     'QN', nan, nan, 'Fd', nan, nan, 'kp', nan, '36', 'i1'],
                     dtype=object)
        m = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1,
                      0, 1, 1, 0, 1, 1, 0, 1, 0, 0], dtype=bool)

        r = e[:]
        r[np.where(m)] = d[np.where(m)]
        assert_array_equal(np.where(m, d, e), r)

        r = e[:]
        r[np.where(~m)] = d[np.where(~m)]
        assert_array_equal(np.where(m, e, d), r)

        assert_array_equal(np.where(m, e, e), e)

        # minimal dtype result with NaN scalar (e.g required by pandas)
        d = np.array([1., 2.], dtype=np.float32)
        e = float('NaN')
        assert_equal(np.where(True, d, e).dtype, np.float32)
        e = float('Infinity')
        assert_equal(np.where(True, d, e).dtype, np.float32)
        e = float('-Infinity')
        assert_equal(np.where(True, d, e).dtype, np.float32)
        # also check upcast
        e = float(1e150)
        assert_equal(np.where(True, d, e).dtype, np.float64)

    def test_ndim(self):
        c = [True, False]
        a = np.zeros((2, 25))
        b = np.ones((2, 25))
        r = np.where(np.array(c)[:,np.newaxis], a, b)
        assert_array_equal(r[0], a[0])
        assert_array_equal(r[1], b[0])

        a = a.T
        b = b.T
        r = np.where(c, a, b)
        assert_array_equal(r[:,0], a[:,0])
        assert_array_equal(r[:,1], b[:,0])

    def test_dtype_mix(self):
        c = np.array([False, True, False, False, False, False, True, False,
                     False, False, True, False])
        a = np.uint8(1)
        b = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.],
                      dtype=np.float64)
        r = np.array([5., 1., 3., 2., -1., -4., 1., -10., 10., 1., 1., 3.],
                     dtype=np.float64)
        assert_equal(np.where(c, a, b), r)

        a = a.astype(np.float32)
        b = b.astype(np.int64)
        assert_equal(np.where(c, a, b), r)

        # non bool mask
        c = c.astype(int)
        c[c != 0] = 34242324
        assert_equal(np.where(c, a, b), r)
        # invert
        tmpmask = c != 0
        c[c == 0] = 41247212
        c[tmpmask] = 0
        assert_equal(np.where(c, b, a), r)

    @pytest.mark.skip(reason='endianness')
    def test_foreign(self):
        c = np.array([False, True, False, False, False, False, True, False,
                     False, False, True, False])
        r = np.array([5., 1., 3., 2., -1., -4., 1., -10., 10., 1., 1., 3.],
                     dtype=np.float64)
        a = np.ones(1, dtype='>i4')
        b = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.],
                     dtype=np.float64)
        assert_equal(np.where(c, a, b), r)

        b = b.astype('>f8')
        assert_equal(np.where(c, a, b), r)

        a = a.astype('<i4')
        assert_equal(np.where(c, a, b), r)

        c = c.astype('>i4')
        assert_equal(np.where(c, a, b), r)

    def test_error(self):
        c = [True, True]
        a = np.ones((4, 5))
        b = np.ones((5, 5))
        assert_raises((RuntimeError, ValueError), np.where, c, a, a)
        assert_raises((RuntimeError, ValueError), np.where, c[0], a, b)

    def test_empty_result(self):
        # pass empty where result through an assignment which reads the data of
        # empty arrays, error detectable with valgrind, see gh-8922
        x = np.zeros((1, 1))
        ibad = np.vstack(np.where(x == 99.))
        assert_array_equal(ibad,
                           np.atleast_2d(np.array([[],[]], dtype=np.intp)))

    def test_largedim(self):
        # invalid read regression gh-9304
        shape = [10, 2, 3, 4, 5, 6]
        np.random.seed(2)
        array = np.random.rand(*shape)

        for i in range(10):
            benchmark = array.nonzero()
            result = array.nonzero()
            assert_array_equal(benchmark, result)

    def test_kwargs(self):
        a = np.zeros(1)
        with assert_raises(TypeError):
            np.where(a, x=a, y=a)


if not IS_PYPY:
    # sys.getsizeof() is not valid on PyPy
    @pytest.mark.xfail(reason='TODO')
    class TestSizeOf:

        def test_empty_array(self):
            x = np.array([])
            assert_(sys.getsizeof(x) > 0)

        def check_array(self, dtype):
            elem_size = dtype(0).itemsize

            for length in [10, 50, 100, 500]:
                x = np.arange(length, dtype=dtype)
                assert_(sys.getsizeof(x) > length * elem_size)

        def test_array_int32(self):
            self.check_array(np.int32)

        def test_array_int64(self):
            self.check_array(np.int64)

        def test_array_float32(self):
            self.check_array(np.float32)

        def test_array_float64(self):
            self.check_array(np.float64)

        def test_view(self):
            d = np.ones(100)
            assert_(sys.getsizeof(d[...]) < sys.getsizeof(d))

        def test_reshape(self):
            d = np.ones(100)
            assert_(sys.getsizeof(d) < sys.getsizeof(d.reshape(100, 1, 1).copy()))

        @_no_tracing
        def test_resize(self):
            d = np.ones(100)
            old = sys.getsizeof(d)
            d.resize(50)
            assert_(old > sys.getsizeof(d))
            d.resize(150)
            assert_(old < sys.getsizeof(d))

        def test_error(self):
            d = np.ones(100)
            assert_raises(TypeError, d.__sizeof__, "a")


class TestHashing:

    def test_arrays_not_hashable(self):
        x = np.ones(3)
        assert_raises(TypeError, hash, x)

    def test_collections_hashable(self):
        x = np.array([])
        assert_(not isinstance(x, collections.abc.Hashable))




@pytest.mark.xfail(reason='TODO')
class TestFormat:

    def test_0d(self):
        a = np.array(np.pi)
        assert_equal('{:0.3g}'.format(a), '3.14')
        assert_equal('{:0.3g}'.format(a[()]), '3.14')

    def test_1d_no_format(self):
        a = np.array([np.pi])
        assert_equal('{}'.format(a), str(a))

    def test_1d_format(self):
        # until gh-5543, ensure that the behaviour matches what it used to be
        a = np.array([np.pi])
        assert_raises(TypeError, '{:30}'.format, a)

from numpy.testing import IS_PYPY


class TestWritebackIfCopy:
    # all these tests use the WRITEBACKIFCOPY mechanism
    def test_argmax_with_out(self):
        mat = np.eye(5)
        out = np.empty(5, dtype='i2')
        res = np.argmax(mat, 0, out=out)
        assert_equal(res, range(5))

    def test_argmin_with_out(self):
        mat = -np.eye(5)
        out = np.empty(5, dtype='i2')
        res = np.argmin(mat, 0, out=out)
        assert_equal(res, range(5))

    @pytest.mark.xfail(reason="XXX: place()")
    def test_insert_noncontiguous(self):
        a = np.arange(6).reshape(2,3).T # force non-c-contiguous
        # uses arr_insert
        np.place(a, a>2, [44, 55])
        assert_equal(a, np.array([[0, 44], [1, 55], [2, 44]]))
        # hit one of the failing paths
        assert_raises(ValueError, np.place, a, a>20, [])

    def test_put_noncontiguous(self):
        a = np.arange(6).reshape(2,3).T # force non-c-contiguous
        assert not a.flags["C_CONTIGUOUS"]  # sanity check
        np.put(a, [0, 2], [44, 55])
        assert_equal(a, np.array([[44, 3], [55, 4], [2, 5]]))

    @pytest.mark.xfail(reason="XXX: putmask()")
    def test_putmask_noncontiguous(self):
        a = np.arange(6).reshape(2,3).T # force non-c-contiguous
        # uses arr_putmask
        np.putmask(a, a>2, a**2)
        assert_equal(a, np.array([[0, 9], [1, 16], [2, 25]]))

    def test_take_mode_raise(self):
        a = np.arange(6, dtype='int')
        out = np.empty(2, dtype='int')
        np.take(a, [0, 2], out=out, mode='raise')
        assert_equal(out, np.array([0, 2]))

    def test_choose_mod_raise(self):
        a = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        out = np.empty((3,3), dtype='int')
        choices = [-10, 10]
        np.choose(a, choices, out=out, mode='raise')
        assert_equal(out, np.array([[ 10, -10,  10],
                                    [-10,  10, -10],
                                    [ 10, -10,  10]]))

    @pytest.mark.xfail(reason="XXX: ndarray.flat")
    def test_flatiter__array__(self):
        a = np.arange(9).reshape(3,3)
        b = a.T.flat
        c = b.__array__()
        # triggers the WRITEBACKIFCOPY resolution, assuming refcount semantics
        del c

    def test_dot_out(self):
        # if HAVE_CBLAS, will use WRITEBACKIFCOPY
        a = np.arange(9, dtype=float).reshape(3,3)
        b = np.dot(a, a, out=a)
        assert_equal(b, np.array([[15, 18, 21], [42, 54, 66], [69, 90, 111]]))

    @pytest.mark.skip(reason="XXX: npy_create_writebackifcopy()")
    def test_view_assign(self):
        from numpy.core._multiarray_tests import npy_create_writebackifcopy, npy_resolve

        arr = np.arange(9).reshape(3, 3).T
        arr_wb = npy_create_writebackifcopy(arr)
        assert_(arr_wb.flags.writebackifcopy)
        assert_(arr_wb.base is arr)
        arr_wb[...] = -100
        npy_resolve(arr_wb)
        # arr changes after resolve, even though we assigned to arr_wb
        assert_equal(arr, -100)
        # after resolve, the two arrays no longer reference each other
        assert_(arr_wb.ctypes.data != 0)
        assert_equal(arr_wb.base, None)
        # assigning to arr_wb does not get transferred to arr
        arr_wb[...] = 100
        assert_equal(arr, -100)

    @pytest.mark.skip(reason="XXX: npy_create_writebackifcopy()")
    @pytest.mark.leaks_references(
            reason="increments self in dealloc; ignore since deprecated path.")
    def test_dealloc_warning(self):
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            arr = np.arange(9).reshape(3, 3)
            v = arr.T
            _multiarray_tests.npy_abuse_writebackifcopy(v)
            assert len(sup.log) == 1

    @pytest.mark.skip(reason="XXX: npy_create_writebackifcopy()")
    def test_view_discard_refcount(self):
        from numpy.core._multiarray_tests import npy_create_writebackifcopy, npy_discard

        arr = np.arange(9).reshape(3, 3).T
        orig = arr.copy()
        if HAS_REFCOUNT:
            arr_cnt = sys.getrefcount(arr)
        arr_wb = npy_create_writebackifcopy(arr)
        assert_(arr_wb.flags.writebackifcopy)
        assert_(arr_wb.base is arr)
        arr_wb[...] = -100
        npy_discard(arr_wb)
        # arr remains unchanged after discard
        assert_equal(arr, orig)
        # after discard, the two arrays no longer reference each other
        assert_(arr_wb.ctypes.data != 0)
        assert_equal(arr_wb.base, None)
        if HAS_REFCOUNT:
            assert_equal(arr_cnt, sys.getrefcount(arr))
        # assigning to arr_wb does not get transferred to arr
        arr_wb[...] = 100
        assert_equal(arr, orig)


class TestArange:
    def test_infinite(self):
        assert_raises(
            (RuntimeError, ValueError),  # "unsupported range",
            np.arange, 0, np.inf
        )

    def test_nan_step(self):
        assert_raises(
            (RuntimeError, ValueError),  # "cannot compute length",
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

    @pytest.mark.xfail(reason="weird arange signature (optionals before required args)")
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

    @pytest.mark.skip(reason="arange for booleans: numpy maybe deprecates?")
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
        with pytest.raises((ValueError, RuntimeError)):
            # Fails discovering start dtype
            np.arange(*args)



@pytest.mark.xfail(reason='comparison: builtin.bools or...?')
def test_richcompare_scalar_boolean_singleton_return():
    # These are currently guaranteed to be the boolean singletons, but maybe
    # returning NumPy booleans would also be OK:
    assert (np.array(0) == "a") is False
    assert (np.array(0) != "a") is True
    assert (np.int16(0) == "a") is False
    assert (np.int16(0) != "a") is True



@pytest.mark.xfail(reason='npymath')
@pytest.mark.parametrize(
    ["fun", "npfun"],
    [
        (_multiarray_tests.npy_cabs, np.absolute),
        (_multiarray_tests.npy_carg, np.angle)
    ]
)
@pytest.mark.parametrize("x", [1, np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("y", [1, np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("test_dtype", np.complexfloating.__subclasses__())
def test_npymath_complex(fun, npfun, x, y, test_dtype):
    # Smoketest npymath functions
    z = test_dtype(complex(x, y))
    got = fun(z)
    expected = npfun(z)
    assert_allclose(got, expected)


@pytest.mark.xfail(reason='npymath')
def test_npymath_real():
    # Smoketest npymath functions
    from numpy.core._multiarray_tests import (
        npy_log10, npy_cosh, npy_sinh, npy_tan, npy_tanh)

    funcs = {npy_log10: np.log10,
             npy_cosh: np.cosh,
             npy_sinh: np.sinh,
             npy_tan: np.tan,
             npy_tanh: np.tanh}
    vals = (1, np.inf, -np.inf, np.nan)
    types = (np.float32, np.float64, np.longdouble)

    with np.errstate(all='ignore'):
        for fun, npfun in funcs.items():
            for x, t in itertools.product(vals, types):
                z = t(x)
                got = fun(z)
                expected = npfun(z)
                assert_allclose(got, expected)


@pytest.mark.xfail(reason='implement views/dtypes')
class TestViewDtype:
    """
    Verify that making a view of a non-contiguous array works as expected.
    """
    def test_smaller_dtype_multiple(self):
        # x is non-contiguous
        x = np.arange(10, dtype='<i4')[::2]
        with pytest.raises(ValueError,
                           match='the last axis must be contiguous'):
            x.view('<i2')
        expected = [[0, 0], [2, 0], [4, 0], [6, 0], [8, 0]]
        assert_array_equal(x[:, np.newaxis].view('<i2'), expected)

    def test_smaller_dtype_not_multiple(self):
        # x is non-contiguous
        x = np.arange(5, dtype='<i4')[::2]

        with pytest.raises(ValueError,
                           match='the last axis must be contiguous'):
            x.view('S3')
        with pytest.raises(ValueError,
                           match='When changing to a smaller dtype'):
            x[:, np.newaxis].view('S3')

        # Make sure the problem is because of the dtype size
        expected = [[b''], [b'\x02'], [b'\x04']]
        assert_array_equal(x[:, np.newaxis].view('S4'), expected)

    def test_larger_dtype_multiple(self):
        # x is non-contiguous in the first dimension, contiguous in the last
        x = np.arange(20, dtype='<i2').reshape(10, 2)[::2, :]
        expected = np.array([[65536], [327684], [589832],
                             [851980], [1114128]], dtype='<i4')
        assert_array_equal(x.view('<i4'), expected)

    def test_larger_dtype_not_multiple(self):
        # x is non-contiguous in the first dimension, contiguous in the last
        x = np.arange(20, dtype='<i2').reshape(10, 2)[::2, :]
        with pytest.raises(ValueError,
                           match='When changing to a larger dtype'):
            x.view('S3')
        # Make sure the problem is because of the dtype size
        expected = [[b'\x00\x00\x01'], [b'\x04\x00\x05'], [b'\x08\x00\t'],
                    [b'\x0c\x00\r'], [b'\x10\x00\x11']]
        assert_array_equal(x.view('S4'), expected)

    def test_f_contiguous(self):
        # x is F-contiguous
        x = np.arange(4 * 3, dtype='<i4').reshape(4, 3).T
        with pytest.raises(ValueError,
                           match='the last axis must be contiguous'):
            x.view('<i2')

    def test_non_c_contiguous(self):
        # x is contiguous in axis=-1, but not C-contiguous in other axes
        x = np.arange(2 * 3 * 4, dtype='i1').\
                    reshape(2, 3, 4).transpose(1, 0, 2)
        expected = [[[256, 770], [3340, 3854]],
                    [[1284, 1798], [4368, 4882]],
                    [[2312, 2826], [5396, 5910]]]
        assert_array_equal(x.view('<i2'), expected)


# Test various array sizes that hit different code paths in quicksort-avx512
@pytest.mark.parametrize("N", [8, 16, 24, 32, 48, 64, 96, 128, 151, 191,
                               256, 383, 512, 1023, 2047])
def test_sort_float(N):
    # Regular data with nan sprinkled
    np.random.seed(42)
    arr = -0.5 + np.random.sample(N).astype('f')
    arr[np.random.choice(arr.shape[0], 3)] = np.nan
    assert_equal(np.sort(arr, kind='quick'), np.sort(arr, kind='heap'))

    # (2) with +INF
    infarr = np.inf*np.ones(N, dtype='f')
    infarr[np.random.choice(infarr.shape[0], 5)] = -1.0
    assert_equal(np.sort(infarr, kind='quick'), np.sort(infarr, kind='heap'))

    # (3) with -INF
    neginfarr = -np.inf*np.ones(N, dtype='f')
    neginfarr[np.random.choice(neginfarr.shape[0], 5)] = 1.0
    assert_equal(np.sort(neginfarr, kind='quick'),
                 np.sort(neginfarr, kind='heap'))

    # (4) with +/-INF
    infarr = np.inf*np.ones(N, dtype='f')
    infarr[np.random.choice(infarr.shape[0], (int)(N/2))] = -np.inf
    assert_equal(np.sort(infarr, kind='quick'), np.sort(infarr, kind='heap'))



def test_sort_int():
    # Random data with NPY_MAX_INT32 and NPY_MIN_INT32 sprinkled
    #rng = np.random.default_rng(42)
    np.random.seed(1234)
    N = 2047
    minv = np.iinfo(np.int32).min
    maxv = np.iinfo(np.int32).max
    arr = np.random.randint(low=minv, high=maxv, size=N).astype('int32')
    arr[np.random.choice(arr.shape[0], 10)] = minv
    arr[np.random.choice(arr.shape[0], 10)] = maxv
    assert_equal(np.sort(arr, kind='quick'), np.sort(arr, kind='heap'))

