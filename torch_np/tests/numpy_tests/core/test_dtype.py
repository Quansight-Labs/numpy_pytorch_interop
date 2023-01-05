import sys
import operator
import pytest
import types
from typing import Any

import torch_np as np
from torch_np.testing import (
    assert_, assert_equal, assert_array_equal)
from pytest import raises as assert_raises


import pickle
from itertools import permutations
import random




def assert_dtype_equal(a, b):
    assert_equal(a, b)
    assert_equal(hash(a), hash(b),
                 "two equivalent types do not hash to the same value !")

def assert_dtype_not_equal(a, b):
    assert_(a != b)
    assert_(hash(a) != hash(b),
            "two different types hash to the same value !")

class TestBuiltin:
    @pytest.mark.parametrize('t', [int, float, complex, np.int32])
    def test_run(self, t):
        """Only test hash runs at all."""
        dt = np.dtype(t)
        hash(dt)

    def test_equivalent_dtype_hashing(self):
        # Make sure equivalent dtypes with different type num hash equal
        uintp = np.dtype(np.uintp)
        if uintp.itemsize == 4:
            left = uintp
            right = np.dtype(np.uint32)
        else:
            left = uintp
            right = np.dtype(np.ulonglong)
        assert_(left == right)
        assert_(hash(left) == hash(right))

    def test_invalid_types(self):
        # Make sure invalid type strings raise an error

        assert_raises(TypeError, np.dtype, 'O3')
        assert_raises(TypeError, np.dtype, 'O5')
        assert_raises(TypeError, np.dtype, 'O7')
        assert_raises(TypeError, np.dtype, 'b3')
        assert_raises(TypeError, np.dtype, 'h4')
        assert_raises(TypeError, np.dtype, 'I5')
        assert_raises(TypeError, np.dtype, 'e3')
        assert_raises(TypeError, np.dtype, 'f5')

        if np.dtype('g').itemsize == 8 or np.dtype('g').itemsize == 16:
            assert_raises(TypeError, np.dtype, 'g12')
        elif np.dtype('g').itemsize == 12:
            assert_raises(TypeError, np.dtype, 'g16')

        if np.dtype('l').itemsize == 8:
            assert_raises(TypeError, np.dtype, 'l4')
            assert_raises(TypeError, np.dtype, 'L4')
        else:
            assert_raises(TypeError, np.dtype, 'l8')
            assert_raises(TypeError, np.dtype, 'L8')

        if np.dtype('q').itemsize == 8:
            assert_raises(TypeError, np.dtype, 'q4')
            assert_raises(TypeError, np.dtype, 'Q4')
        else:
            assert_raises(TypeError, np.dtype, 'q8')
            assert_raises(TypeError, np.dtype, 'Q8')

    def test_richcompare_invalid_dtype_equality(self):
        # Make sure objects that cannot be converted to valid
        # dtypes results in False/True when compared to valid dtypes.
        # Here 7 cannot be converted to dtype. No exceptions should be raised

        assert not np.dtype(np.int32) == 7, "dtype richcompare failed for =="
        assert np.dtype(np.int32) != 7, "dtype richcompare failed for !="

    @pytest.mark.parametrize(
        'operation',
        [operator.le, operator.lt, operator.ge, operator.gt])
    def test_richcompare_invalid_dtype_comparison(self, operation):
        # Make sure TypeError is raised for comparison operators
        # for invalid dtypes. Here 7 is an invalid dtype.

        with pytest.raises(TypeError):
            operation(np.dtype(np.int32), 7)

    @pytest.mark.parametrize("dtype",
             ['Bool', 'Bytes0', 'Complex32', 'Complex64',
              'Datetime64', 'Float16', 'Float32', 'Float64',
              'Int8', 'Int16', 'Int32', 'Int64',
              'Object0', 'Str0', 'Timedelta64',
              'UInt8', 'UInt16', 'Uint32', 'UInt32',
              'Uint64', 'UInt64', 'Void0',
              "Float128", "Complex128"])
    def test_numeric_style_types_are_invalid(self, dtype):
        with assert_raises(TypeError):
            np.dtype(dtype)

    def test_remaining_dtypes_with_bad_bytesize(self):
        # The np.<name> aliases were deprecated, these probably should be too 
        assert np.dtype("int0") is np.dtype("intp")
        assert np.dtype("uint0") is np.dtype("uintp")
        assert np.dtype("bool8") is np.dtype("bool")
        assert np.dtype("bytes0") is np.dtype("bytes")
        assert np.dtype("str0") is np.dtype("str")
        assert np.dtype("object0") is np.dtype("object")

    @pytest.mark.parametrize(
        'value',
        [
         'i4, f4'
        ])
    def test_dtype_bytes_str_equivalence(self, value):
        bytes_value = value.encode('ascii')
        from_bytes = np.dtype(bytes_value)
        from_str = np.dtype(value)
        assert_dtype_equal(from_bytes, from_str)

    def test_dtype_from_bytes(self):
        # Single character where value is a valid type code
        assert_dtype_equal(np.dtype(b'f'), np.dtype('float32'))

        # Bytes with non-ascii values raise errors
        assert_raises(TypeError, np.dtype, b'\xff')
        assert_raises(TypeError, np.dtype, b's\xff')








def iter_struct_object_dtypes():
    """
    Iterates over a few complex dtypes and object pattern which
    fill the array with a given object (defaults to a singleton).

    Yields
    ------
    dtype : dtype
    pattern : tuple
        Structured tuple for use with `np.array`.
    count : int
        Number of objects stored in the dtype.
    singleton : object
        A singleton object. The returned pattern is constructed so that
        all objects inside the datatype are set to the singleton.
    """
    obj = object()

    dt = np.dtype([('b', 'O', (2, 3))])
    p = ([[obj] * 3] * 2,)
    yield pytest.param(dt, p, 6, obj, id="<subarray>")

    dt = np.dtype([('a', 'i4'), ('b', 'O', (2, 3))])
    p = (0, [[obj] * 3] * 2)
    yield pytest.param(dt, p, 6, obj, id="<subarray in field>")

    dt = np.dtype([('a', 'i4'),
                   ('b', [('ba', 'O'), ('bb', 'i1')], (2, 3))])
    p = (0, [[(obj, 0)] * 3] * 2)
    yield pytest.param(dt, p, 6, obj, id="<structured subarray 1>")

    dt = np.dtype([('a', 'i4'),
                   ('b', [('ba', 'O'), ('bb', 'O')], (2, 3))])
    p = (0, [[(obj, obj)] * 3] * 2)
    yield pytest.param(dt, p, 12, obj, id="<structured subarray 2>")







class TestDtypeAttributeDeletion:

    def test_dtype_non_writable_attributes_deletion(self):
        dt = np.dtype(np.double)
        attr = ["subdtype", "descr", "str", "name", "base", "shape",
                "isbuiltin", "isnative", "isalignedstruct", "fields",
                "metadata", "hasobject"]

        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)

    def test_dtype_writable_attributes_deletion(self):
        dt = np.dtype(np.double)
        attr = ["names"]
        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)





class TestPickling:

    def check_pickling(self, dtype):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            buf = pickle.dumps(dtype, proto)
            # The dtype pickling itself pickles `np.dtype` if it is pickled
            # as a singleton `dtype` should be stored in the buffer:
            assert b"_DType_reconstruct" not in buf
            assert b"dtype" in buf
            pickled = pickle.loads(buf)
            assert_equal(pickled, dtype)
            assert_equal(pickled.descr, dtype.descr)
            if dtype.metadata is not None:
                assert_equal(pickled.metadata, dtype.metadata)
            # Check the reconstructed dtype is functional
            x = np.zeros(3, dtype=dtype)
            y = np.zeros(3, dtype=pickled)
            assert_equal(x, y)
            assert_equal(x[0], y[0])

    @pytest.mark.parametrize('t', [int, float, complex, np.int32, bool])
    def test_builtin(self, t):
        self.check_pickling(np.dtype(t))


    @pytest.mark.parametrize("DType",
        [type(np.dtype(t)) for t in np.typecodes['All']] +
        [np.dtype])
    def test_pickle_types(self, DType):
        # Check that DTypes (the classes/types) roundtrip when pickling
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            roundtrip_DType = pickle.loads(pickle.dumps(DType, proto))
            assert roundtrip_DType is DType


class TestPromotion:
    """Test cases related to more complex DType promotions.  Further promotion
    tests are defined in `test_numeric.py`
    """
    @pytest.mark.parametrize(["other", "expected", "expected_weak"],
            [(2**16-1, np.complex64, None),
             (2**32-1, np.complex128, np.complex64),
             (np.float16(2), np.complex64, None),
             (np.float32(2), np.complex64, None),
             # repeat for complex scalars:
             (np.complex64(2), np.complex64, None),
             ])
    def test_complex_other_value_based(self,
            weak_promotion, other, expected, expected_weak):
        if weak_promotion and expected_weak is not None:
            expected = expected_weak

        # This would change if we modify the value based promotion
        min_complex = np.dtype(np.complex64)

        res = np.result_type(other, min_complex)
        assert res == expected
        # Check the same for a simple ufunc call that uses the same logic:
        res = np.minimum(other, np.ones(3, dtype=min_complex)).dtype
        assert res == expected

    @pytest.mark.parametrize(["other", "expected"],
                 [(np.bool_, np.complex128),
                  (np.int64, np.complex128),
                  (np.float16, np.complex64),
                  (np.float32, np.complex64),
                  (np.float64, np.complex128),
                  (np.complex64, np.complex64),
                  (np.complex128, np.complex128),
                  ])
    def test_complex_scalar_value_based(self, other, expected):
        # This would change if we modify the value based promotion
        complex_scalar = 1j

        res = np.result_type(other, complex_scalar)
        assert res == expected
        # Check the same for a simple ufunc call that uses the same logic:
        res = np.minimum(np.ones(3, dtype=other), complex_scalar).dtype
        assert res == expected


    @pytest.mark.parametrize("val", [2, 2**32, 2**63, 2**64, 2*100])
    def test_python_integer_promotion(self, val):
        # If we only path scalars (mainly python ones!), the result must take
        # into account that the integer may be considered int32, int64, uint64,
        # or object depending on the input value.  So test those paths!
        expected_dtype = np.result_type(np.array(val).dtype, np.array(0).dtype)
        assert np.result_type(val, 0) == expected_dtype
        # For completeness sake, also check with a NumPy scalar as second arg:
        assert np.result_type(val, np.int8(0)) == expected_dtype


    @pytest.mark.parametrize(["dtypes", "expected"], [
             # These promotions are not associative/commutative:
             ([np.int16, np.float16], np.float32),
             ([np.int8, np.float16], np.float32),
             ([np.uint8, np.int16, np.float16], np.float32),
             # The following promotions are not ambiguous, but cover code
             # paths of abstract promotion (no particular logic being tested)
             ([1, 1, np.float64], np.float64),
             ([1, 1., np.complex128], np.complex128),
             ([1, 1j, np.float64], np.complex128),
             ([1., 1., np.int64], np.float64),
             ([1., 1j, np.float64], np.complex128),
             ([1j, 1j, np.float64], np.complex128),
             ([1, True, np.bool_], np.int_),
            ])
    def test_permutations_do_not_influence_result(self, dtypes, expected):
        # Tests that most permutations do not influence the result.  In the
        # above some uint and int combintations promote to a larger integer
        # type, which would then promote to a larger than necessary float.
        for perm in permutations(dtypes):
            assert np.result_type(*perm) == expected



def test_dtypes_are_true():
    # test for gh-6294
    assert bool(np.dtype('f8'))
    assert bool(np.dtype('i8'))




def test_keyword_argument():
    # test for https://github.com/numpy/numpy/pull/16574#issuecomment-642660971
    assert np.dtype(dtype=np.float64) == np.dtype(np.float64)


class TestFromDTypeAttribute:
    def test_simple(self):
        class dt:
            dtype = np.dtype("f8")

        assert np.dtype(dt) == np.float64
        assert np.dtype(dt()) == np.float64

    def test_recursion(self):
        class dt:
            pass

        dt.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt)

        dt_instance = dt()
        dt_instance.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt_instance)


class TestDTypeClasses:
    @pytest.mark.parametrize("dtype", list(np.typecodes['All']))
    def test_basic_dtypes_subclass_properties(self, dtype):
        # Note: Except for the isinstance and type checks, these attributes
        #       are considered currently private and may change.
        dtype = np.dtype(dtype)
        assert isinstance(dtype, np.dtype)
        assert type(dtype) is not np.dtype
        assert type(dtype).__name__ == f"dtype[{dtype.type.__name__}]"
        assert type(dtype).__module__ == "numpy"
        assert not type(dtype)._abstract

        # the flexible dtypes and datetime/timedelta have additional parameters
        # which are more than just storage information, these would need to be
        # given when creating a dtype:
        parametric = (np.void, np.str_, np.bytes_, np.datetime64, np.timedelta64)
        if dtype.type not in parametric:
            assert not type(dtype)._parametric
            assert type(dtype)() is dtype
        else:
            assert type(dtype)._parametric
            with assert_raises(TypeError):
                type(dtype)()

    def test_dtype_superclass(self):
        assert type(np.dtype) is not type
        assert isinstance(np.dtype, type)

        assert type(np.dtype).__name__ == "_DTypeMeta"
        assert type(np.dtype).__module__ == "numpy"
        assert np.dtype._abstract



@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python 3.9")
class TestClassGetItem:
    def test_dtype(self) -> None:
        alias = np.dtype[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is np.dtype

    @pytest.mark.parametrize("code", np.typecodes["All"])
    def test_dtype_subclass(self, code: str) -> None:
        cls = type(np.dtype(code))
        alias = cls[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is cls

    @pytest.mark.parametrize("arg_len", range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            assert np.dtype[arg_tup]
        else:
            with pytest.raises(TypeError):
                np.dtype[arg_tup]

    def test_subscript_scalar(self) -> None:
        assert np.dtype[Any]


def test_result_type_integers_and_unitless_timedelta64():
    # Regression test for gh-20077.  The following call of `result_type`
    # would cause a seg. fault.
    td = np.timedelta64(4)
    result = np.result_type(0, td)
    assert_dtype_equal(result, td.dtype)


@pytest.mark.skipif(sys.version_info >= (3, 9), reason="Requires python 3.8")
def test_class_getitem_38() -> None:
    match = "Type subscription requires python >= 3.9"
    with pytest.raises(TypeError, match=match):
        np.dtype[Any]
