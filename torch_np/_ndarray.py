import operator

import torch

from . import _binary_ufuncs, _dtypes, _funcs, _helpers, _unary_ufuncs
from ._detail import _dtypes_impl, _util
from ._detail import implementations as _impl

newaxis = None

FLAGS = [
    "C_CONTIGUOUS",
    "F_CONTIGUOUS",
    "OWNDATA",
    "WRITEABLE",
    "ALIGNED",
    "WRITEBACKIFCOPY",
    "FNC",
    "FORC",
    "BEHAVED",
    "CARRAY",
    "FARRAY",
]

SHORTHAND_TO_FLAGS = {
    "C": "C_CONTIGUOUS",
    "F": "F_CONTIGUOUS",
    "O": "OWNDATA",
    "W": "WRITEABLE",
    "A": "ALIGNED",
    "X": "WRITEBACKIFCOPY",
    "B": "BEHAVED",
    "CA": "CARRAY",
    "FA": "FARRAY",
}


class Flags:
    def __init__(self, flag_to_value: dict):
        assert all(k in FLAGS for k in flag_to_value.keys())  # sanity check
        self._flag_to_value = flag_to_value

    def __getattr__(self, attr: str):
        if attr.islower() and attr.upper() in FLAGS:
            return self[attr.upper()]
        else:
            raise AttributeError(f"No flag attribute '{attr}'")

    def __getitem__(self, key):
        if key in SHORTHAND_TO_FLAGS.keys():
            key = SHORTHAND_TO_FLAGS[key]
        if key in FLAGS:
            try:
                return self._flag_to_value[key]
            except KeyError as e:
                raise NotImplementedError(f"{key=}") from e
        else:
            raise KeyError(f"No flag key '{key}'")


##################### ndarray class ###########################


class ndarray:
    def __init__(self, t=None):
        if t is None:
            self.tensor = torch.Tensor()
        else:
            self.tensor = torch.as_tensor(t)

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def size(self):
        return self.tensor.numel()

    @property
    def ndim(self):
        return self.tensor.ndim

    @property
    def dtype(self):
        return _dtypes.dtype(self.tensor.dtype)

    @property
    def strides(self):
        elsize = self.tensor.element_size()
        return tuple(stride * elsize for stride in self.tensor.stride())

    @property
    def itemsize(self):
        return self.tensor.element_size()

    @property
    def flags(self):
        # Note contiguous in torch is assumed C-style

        # check if F contiguous
        from itertools import accumulate

        f_strides = tuple(accumulate(list(self.tensor.shape), func=lambda x, y: x * y))
        f_strides = (1,) + f_strides[:-1]
        is_f_contiguous = f_strides == self.tensor.stride()

        return Flags(
            {
                "C_CONTIGUOUS": self.tensor.is_contiguous(),
                "F_CONTIGUOUS": is_f_contiguous,
                "OWNDATA": self.tensor._base is None,
                "WRITEABLE": True,  # pytorch does not have readonly tensors
            }
        )

    @property
    def T(self):
        return self.transpose()

    @property
    def real(self):
        return _funcs.real(self)

    @real.setter
    def real(self, value):
        self.tensor.real = asarray(value).tensor

    @property
    def imag(self):
        return _funcs.imag(self)

    @imag.setter
    def imag(self, value):
        self.tensor.imag = asarray(value).tensor

    round = _funcs.round

    # ctors
    def astype(self, dtype):
        newt = ndarray()
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        newt.tensor = self.tensor.to(torch_dtype)
        return newt

    def copy(self, order="C"):
        if order != "C":
            raise NotImplementedError
        tensor = self.tensor.clone()
        return ndarray(tensor)

    def tolist(self):
        return self.tensor.tolist()

    ###  niceties ###
    def __str__(self):
        return (
            str(self.tensor)
            .replace("tensor", "array_w")
            .replace("dtype=torch.", "dtype=")
        )

    __repr__ = __str__

    ### comparisons ###
    def __eq__(self, other):
        try:
            return _binary_ufuncs.equal(self, other)
        except (RuntimeError, TypeError):
            # Failed to convert other to array: definitely not equal.
            falsy = torch.full(self.shape, fill_value=False, dtype=bool)
            return asarray(falsy)

    def __ne__(self, other):
        try:
            return _binary_ufuncs.not_equal(self, other)
        except (RuntimeError, TypeError):
            # Failed to convert other to array: definitely not equal.
            falsy = torch.full(self.shape, fill_value=True, dtype=bool)
            return asarray(falsy)

    __gt__ = _binary_ufuncs.greater
    __lt__ = _binary_ufuncs.less
    __ge__ = _binary_ufuncs.greater_equal
    __le__ = _binary_ufuncs.less_equal

    def __bool__(self):
        try:
            return bool(self.tensor)
        except RuntimeError:
            raise ValueError(
                "The truth value of an array with more than one "
                "element is ambiguous. Use a.any() or a.all()"
            )

    def __index__(self):
        try:
            return operator.index(self.tensor.item())
        except Exception:
            mesg = "only integer scalar arrays can be converted to a scalar index"
            raise TypeError(mesg)

    def __float__(self):
        return float(self.tensor)

    def __complex__(self):
        try:
            return complex(self.tensor)
        except ValueError as e:
            raise TypeError(*e.args)

    def __int__(self):
        return int(self.tensor)

    # XXX : are single-element ndarrays scalars?
    # in numpy, only array scalars have the `is_integer` method
    def is_integer(self):
        try:
            result = int(self.tensor) == self.tensor
        except Exception:
            result = False
        return result

    ### sequence ###
    def __len__(self):
        return self.tensor.shape[0]

    ### arithmetic ###

    # add, self + other
    __add__ = __radd__ = _binary_ufuncs.add

    def __iadd__(self, other):
        return _binary_ufuncs.add(self, other, out=self)

    # sub, self - other
    __sub__ = _binary_ufuncs.subtract

    # XXX: generate a function just for this? AND other non-commutative ops.
    def __rsub__(self, other):
        return _binary_ufuncs.subtract(other, self)

    def __isub__(self, other):
        return _binary_ufuncs.subtract(self, other, out=self)

    # mul, self * other
    __mul__ = __rmul__ = _binary_ufuncs.multiply

    def __imul__(self, other):
        return _binary_ufuncs.multiply(self, other, out=self)

    # div, self / other
    __truediv__ = _binary_ufuncs.divide

    def __rtruediv__(self, other):
        return _binary_ufuncs.divide(other, self)

    def __itruediv__(self, other):
        return _binary_ufuncs.divide(self, other, out=self)

    # floordiv, self // other
    __floordiv__ = _binary_ufuncs.floor_divide

    def __rfloordiv__(self, other):
        return _binary_ufuncs.floor_divide(other, self)

    def __ifloordiv__(self, other):
        return _binary_ufuncs.floor_divide(self, other, out=self)

    __divmod__ = _binary_ufuncs.divmod

    # power, self**exponent
    __pow__ = __rpow__ = _binary_ufuncs.float_power

    def __rpow__(self, exponent):
        return _binary_ufuncs.float_power(exponent, self)

    def __ipow__(self, exponent):
        return _binary_ufuncs.float_power(self, exponent, out=self)

    # remainder, self % other
    __mod__ = __rmod__ = _binary_ufuncs.remainder

    def __imod__(self, other):
        return _binary_ufuncs.remainder(self, other, out=self)

    # bitwise ops
    # and, self & other
    __and__ = __rand__ = _binary_ufuncs.bitwise_and

    def __iand__(self, other):
        return _binary_ufuncs.bitwise_and(self, other, out=self)

    # or, self | other
    __or__ = __ror__ = _binary_ufuncs.bitwise_or

    def __ior__(self, other):
        return _binary_ufuncs.bitwise_or(self, other, out=self)

    # xor, self ^ other
    __xor__ = __rxor__ = _binary_ufuncs.bitwise_xor

    def __ixor__(self, other):
        return _binary_ufuncs.bitwise_xor(self, other, out=self)

    # bit shifts
    __lshift__ = __rlshift__ = _binary_ufuncs.left_shift

    def __ilshift__(self, other):
        return _binary_ufuncs.left_shift(self, other, out=self)

    __rshift__ = __rrshift__ = _binary_ufuncs.right_shift

    def __irshift__(self, other):
        return _binary_ufuncs.right_shift(self, other, out=self)

    __matmul__ = _binary_ufuncs.matmul

    def __rmatmul__(self, other):
        return _binary_ufuncs.matmul(other, self)

    def __imatmul__(self, other):
        return _binary_ufuncs.matmul(self, other, out=self)

    # unary ops
    __invert__ = _unary_ufuncs.invert
    __abs__ = _unary_ufuncs.absolute
    __pos__ = _unary_ufuncs.positive
    __neg__ = _unary_ufuncs.negative

    conjugate = _unary_ufuncs.conjugate
    conj = conjugate

    ### methods to match namespace functions

    squeeze = _funcs.squeeze
    swapaxes = _funcs.swapaxes

    def transpose(self, *axes):
        # np.transpose(arr, axis=None) but arr.transpose(*axes)
        return _funcs.transpose(self, axes)

    def reshape(self, *shape, order="C"):
        # arr.reshape(shape) and arr.reshape(*shape)
        return _funcs.reshape(self, shape, order=order)

    ravel = _funcs.ravel
    flatten = _funcs._flatten

    nonzero = _funcs.nonzero
    clip = _funcs.clip
    repeat = _funcs.repeat

    diagonal = _funcs.diagonal
    trace = _funcs.trace
    dot = _funcs.dot

    ### sorting ###

    def sort(self, axis=-1, kind=None, order=None):
        # ndarray.sort works in-place
        self.tensor.copy_(_funcs._sort(self.tensor, axis, kind, order))

    argsort = _funcs.argsort
    searchsorted = _funcs.searchsorted

    ### reductions ###
    argmax = _funcs.argmax
    argmin = _funcs.argmin

    any = _funcs.any
    all = _funcs.all
    max = _funcs.max
    min = _funcs.min
    ptp = _funcs.ptp

    sum = _funcs.sum
    prod = _funcs.prod
    mean = _funcs.mean
    var = _funcs.var
    std = _funcs.std

    cumsum = _funcs.cumsum
    cumprod = _funcs.cumprod

    ### indexing ###
    @staticmethod
    def _upcast_int_indices(index):
        if isinstance(index, torch.Tensor):
            if index.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                return index.to(torch.int64)
        elif isinstance(index, tuple):
            return tuple(ndarray._upcast_int_indices(i) for i in index)
        return index

    def __getitem__(self, index):
        index = _helpers.ndarrays_to_tensors(index)
        index = ndarray._upcast_int_indices(index)
        return ndarray(self.tensor.__getitem__(index))

    def __setitem__(self, index, value):
        index = _helpers.ndarrays_to_tensors(index)
        index = ndarray._upcast_int_indices(index)
        value = _helpers.ndarrays_to_tensors(value)
        return self.tensor.__setitem__(index, value)


# This is the ideally the only place which talks to ndarray directly.
# The rest goes through asarray (preferred) or array.


def array(obj, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None):
    _util.subok_not_ok(like, subok)
    if order != "K":
        raise NotImplementedError

    # a happy path
    if isinstance(obj, ndarray):
        if copy is False and dtype is None and ndmin <= obj.ndim:
            return obj

    # lists of ndarrays: [1, [2, 3], ndarray(4)] convert to lists of lists
    if isinstance(obj, (list, tuple)):
        a1 = []
        for elem in obj:
            if isinstance(elem, ndarray):
                a1.append(elem.tensor.tolist())
            else:
                a1.append(elem)
        obj = a1

    # is obj an ndarray already?
    if isinstance(obj, ndarray):
        obj = obj.tensor

    # is a specific dtype requrested?
    torch_dtype = None
    if dtype is not None:
        torch_dtype = _dtypes.dtype(dtype).torch_dtype

    tensor = _util._coerce_to_tensor(obj, torch_dtype, copy, ndmin)
    return ndarray(tensor)


def asarray(a, dtype=None, order=None, *, like=None):
    if order is None:
        order = "K"
    return array(a, dtype=dtype, order=order, like=like, copy=False, ndmin=0)


###### dtype routines


def _extract_dtype(entry):
    try:
        dty = _dtypes.dtype(entry)
    except Exception:
        dty = asarray(entry).dtype
    return dty


def can_cast(from_, to, casting="safe"):
    from_ = _extract_dtype(from_)
    to_ = _extract_dtype(to)

    return _dtypes_impl.can_cast_impl(from_.torch_dtype, to_.torch_dtype, casting)


def result_type(*arrays_and_dtypes):
    dtypes = []

    for entry in arrays_and_dtypes:
        dty = _extract_dtype(entry)
        dtypes.append(dty.torch_dtype)

    torch_dtype = _dtypes_impl.result_type_impl(dtypes)
    return _dtypes.dtype(torch_dtype)
