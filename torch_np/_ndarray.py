import functools

import torch

from ._detail import _util
from ._detail import _reductions
from . import _helpers
from . import _dtypes
from . import _unary_ufuncs
from . import _binary_ufuncs

from ._decorators import emulate_out_arg, axis_keepdims_wrapper, dtype_to_torch

from ._decorators import NoValue

newaxis = None


##################### ndarray class ###########################

class ndarray:
    def __init__(self):
        self._tensor = torch.Tensor()
        self._base = None

    @classmethod
    def _from_tensor_and_base(cls, tensor, base):
        self = cls()
        self._tensor = tensor
        self._base = base
        return self

    def get(self):
        return self._tensor

    @property
    def shape(self):
        return tuple(self._tensor.shape)

    @property
    def size(self):
        return self._tensor.numel()

    @property
    def ndim(self):
        return self._tensor.ndim

    @property
    def dtype(self):
        return _dtypes.dtype(self._tensor.dtype)

    @property
    def strides(self):
        return self._tensor.stride()   # XXX: byte strides

    @property
    def base(self):
        return self._base

    @property
    def T(self):
        return self.transpose()

    @property
    def real(self):
        return asarray(self._tensor.real)

    @real.setter
    def real(self, value):
        self._tensor.real = asarray(value).get()

    @property
    def imag(self):
        try:
            return asarray(self._tensor.imag)
        except RuntimeError:
            zeros = torch.zeros_like(self._tensor)
            return ndarray._from_tensor_and_base(zeros, None)

    @imag.setter
    def imag(self, value):
        self._tensor.imag = asarray(value).get()

    # ctors
    def astype(self, dtype):
        newt = ndarray()
        torch_dtype = _dtypes.torch_dtype_from(dtype)
        newt._tensor = self._tensor.to(torch_dtype)
        return newt

    def copy(self, order='C'):
        if order != 'C':
            raise NotImplementedError
        tensor = self._tensor.clone()
        return ndarray._from_tensor_and_base(tensor, None)

    def tolist(self):
        return self._tensor.tolist()

    ###  niceties ###
    def __str__(self):
        return str(self._tensor).replace("tensor", "array_w").replace("dtype=torch.", "dtype=")

    __repr__ = __str__

    ### comparisons ###
    def __eq__(self, other):
        try:
            return _binary_ufuncs.equal(self, other)
        except (RuntimeError, TypeError):
            # Failed to convert other to array: definitely not equal.
            falsy = torch.full(self.shape, fill_value=False, dtype=bool)
            return asarray(falsy)

    def __neq__(self, other):
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
            return bool(self._tensor)
        except RuntimeError:
            raise ValueError("The truth value of an array with more than one "
                             "element is ambiguous. Use a.any() or a.all()")

    def __index__(self):
        if self.size == 1:
            if _dtypes.is_integer(self.dtype):
                return int(self._tensor.item())
        mesg = "only integer scalar arrays can be converted to a scalar index"
        raise TypeError(mesg)

    def __float__(self):
        return float(self._tensor)

    def __int__(self):
        return int(self._tensor)

    # XXX : are single-element ndarrays scalars?
    def is_integer(self):
        if self.shape == ():
            if _dtypes.is_integer(self.dtype):
                return True
            return self._tensor.item().is_integer()
        else:
            return False


    ### sequence ###
    def __len__(self):
        return self._tensor.shape[0]

    ### arithmetic ###

    # add, self + other
    __add__ = __radd__ = _binary_ufuncs.add

    def __iadd__(self, other):
        return _binary_ufuncs.add(self, other, out=self)


    # sub, self - other
    __sub__ = __rsub__ = _binary_ufuncs.subtract

    def __isub__(self, other):
        return _binary_ufuncs.subtract(self, other, out=self)


    # mul, self * other
    __mul__ = __rmul__ = _binary_ufuncs.multiply

    def __imul__(self, other):
        return _binary_ufuncs.multiply(self, other, out=self)


    # div, self / other
    __truediv__ = __rtruediv__ = _binary_ufuncs.divide

    def __itruediv__(self, other):
        return _binary_ufuncs.divide(self, other, out=self)


    # floordiv, self // other
    __floordiv__ = __rfloordiv__ = _binary_ufuncs.floor_divide

    def __ifloordiv__(self, other):
        return _binary_ufuncs.floor_divide(self, other, out=self)


    # power, self**exponent
    __pow__ = __rpow__ = _binary_ufuncs.float_power

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


    # unary ops
    __invert__ = _unary_ufuncs.invert
    __abs__ = _unary_ufuncs.absolute
    __pos__ = _unary_ufuncs.positive
    __neg__ = _unary_ufuncs.negative

    ### methods to match namespace functions

    def squeeze(self, axis=None):
        if axis == ():
            tensor = self._tensor
        elif axis is None:
            tensor = self._tensor.squeeze()
        else:
            tensor = self._tensor.squeeze(axis)
        return ndarray._from_tensor_and_base(tensor, self)


    def reshape(self, *shape, order='C'):
        newshape = shape[0] if len(shape) == 1 else shape
        # if sh = (1, 2, 3), numpy allows both .reshape(sh) and .reshape(*sh)
        if order != 'C':
            raise NotImplementedError
        tensor = self._tensor.reshape(newshape)
        return ndarray._from_tensor_and_base(tensor, self)

    def transpose(self, *axes):
        # numpy allows both .reshape(sh) and .reshape(*sh)
        axes = axes[0] if len(axes) == 1 else axes
        if axes == () or axes is None:
            axes = tuple(range(self.ndim))[::-1]
        try:
            tensor = self._tensor.permute(axes)
        except RuntimeError:
            raise ValueError("axes don't match array")
        return ndarray._from_tensor_and_base(tensor, self)

    def ravel(self, order='C'):
        if order != 'C':
            raise NotImplementedError
        return ndarray._from_tensor_and_base(self._tensor.ravel(), self)

    def nonzero(self):
        tensor = self._tensor
        return tuple(asarray(_) for _ in tensor.nonzero(as_tuple=True))

    argmin = emulate_out_arg(axis_keepdims_wrapper(_reductions.argmin))
    argmax = emulate_out_arg(axis_keepdims_wrapper(_reductions.argmax))

    any = emulate_out_arg(axis_keepdims_wrapper(_reductions.any))
    all = emulate_out_arg(axis_keepdims_wrapper(_reductions.all))
    max = emulate_out_arg(axis_keepdims_wrapper(_reductions.max))
    min = emulate_out_arg(axis_keepdims_wrapper(_reductions.min))

    sum = emulate_out_arg(axis_keepdims_wrapper(dtype_to_torch(_reductions.sum)))
    prod = emulate_out_arg(axis_keepdims_wrapper(dtype_to_torch(_reductions.prod)))
    mean = emulate_out_arg(axis_keepdims_wrapper(dtype_to_torch(_reductions.mean)))
    var = emulate_out_arg(axis_keepdims_wrapper(dtype_to_torch(_reductions.var)))
    std = emulate_out_arg(axis_keepdims_wrapper(dtype_to_torch(_reductions.std)))


    ### indexing ###
    def __getitem__(self, *args, **kwds):
        t_args = _helpers.ndarrays_to_tensors(*args)
        return ndarray._from_tensor_and_base(self._tensor.__getitem__(*t_args, **kwds), self)

    def __setitem__(self, index, value):
        value = asarray(value).get()
        return self._tensor.__setitem__(index, value)


# This is the ideally the only place which talks to ndarray directly.
# The rest goes through asarray (preferred) or array.

def array(obj, dtype=None, *, copy=True, order='K', subok=False, ndmin=0,
          like=None):
    _util.subok_not_ok(like, subok)
    if order != 'K':
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
                a1.append(elem.get().tolist())
            else:
                a1.append(elem)
        obj = a1

    # is obj an ndarray already?
    base = None
    if isinstance(obj, ndarray):
        obj = obj._tensor
        base = obj

    # is a specific dtype requrested?
    torch_dtype = None
    if dtype is not None:
        torch_dtype = _dtypes.torch_dtype_from(dtype)
        base = None

    tensor = _util._coerce_to_tensor(obj, torch_dtype, copy, ndmin)
    return ndarray._from_tensor_and_base(tensor, base)


def asarray(a, dtype=None, order=None, *, like=None):
    if order is None:
        order = 'K'
    return array(a, dtype=dtype, order=order, like=like, copy=False, ndmin=0)


class asarray_replacer:
    def __init__(self, dispatch='one'):
        if dispatch not in ['one', 'two']:
            raise ValueError("ararray_replacer: unknown dispatch %s" % dispatch)
        self._dispatch = dispatch

    def __call__(self, func):
        if self._dispatch == 'one':
            @functools.wraps(func)
            def wrapped(x, *args, **kwds):
                x_tensor = asarray(x).get()
                return asarray(func(x_tensor, *args, **kwds))
            return wrapped
        else:
            raise ValueError


###### dtype routines

def can_cast(from_, to, casting='safe'):
    # XXX: merge with _dtypes.can_cast. The Q is who converts from ndarray, if needed.
    from_dtype = from_.dtype if isinstance(from_, ndarray) else _dtypes.dtype(from_)
    to_dtype = to.dtype if isinstance(to, ndarray) else _dtypes.dtype(to)

    return _dtypes.can_cast(from_dtype, to_dtype, casting)


def result_type(*arrays_and_dtypes):
    # XXX: clean up
    dtypes = []

    from ._dtypes import issubclass_

    for entry in arrays_and_dtypes:
        if issubclass_(entry, _dtypes._scalar_types.generic):
            dtypes.append(_dtypes.dtype(entry))
        elif isinstance(entry, _dtypes.DType):
            dtypes.append(entry)
        else:
            dtypes.append(asarray(entry).dtype)

    dtyp = dtypes[0]
    if len(dtypes) == 1:
        return dtyp

    for curr in dtypes[1:]:
        name = _dtypes._result_type_dict[dtyp.type.torch_dtype][curr.type.torch_dtype]
        dtyp = _dtypes.dtype(name)

    return dtyp


