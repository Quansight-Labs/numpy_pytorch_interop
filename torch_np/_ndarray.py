import functools

import torch

from . import _dtypes, _helpers, _ufunc_impl, _util

NoValue = None
newaxis = None


def axis_out_keepdims_wrapper(func):
    """`func` accepts an array-like as a 1st arg, returns a tensor.

    This decorator implements the generic handling of axis, out and keepdims
    arguments for reduction functions.
    """
    # XXX: move this out of _ndarray.py (circular imports)
    @functools.wraps(func)
    def wrapped(a, axis=None, out=None, keepdims=NoValue, *args, **kwds):
        arr = asarray(a)
        axis = _helpers.standardize_axis_arg(axis, arr.ndim)

        if axis == ():
            newshape = _util.expand_shape(arr.shape, axis=0)
            arr = arr.reshape(newshape)
            axis = (0,)

        result = func(arr, axis=axis, *args, **kwds)

        if keepdims:
            result = _helpers.apply_keepdims(result, axis, arr.ndim)
        return _helpers.result_or_out(result, out)

    return wrapped


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
        return _dtypes.dtype_from_torch(self._tensor.dtype)

    @property
    def strides(self):
        return self._tensor.stride()  # XXX: byte strides

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

    def copy(self, order="C"):
        if order != "C":
            raise NotImplementedError
        tensor = self._tensor.clone()
        return ndarray._from_tensor_and_base(tensor, None)

    def tolist(self):
        return self._tensor.tolist()

    ###  niceties ###
    def __str__(self):
        return (
            str(self._tensor)
            .replace("tensor", "array_w")
            .replace("dtype=torch.", "dtype=")
        )

    __repr__ = __str__

    ### comparisons ###
    def __eq__(self, other):
        try:
            return _ufunc_impl.equal(self, asarray(other))
        except RuntimeError:
            # Failed to convert other to array: definitely not equal.
            falsy = torch.full(self.shape, fill_value=False, dtype=bool)
            return asarray(falsy)

    def __neq__(self, other):
        try:
            return _ufunc_impl.not_equal(self, asarray(other))
        except RuntimeError:
            # Failed to convert other to array: definitely not equal.
            falsy = torch.full(self.shape, fill_value=True, dtype=bool)
            return asarray(falsy)

    def __gt__(self, other):
        return _ufunc_impl.greater(self, asarray(other))

    def __lt__(self, other):
        return _ufunc_impl.less(self, asarray(other))

    def __ge__(self, other):
        return _ufunc_impl.greater_equal(self, asarray(other))

    def __le__(self, other):
        return _ufunc_impl.less_equal(self, asarray(other))

    def __bool__(self):
        try:
            return bool(self._tensor)
        except RuntimeError:
            raise ValueError(
                "The truth value of an array with more than one "
                "element is ambiguous. Use a.any() or a.all()"
            )

    def __index__(self):
        if self.size == 1:
            if _dtypes.is_integer(self.dtype):
                return int(self._tensor.item())
        mesg = "only integer scalar arrays can be converted to a scalar index"
        raise TypeError(mesg)

    # HACK : otherwise cannot check array.dtype in _dtypes.dict
    def __hash__(self):
        return id(self)

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
    def __add__(self, other):
        return _ufunc_impl.add(self, asarray(other))

    def __radd__(self, other):
        return _ufunc_impl.add(self, asarray(other))

    def __iadd__(self, other):
        return _ufunc_impl.add(self, asarray(other), out=self)

    # sub, self - other
    def __sub__(self, other):
        return _ufunc_impl.subtract(self, asarray(other))

    def __rsub__(self, other):
        return _ufunc_impl.subtract(self, asarray(other))

    def __isub__(self, other):
        return _ufunc_impl.subtract(self, asarray(other), out=self)

    # mul, self * other
    def __mul__(self, other):
        return _ufunc_impl.multiply(self, asarray(other))

    def __rmul__(self, other):
        return _ufunc_impl.multiply(self, asarray(other))

    def __imul__(self, other):
        return _ufunc_impl.multiply(self, asarray(other), out=self)

    # div, self / other
    def __truediv__(self, other):
        return _ufunc_impl.divide(self, asarray(other))

    def __rtruediv__(self, other):
        return _ufunc_impl.divide(self, asarray(other))

    def __itruediv__(self, other):
        return _ufunc_impl.divide(self, asarray(other), out=self)

    # floordiv, self // other
    def __floordiv__(self, other):
        return _ufunc_impl.floor_divide(self, asarray(other))

    def __rfloordiv__(self, other):
        return _ufunc_impl.floor_divide(self, asarray(other))

    def __ifloordiv__(self, other):
        return _ufunc_impl.floor_divide(self, asarray(other), out=self)

    # power, self**exponent
    def __pow__(self, exponent):
        return _ufunc_impl.float_power(self, asarray(exponent))

    def __rpow__(self, exponent):
        return _ufunc_impl.float_power(self, asarray(exponent))

    def __ipow__(self, exponent):
        return _ufunc_impl.float_power(self, asarray(exponent), out=self)

    # remainder, self % other
    def __mod__(self, other):
        return _ufunc_impl.remainder(self, asarray(other))

    def __rmod__(self, other):
        return _ufunc_impl.remainder(self, asarray(other))

    def __imod__(self, other):
        return _ufunc_impl.remainder(self, asarray(other), out=self)

    # bitwise ops
    # and, self & other
    def __and__(self, other):
        return _ufunc_impl.bitwise_and(self, asarray(other))

    def __rand__(self, other):
        return _ufunc_impl.bitwise_and(self, asarray(other))

    def __iand__(self, other):
        return _ufunc_impl.bitwise_and(self, asarray(other), out=self)

    # or, self | other
    def __or__(self, other):
        return _ufunc_impl.bitwise_or(self, asarray(other))

    def __ror__(self, other):
        return _ufunc_impl.bitwise_or(self, asarray(other))

    def __ior__(self, other):
        return _ufunc_impl.bitwise_or(self, asarray(other), out=self)

    # xor, self ^ other
    def __xor__(self, other):
        return _ufunc_impl.bitwise_xor(self, asarray(other))

    def __rxor__(self, other):
        return _ufunc_impl.bitwise_xor(self, asarray(other))

    def __ixor__(self, other):
        return _ufunc_impl.bitwise_xor(self, asarray(other), out=self)

    # unary ops
    def __invert__(self):
        return _ufunc_impl.invert(self)

    def __abs__(self):
        return _ufunc_impl.absolute(self)

    def __pos__(self):
        return _ufunc_impl.positive(self)

    def __neg__(self):
        return _ufunc_impl.negative(self)

    ### methods to match namespace functions

    def squeeze(self, axis=None):
        if axis == ():
            tensor = self._tensor
        elif axis is None:
            tensor = self._tensor.squeeze()
        else:
            tensor = self._tensor.squeeze(axis)
        return ndarray._from_tensor_and_base(tensor, self)

    @axis_out_keepdims_wrapper
    def argmax(self, axis=None, out=None, *, keepdims=NoValue):
        axis = _helpers.allow_only_single_axis(axis)
        tensor = torch.argmax(self._tensor, axis)
        return tensor

    @axis_out_keepdims_wrapper
    def argmin(self, axis=None, out=None, *, keepdims=NoValue):
        axis = _helpers.allow_only_single_axis(axis)
        tensor = torch.argmin(self._tensor, axis)
        return tensor

    def reshape(self, *shape, order="C"):
        newshape = shape[0] if len(shape) == 1 else shape
        # if sh = (1, 2, 3), numpy allows both .reshape(sh) and .reshape(*sh)
        if order != "C":
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

    def ravel(self, order="C"):
        if order != "C":
            raise NotImplementedError
        return ndarray._from_tensor_and_base(self._tensor.ravel(), self)

    def nonzero(self):
        tensor = self._tensor
        return tuple(asarray(_) for _ in tensor.nonzero(as_tuple=True))

    @axis_out_keepdims_wrapper
    def any(self, axis=None, out=None, keepdims=NoValue, *, where=NoValue):
        if where is not None:
            raise NotImplementedError

        axis = _helpers.allow_only_single_axis(axis)

        if axis is None:
            result = self._tensor.any()
        else:
            result = self._tensor.any(axis)
        return result

    @axis_out_keepdims_wrapper
    def all(self, axis=None, out=None, keepdims=NoValue, *, where=NoValue):
        if where is not None:
            raise NotImplementedError

        axis = _helpers.allow_only_single_axis(axis)

        if axis is None:
            result = self._tensor.all()
        else:
            result = self._tensor.all(axis)
        return result

    @axis_out_keepdims_wrapper
    def max(
        self, axis=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue
    ):
        if where is not None:
            raise NotImplementedError
        if initial is not None:
            raise NotImplementedError

        result = self._tensor.amax(axis)
        return result

    @axis_out_keepdims_wrapper
    def min(
        self, axis=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue
    ):
        if where is not None:
            raise NotImplementedError
        if initial is not None:
            raise NotImplementedError

        result = self._tensor.amin(axis)
        return result

    @axis_out_keepdims_wrapper
    def mean(self, axis=None, dtype=None, out=None, keepdims=NoValue, *, where=NoValue):
        if where is not None:
            raise NotImplementedError

        torch_dtype = _helpers.float_or_default(dtype, self.dtype, enforce_float=True)
        if axis is None:
            result = self._tensor.mean(dtype=torch_dtype)
        else:
            result = self._tensor.mean(dtype=torch_dtype, dim=axis)

        return result

    @axis_out_keepdims_wrapper
    def sum(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=NoValue,
        initial=NoValue,
        where=NoValue,
    ):
        if initial is not None or where is not None:
            raise NotImplementedError

        torch_dtype = _helpers.float_or_default(dtype, self.dtype)
        if axis is None:
            result = self._tensor.sum(dtype=torch_dtype)
        else:
            result = self._tensor.sum(dtype=torch_dtype, dim=axis)

        return result

    @axis_out_keepdims_wrapper
    def prod(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=NoValue,
        initial=NoValue,
        where=NoValue,
    ):
        if initial is not None or where is not None:
            raise NotImplementedError

        axis = _helpers.allow_only_single_axis(axis)

        torch_dtype = _helpers.float_or_default(dtype, self.dtype)
        if axis is None:
            result = self._tensor.prod(dtype=torch_dtype)
        else:
            result = self._tensor.prod(dtype=torch_dtype, dim=axis)

        return result

    @axis_out_keepdims_wrapper
    def std(
        self,
        axis=None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=NoValue,
        *,
        where=NoValue
    ):
        if where is not None:
            raise NotImplementedError

        torch_dtype = _helpers.float_or_default(dtype, self.dtype, enforce_float=True)
        tensor = self._tensor.to(torch_dtype)

        result = tensor.std(dim=axis, correction=ddof)

        return result

    @axis_out_keepdims_wrapper
    def var(
        self,
        axis=None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=NoValue,
        *,
        where=NoValue
    ):
        if where is not None:
            raise NotImplementedError

        torch_dtype = _helpers.float_or_default(dtype, self.dtype, enforce_float=True)
        tensor = self._tensor.to(torch_dtype)

        result = tensor.var(dim=axis, correction=ddof)

        return result

    ### indexing ###
    def __getitem__(self, *args, **kwds):
        t_args = _helpers.to_tensors(*args)
        return ndarray._from_tensor_and_base(
            self._tensor.__getitem__(*t_args, **kwds), self
        )

    def __setitem__(self, index, value):
        value = asarray(value).get()
        return self._tensor.__setitem__(index, value)


# This is the ideally the only place which talks to ndarray directly.
# The rest goes through asarray (preferred) or array.


def array(object, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None):
    _util.subok_not_ok(like, subok)
    if order != "K":
        raise NotImplementedError

    # a happy path
    if isinstance(object, ndarray):
        if copy is False and dtype is None and ndmin <= object.ndim:
            return object

    # lists of ndarrays: [1, [2, 3], ndarray(4)] convert to lists of lists
    if isinstance(object, (list, tuple)):
        a1 = []
        for elem in object:
            if isinstance(elem, ndarray):
                a1.append(elem.get().tolist())
            else:
                a1.append(elem)
        object = a1

    # get the tensor from "object"
    if isinstance(object, ndarray):
        tensor = object._tensor
        base = object
    elif isinstance(object, torch.Tensor):
        tensor = object
        base = None
    else:
        tensor = torch.as_tensor(object)
        base = None

        # At this point, `tensor.dtype` is the pytorch default. Our default may
        # differ, so need to typecast. However, we cannot just do `tensor.to`,
        # because if our desired dtype is wider then pytorch's, `tensor`
        # may have lost precision:

        # int(torch.as_tensor(1e12)) - 1e12 equals -4096 (try it!)

        # Therefore, we treat `tensor.dtype` as a hint, and convert the
        # original object *again*, this time with an explicit dtype.
        dtyp = _dtypes.dtype_from_torch(tensor.dtype)
        default = _dtypes.get_default_dtype_for(dtyp)
        torch_dtype = _dtypes.torch_dtype_from(default)

        tensor = torch.as_tensor(object, dtype=torch_dtype)

    # type cast if requested
    if dtype is not None:
        torch_dtype = _dtypes.torch_dtype_from(dtype)
        tensor = tensor.to(torch_dtype)
        base = None

    # adjust ndim if needed
    ndim_extra = ndmin - tensor.ndim
    if ndim_extra > 0:
        tensor = tensor.view((1,) * ndim_extra + tensor.shape)
        base = None

    # copy if requested
    if copy:
        tensor = tensor.clone()
        base = None

    return ndarray._from_tensor_and_base(tensor, base)


def asarray(a, dtype=None, order=None, *, like=None):
    if order is None:
        order = "K"
    return array(a, dtype=dtype, order=order, like=like, copy=False, ndmin=0)


class asarray_replacer:
    def __init__(self, dispatch="one"):
        if dispatch not in ["one", "two"]:
            raise ValueError("ararray_replacer: unknown dispatch %s" % dispatch)
        self._dispatch = dispatch

    def __call__(self, func):
        if self._dispatch == "one":

            @functools.wraps(func)
            def wrapped(x, *args, **kwds):
                x_tensor = asarray(x).get()
                return asarray(func(x_tensor, *args, **kwds))

            return wrapped
        else:
            raise ValueError


###### dtype routines


def can_cast(from_, to, casting="safe"):
    from_dtype = from_.dtype if isinstance(from_, ndarray) else _dtypes.dtype(from_)
    to_dtype = to.dtype if isinstance(to, ndarray) else _dtypes.dtype(to)

    return _dtypes._can_cast_dict[casting][from_dtype.name][to_dtype.name]


def result_type(*arrays_and_dtypes):
    dtypes = []

    from ._dtypes import issubclass_

    for entry in arrays_and_dtypes:
        if issubclass_(entry, _dtypes._scalar_types.generic):
            dtypes.append(_dtypes.dtype(entry))
        elif isinstance(entry, _dtypes.dtype):
            dtypes.append(entry)
        else:
            dtypes.append(asarray(entry).dtype)

    dtyp = dtypes[0]
    if len(dtypes) == 1:
        return dtyp

    for curr in dtypes[1:]:
        name = _dtypes._result_type_dict[dtyp.name][curr.name]
        dtyp = _dtypes.dtype(name)

    return dtyp
