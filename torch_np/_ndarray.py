import functools

import torch

from . import _util
from . import _helpers
from . import _dtypes

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
        return self._tensor.stride()   # XXX: byte strides

    @property
    def base(self):
        return self._base

    @property
    def T(self):
        return self.transpose()

    # ctors
    def astype(self, dtype):
        newt = ndarray()
        torch_dtype = _dtypes.torch_dtype_from(dtype)
        newt._tensor = self._tensor.to(torch_dtype)
        return newt

    def copy(self, order='C'):
        if order != 'C':
            raise NotImplementedError
        tensor = self._tensor.clone()   # XXX: clone or detach?
        return ndarray._from_tensor_and_base(tensor, None)

    ###  niceties ###
    def __str__(self):
        return str(self._tensor).replace("tensor", "array_w").replace("dtype=torch.", "dtype=")

    __repr__ = __str__

    ### comparisons ###
    def __eq__(self, other):
        return asarray(self._tensor == asarray(other).get())

    def __neq__(self, other):
        return asarray(self._tensor != asarray(other).get())

    def __gt__(self, other):
        return asarray(self._tensor > asarray(other).get())

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

    # HACK : otherwise cannot check array.dtype in _dtypes.dict
    def __hash__(self):
        return id(self)

    def __float__(self):
        return float(self._tensor)


    ### sequence ###
    def __len__(self):
        return self._tensor.shape[0]

    ### arithmetic ###

    def __add__(self, other):
        other_tensor = asarray(other).get()
        return asarray(self._tensor.__add__(other_tensor))

    def __iadd__(self, other):
        other_tensor = asarray(other).get()
        return asarray(self._tensor.__iadd__(other_tensor))

    def __sub__(self, other):
        other_tensor = asarray(other).get()
        return asarray(self._tensor.__sub__(other_tensor))

    def __mul__(self, other):
        other_tensor = asarray(other).get()
        return asarray(self._tensor.__mul__(other_tensor))

    def __rmul__(self, other):
        other_tensor = asarray(other).get()
        return asarray(self._tensor.__rmul__(other_tensor))

    def __truediv__(self, other):
        other_tensor = asarray(other).get()
        return asarray(self._tensor.__truediv__(other_tensor))

    def __invert__(self):
        return asarray(self._tensor.__invert__())

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
    def max(self, axis=None, out=None, keepdims=NoValue, initial=NoValue,
             where=NoValue):
        if where is not None:
            raise NotImplementedError
        if initial is not None:
            raise NotImplementedError

        result = self._tensor.amax(axis)
        return result

    @axis_out_keepdims_wrapper
    def min(self, axis=None, out=None, keepdims=NoValue, initial=NoValue,
             where=NoValue):
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

        if dtype is None:
            dtype = self.dtype
        if not _dtypes.is_floating(dtype):
            dtype = _dtypes.default_float_type()
        torch_dtype = _dtypes.torch_dtype_from(dtype)

        if axis is None:
            result = self._tensor.mean(dtype=torch_dtype)
        else:
            result = self._tensor.mean(dtype=torch_dtype, dim=axis)

        return result


    @axis_out_keepdims_wrapper
    def sum(self, axis=None, dtype=None, out=None, keepdims=NoValue,
            initial=NoValue, where=NoValue):
        if initial is not None or where is not None:
            raise NotImplementedError

        if dtype is None:
            dtype = self.dtype
        if not _dtypes.is_floating(dtype):
            dtype = _dtypes.default_float_type()
        torch_dtype = _dtypes.torch_dtype_from(dtype)

        if axis is None:
            result = self._tensor.sum(dtype=torch_dtype)
        else:
            result = self._tensor.sum(dtype=torch_dtype, dim=axis)

        return result


    ### indexing ###
    def __getitem__(self, *args, **kwds):
        return ndarray._from_tensor_and_base(self._tensor.__getitem__(*args, **kwds), self)

    def __setitem__(self, index, value):
        value = asarray(value).get()
        return self._tensor.__setitem__(index, value)


def asarray(a, dtype=None, order=None, *, like=None):
    _util.subok_not_ok(like)
    if order is not None:
        raise NotImplementedError

    if isinstance(a, ndarray):
        return a

    if isinstance(a, (list, tuple)):
        # handle lists of ndarrays, [1, [2, 3], ndarray(4)] etc
        a1 = []
        for elem in a:
            if isinstance(elem, ndarray):
                a1.append(elem.get().tolist())
            else:
                a1.append(elem)
    else:
        a1 = a

    torch_dtype = _dtypes.torch_dtype_from(dtype)

    # This and array(...) are the only places which talk to ndarray directly.
    # The rest goes through asarray (preferred) or array.
    out = ndarray()
    tt = torch.as_tensor(a1, dtype=torch_dtype)
    out._tensor = tt
    return out


def array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0,
          like=None):
    _util.subok_not_ok(like, subok)
    if order != 'K':
        raise NotImplementedError

    if isinstance(object, (list, tuple)):
        obj = asarray(object)
        return array(obj, dtype, copy=copy, order=order, subok=subok,
                     ndmin=ndmin, like=like)

    if isinstance(object, ndarray):
        result = object._tensor
    else:
        torch_dtype = _dtypes.torch_dtype_from(dtype)
        result = torch.as_tensor(object, dtype=torch_dtype)

    if copy:
        result = result.clone()    

    ndim_extra = ndmin - result.ndim
    if ndim_extra > 0:
        result = result.reshape((1,)*ndim_extra + result.shape)
    out = ndarray()
    out._tensor = result
    return out



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

        elif self._dispatch == 'two':
            @functools.wraps(func)
            def wrapped(x, y, *args, **kwds):
                x_tensor = asarray(x).get()
                y_tensor = asarray(y).get()
                return asarray(func(x_tensor, y_tensor, *args, **kwds))
            return wrapped

        else:
            raise ValueError


###### dtype routines

def can_cast(from_, to, casting='safe'):
    from_dtype = from_.dtype if isinstance(from_, ndarray) else _dtypes.dtype(from_)
    to_dtype = to.dtype if isinstance(to, ndarray) else _dtypes.dtype(to)

    return _dtypes._can_cast_dict[casting][from_dtype.name][to_dtype.name]


def result_type(*arrays_and_dtypes):
    dtypes = [elem if isinstance(elem, _dtypes.dtype) else asarray(elem).dtype
              for elem in arrays_and_dtypes]

    dtyp = dtypes[0]
    if len(dtypes) == 1:
        return dtyp

    for curr in dtypes[1:]:
        name = _dtypes._result_type_dict[dtyp.name][curr.name]
        dtyp = _dtypes.dtype(name)

    return dtyp


