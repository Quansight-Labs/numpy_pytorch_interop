import functools

import torch

from . import _util
from . import _dtypes

NoValue = None

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
        torch_dtype = _dtypes.torch_dtype_from_dtype(dtype)
        newt._tensor = self._tensor.to(torch_dtype)
        return newt

    def copy(self, order='C'):
        if order != 'C':
            raise NotImplementedError
        tensor = self._tensor.clone()   # XXX: clone or detach?
        return ndarray._from_tensor_and_base(tensor, None)

    ###  niceties ###
    def __str__(self):
        # FIXME: prints dtype=torch.float64 etc
        return str(self._tensor).replace("tensor", "array_w").replace("dtype=torch.", "dtype=")

    __repr__ = __str__

    ### comparisons ###
    def __eq__(self, other):
        return asarray(self._tensor == asarray(other).get())

    def __neq__(self, other):
        return asarray(self._tensor != asarray(other).get())

    def __bool__(self):
        try:
            return bool(self._tensor)
        except RuntimeError:
            raise ValueError("The truth value of an array with more than one "
                             "element is ambiguous. Use a.any() or a.all()")

    # HACK : otherwise cannot check array.dtype in _dtypes.dict
    def __hash__(self):
        return id(self)

    def __float__(self):
        return float(self._tensor)


    ### arithmetic ###

    def __add__(self, other):
        return self._tensor.__add__(other)

    def __iadd__(self, other):
        return self._tensor.__add__(other)

    def __sub__(self, other):
        return self._tensor.__sub__(other)

    def __mul__(self, other):
        return self._tensor.__mul__(other)

    ### methods to match namespace functions

    def squeeze(self, axis=None):
        return squeeze(self._tensor, axis)

    def argmax(self, axis=None, out=None, *, keepdims=NoValue):
        return argmax(self._tensor, axis, out=out, keepdims=keepdims)

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

    ### indexing ###
    def __getitem__(self, *args, **kwds):
        return ndarray._from_tensor_and_base(self._tensor.__getitem__(*args, **kwds), self)

    def __setitem__(self, index, value):
        return self._tensor.__setitem__(index, value)


def asarray(a, dtype=None, order=None, *, like=None):
    _util.subok_not_ok(like)
    if order is not None:
        raise NotImplementedError

    if isinstance(a, ndarray):
        return a

    torch_dtype = _dtypes.torch_dtype_from(dtype)

    # This and array(...) are the only places which talk to ndarray directly.
    # The rest goes through asarray (preferred) or array.
    out = ndarray()
    tt = torch.as_tensor(a, dtype=torch_dtype)
    out._tensor = tt
    return out


def array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0,
          like=None):
    _util.subok_not_ok(like, subok)
    if order != 'K':
        raise NotImplementedError

    if isinstance(object, ndarray):
        result = object._tensor
    else:
        result = torch.as_tensor(object, dtype=dtype)

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


