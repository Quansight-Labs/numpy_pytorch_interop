"""A thin pytorch / numpy compat layer.

Things imported from here have numpy-compatible signatures but operate on
pytorch tensors.
"""
import numpy as np
import torch

# Things to decide on (punt for now)
#
# 1. Q: What are the return types of wrapper functions: plain torch.Tensors or
#       wrapper ndarrays.
#    A: Tensors, apparently
#
# 2. Q: Default dtypes: numpy defaults to float64, pytorch defaults to float32
#    A: Stick to pytorch defaults?
#    NB: numpy recommends `dtype=float`?
#
# 3. Q: Masked arrays. Record, structured arrays.
#    A: Ignore for now
#
# 4. Q: What are the defaults for pytorch-specific args not present in numpy signatures?
#       device=..., requires_grad=... etc 
#    A: ignore, keep whatever they are from inputs; test w/various combinations 
#
# 5. Q: What is the useful action for numpy-specific arguments? e.g. like=... 
#    A: raise ValueErrors?
#       initial=... can be useful though, punt on
#       where=...   punt on for now


# TODO
# 1. Mapping of the numpy layout ('C', 'K' etc) to torch layout / memory_format.
# 2. np.dtype <-> torch.dtype
# 3. numpy type casting rules (to be cleaned up in numpy: follow old or new)


NoValue = None

###### array creation routines

def asarray(a, dtype=None, order=None, *, like=None):
    if order is not None or like is not None:
        raise NotImplementedError
    return torch.asarray(a, dtype=dtype)


def array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0,
          like=None):
    if order != 'K' or subok or like is not None:
        raise NotImplementedError
    result = torch.asarray(object, dtype=dtype, copy=copy)
    ndim_extra = ndmin - result.ndim
    if ndim_extra > 0:
        result = result.reshape((1,)*ndim_extra + result.shape)
    return result


def copy(a, order='K', subok=False):
    if order != 'K' or subok:
        raise NotImplementedError
    return torch.clone(a)


from torch import atleast_1d, atleast_2d, atleast_3d


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
             axis=0):
    if axis !=0 or retstep or not endpoint:
        raise NotImplementedError
    # XXX: raises TypeError if start or stop are not scalars
    return torch.linspace(start, stop, num, dtype=dtype)


def empty(shape, dtype=float, order='C', *, like=None):
    if order != 'C' or like is not None:
        raise NotImplementedError
    return torch.empty(shape, dtype=dtype)


def empty_like(prototype, dtype=None, order='K', subok=True, shape=None):
    if order != 'K' or not subok:
        raise NotImplementedError
    result = torch.empty(prototype, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def full(shape, fill_value, dtype=None, order='C', *, like=None):
    if order != 'C' or like is not None:
        raise NotImplementedError
    return torch.full(shape, fill_value, dtype=dtype)


def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):
    if order != 'K' or not subok:
        raise NotImplementedError
    result = torch.full_like(a, fill_value, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def ones(shape, dtype=None, order='C', *, like=None):
    if order != 'C' or like is not None:
        raise NotImplementedError
    return torch.ones(shape, dtype=dtype)


def ones_like(a, dtype=None, order='K', subok=True, shape=None):
    if order != 'K' or not subok:
        raise NotImplementedError
    result = torch.ones_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


# XXX: dtype=float
def zeros(shape, dtype=float, order='C', *, like=None):
    if order != 'C' or like is not None:
        raise NotImplementedError
    return torch.zeros(shape, dtype=dtype)


def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    if order != 'K' or not subok:
        raise NotImplementedError
    result = torch.zeros_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


# XXX: dtype=float
def eye(N, M=None, k=0, dtype=float, order='C', *, like=None):
    if order != 'C' or like is not None:
        raise NotImplementedError
    if M is None:
        M = N
    s = M - k if k >= 0 else N + k
    z = torch.zeros(N, M, dtype=dtype)
    z.diagonal(k).fill_(1)
    return z


def identity(n, dtype=None, *, like=None):
    if like is not None:
        raise NotImplementedError
    return torch.eye(n, dtype=dtype)


###### misc/unordered

def prod(a, axis=None, dtype=None, out=None, keepdims=NoValue,
         initial=NoValue, where=NoValue):
    if initial is not None or where is not None:
        raise NotImplementedError
    if axis is None:
        if keepdims is not None:
            raise NotImplementedError
        return torch.prod(a, dtype=dtype, out=out)
    elif is_sequence(axis):
        raise NotImplementedError
    return torch.prod(a, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


def corrcoef(x, y=None, rowvar=True, bias=NoValue, ddof=NoValue, *, dtype=None):
    if bias is not None or ddof is not None:
        # deprecated in NumPy
        raise NotImplementedError
    if rowvar is False:
        x = x.T
    if y is not None:
        raise NotImplementedError
    if dtype is not None:
        x = x.type(dtype)
    return torch.corrcoef(x)


def concatenate(ar_tuple, axis=0, out=None, dtype=None, casting="same_kind"):
    if casting != "same_kind":
        raise NotImplementedError   # XXX
    if dtype is not None:
        # XXX: map numpy dtypes
        ar_tuple = tuple(ar.type(dtype) for ar in ar_typle)
    return torch.cat(ar_tuple, axis, out=out)


def squeeze(a, axis=None):
    if axis is None:
        return torch.squeeze(a)
    else:
        return torch.squeeze(a, axis)


def bincount(x, /, weights=None, minlength=0):
    return torch.bincount(x, weights, minlength)


###### module-level queries of object properties

def ndim(a):
    return torch.as_tensor(a).ndim


def shape(a):
    return tuple(torch.as_tensor(a).shape)


def size(a, axis=None):
    if axis is None:
        return torch.as_tensor(a).numel()
    else:
        return torch.as_tensor(a).shape[axis]


###### shape manipulations and indexing

def reshape(a, newshape, order='C'):
    if order != 'C':
        raise NotImplementedError
    return torch.reshape(a, newshape)


def broadcast_to(array, shape, subok=False):
    if subok:
        raise NotImplementedError
    return torch.broadcast_to(array, shape)


from torch import broadcast_shapes


def broadcast_arrays(*args, subok=False):
    if subok:
        raise NotImplementedError
    return torch.broadcast_tensors(*args)


def unravel_index(indices, shape, order='C'):
# cf https://github.com/pytorch/pytorch/pull/66687
# this version is from 
# https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/3
    if order != 'C':
        raise NotImplementedError
    result = []
    for index in indices:
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        result.append(tuple(reversed(out)))
    return result


def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    # XXX: not available in pytorch, implement
    return sum(idx*dim for idx, dim in zip(multi_index, dims))



###### reductions

def argmax(a, axis=None, out=None, *, keepdims=NoValue):
    if axis is None:
        result = torch.argmax(a, keepdims=bool(keepdims))
    else:
        result = torch.argmax(a, axis, keepdims=bool(keepdims))
    if out is not None:
        out.copy_(result)
    return result


##### math functions

from _unary import *
abs = absolute


def angle(z, deg=False):
    result = torch.angle(z)
    if deg:
        result *= 180 / torch.pi
    return result

from torch import imag, real


def real_if_close(a, tol=100):
    if not torch.is_complex(a):
        return a
    if torch.abs(torch.imag) < tol * torch.finfo(a.dtype).eps:
        return torch.real(a)
    else:
        return a


def iscomplex(x):
    if torch.is_complex(x):
        return torch.as_tensor(x).imag != 0
    result = torch.zeros_like(x, dtype=torch.bool)
    return result[()]


def isreal(x):
    if torch.is_complex(x):
        return torch.as_tensor(x).imag == 0
    result = torch.zeros_like(x, dtype=torch.bool)
    return result[()]


def iscomplexobj(x):
    return torch.is_complex(x)


def isrealobj(x):
    return not torch.is_complex(x)


def isneginf(x, out=None):
    return torch.isneginf(x, out=out)


def isposinf(x, out=None):
    return torch.isposinf(x, out=out)


def i0(x):
    return torch.special.i0(x)


###### mapping from numpy API objects to wrappers from this module ######

mapping = {
    # array creation routines
    np.asarray : asarray,
    np.array : array,
    np.copy : copy,
    np.atleast_1d : atleast_1d,
    np.atleast_2d : atleast_2d,
    np.atleast_3d : atleast_3d,

    np.empty : empty,
    np.empty_like : empty_like,
    np.full : full,
    np.full_like : full_like,
    np.ones : ones,
    np.ones_like : ones_like,
    np.zeros : zeros,
    np.zeros_like : zeros_like,
    np.identity : identity,
    np.eye : eye,

    # utilities
    np.linspace : linspace,
    np.prod : prod,
    np.corrcoef : corrcoef,    # XXX: numpy two-arg version
    np.concatenate: concatenate,
    np.squeeze : squeeze,
    np.bincount : bincount,
    np.argmax : argmax,
    np.ndim : ndim,
    np.shape : shape,
    np.size : size,
    np.reshape : reshape,

    # broadcasting and indexing
    np.broadcast_to : broadcast_to,
    np.broadcast_shapes : broadcast_shapes,
    np.broadcast_arrays: broadcast_arrays,
    np.unravel_index : unravel_index,
    np.ravel_multi_index : ravel_multi_index,
    # math functions
    np.angle : angle,
    np.real : real,
    np.imag : imag,
    np.real_if_close : real_if_close,
    np.iscomplex: iscomplex,
    np.iscomplexobj: iscomplexobj,
    np.isreal: isreal,
    np.isrealobj: isrealobj,
    np.isposinf : isposinf,
    np.isneginf : isneginf,
    np.i0 : i0,
}
# XXX: automate populating this dict?

##################### ndarray class ###########################

class ndarray:
    def __init__(self, *args, **kwds):
        self._tensor = torch.Tensor(*args, **kwds)

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
        return self._tensor.dtype

    @property
    def strides(self):
        return self._tensor.stride()   # XXX: byte strides

    ### arithmetic ###

    def __add__(self, other):
        return self._tensor__add__(other)

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

    def reshape(self, shape, order='C'):
        return reshape(self._tensor, shape, order)



# https://github.com/numpy/numpy/blob/v1.23.0/numpy/distutils/misc_util.py#L497-L504
def is_sequence(seq):
    if is_string(seq):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True

######################################
# Everything below is autogenerated with
# $ python dump_namespace.py
# and lightly edited manually to be importable. Implemented functions are moved
# above and removed from the list below.
# Not present: scalars, types, dtypes, `np.r_` etc

def arange(start, stop, step, dtype=None, *, like=None):
    raise NotImplementedError



def asanyarray(a, dtype=None, order=None, *, like=None):
    raise NotImplementedError


def ascontiguousarray(a, dtype=None, *, like=None):
    raise NotImplementedError

def asfortranarray(a, dtype=None, *, like=None):
    raise NotImplementedError

def compare_chararrays(a, b, cmp_op, rstrip):
    raise NotImplementedError

def datetime_data(dtype, /):
    raise NotImplementedError


def _fastCopyAndTranspose(a):
    raise NotImplementedError

def frombuffer(buffer, dtype=float, count=-1, offset=0, *, like=None):
    raise NotImplementedError

def fromfile(file, dtype=float, count=-1, sep='', offset=0, *, like=None):
    raise NotImplementedError

def fromiter(iter, dtype, count=-1, *, like=None):
    raise NotImplementedError

def frompyfunc(func, /, nin, nout, *, identity):
    raise NotImplementedError

def fromstring(string, dtype=float, count=-1, *, sep, like=None):
    raise NotImplementedError

def geterrobj():
    raise NotImplementedError

def promote_types(type1, type2):
    raise NotImplementedError

def set_numeric_ops(op1, op2, *args, **kwargs):
    raise NotImplementedError

def seterrobj(errobj):
    raise NotImplementedError



def alen(a):
    raise NotImplementedError

def all(a, axis=None, out=None, keepdims=NoValue, *, where=NoValue):
    raise NotImplementedError

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    raise NotImplementedError

def alltrue(*args, **kwargs):
    raise NotImplementedError

def amax(a, axis=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    raise NotImplementedError

def amin(a, axis=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    raise NotImplementedError


def any(a, axis=None, out=None, keepdims=NoValue, *, where=NoValue):
    raise NotImplementedError

def append(arr, values, axis=None):
    raise NotImplementedError

def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    raise NotImplementedError

def apply_over_axes(func, a, axes):
    raise NotImplementedError


def argmin(a, axis=None, out=None, *, keepdims=NoValue):
    raise NotImplementedError

def argpartition(a, kth, axis=-1, kind='introselect', order=None):
    raise NotImplementedError

def argsort(a, axis=-1, kind=None, order=None):
    raise NotImplementedError

def argwhere(a):
    raise NotImplementedError

def around(a, decimals=0, out=None):
    raise NotImplementedError

def array2string(a, max_line_width=None, precision=None, suppress_small=None, separator=' ', prefix='', style=NoValue, formatter=None, threshold=None, edgeitems=None, sign=None, floatmode=None, suffix='', *, legacy=None):
    raise NotImplementedError

def array_equal(a1, a2, equal_nan=False):
    raise NotImplementedError

def array_equiv(a1, a2):
    raise NotImplementedError

def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    raise NotImplementedError

def array_split(ary, indices_or_sections, axis=0):
    raise NotImplementedError

def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    raise NotImplementedError

def asarray_chkfinite(a, dtype=None, order=None):
    raise NotImplementedError

def asfarray(a, dtype='numpy.float64'):
    raise NotImplementedError

def asmatrix(data, dtype=None):
    raise NotImplementedError

def asscalar(a):
    raise NotImplementedError


def average(a, axis=None, weights=None, returned=False, *, keepdims=NoValue):
    raise NotImplementedError

def bartlett(M):
    raise NotImplementedError

def base_repr(number, base=2, padding=0):
    raise NotImplementedError

def binary_repr(num, width=None):
    raise NotImplementedError


def blackman(M):
    raise NotImplementedError

def block(arrays):
    raise NotImplementedError

def bmat(obj, ldict=None, gdict=None):
    raise NotImplementedError

def busday_count(begindates, enddates, weekmask='1111100', holidays=[], busdaycal=None, out=None):
    raise NotImplementedError

def busday_offset(dates, offsets, roll='raise', weekmask='1111100', holidays=None, busdaycal=None, out=None):
    raise NotImplementedError

def byte_bounds(a):
    raise NotImplementedError

def can_cast(from_, to, casting='safe'):
    raise NotImplementedError

def choose(a, choices, out=None, mode='raise'):
    raise NotImplementedError

def clip(a, a_min, a_max, out=None, **kwargs):
    raise NotImplementedError

def column_stack(tup):
    raise NotImplementedError

def common_type(*arrays):
    raise NotImplementedError

def compress(condition, a, axis=None, out=None):
    raise NotImplementedError


def convolve(a, v, mode='full'):
    raise NotImplementedError



def copyto(dst, src, casting='same_kind', where=True):
    raise NotImplementedError


def correlate(a, v, mode='valid'):
    raise NotImplementedError

def count_nonzero(a, axis=None, *, keepdims=False):
    raise NotImplementedError

def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None):
    raise NotImplementedError

def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    raise NotImplementedError

def cumprod(a, axis=None, dtype=None, out=None):
    raise NotImplementedError

def cumproduct(*args, **kwargs):
    raise NotImplementedError

def cumsum(a, axis=None, dtype=None, out=None):
    raise NotImplementedError

def datetime_as_string(arr, unit=None, timezone='naive', casting='same_kind'):
    raise NotImplementedError

def delete(arr, obj, axis=None):
    raise NotImplementedError

def deprecate(*args, **kwargs):
    raise NotImplementedError

def deprecate_with_doc(msg):
    raise NotImplementedError

def diag(v, k=0):
    raise NotImplementedError

def diag_indices(n, ndim=2):
    raise NotImplementedError

def diag_indices_from(arr):
    raise NotImplementedError

def diagflat(v, k=0):
    raise NotImplementedError

def diagonal(a, offset=0, axis1=0, axis2=1):
    raise NotImplementedError

def diff(a, n=1, axis=-1, prepend=NoValue, append=NoValue):
    raise NotImplementedError

def digitize(x, bins, right=False):
    raise NotImplementedError

def disp(mesg, device=None, linefeed=True):
    raise NotImplementedError

def dot(a, b, out=None):
    raise NotImplementedError

def dsplit(ary, indices_or_sections):
    raise NotImplementedError

def dstack(tup):
    raise NotImplementedError

def ediff1d(ary, to_end=None, to_begin=None):
    raise NotImplementedError

def einsum(*operands, out=None, optimize=False, **kwargs):
    raise NotImplementedError

def einsum_path(*operands, optimize='greedy', einsum_call=False):
    raise NotImplementedError


def expand_dims(a, axis):
    raise NotImplementedError

def extract(condition, arr):
    raise NotImplementedError



def fill_diagonal(a, val, wrap=False):
    raise NotImplementedError

def find_common_type(array_types, scalar_types):
    raise NotImplementedError

def fix(x, out=None):
    raise NotImplementedError

def flatnonzero(a):
    raise NotImplementedError

def flip(m, axis=None):
    raise NotImplementedError

def fliplr(m):
    raise NotImplementedError

def flipud(m):
    raise NotImplementedError

def format_float_positional(x, precision=None, unique=True, fractional=True, trim='k', sign=False, pad_left=None, pad_right=None, min_digits=None):
    raise NotImplementedError

def format_float_scientific(x, precision=None, unique=True, trim='k', sign=False, pad_left=None, exp_digits=None, min_digits=None):
    raise NotImplementedError

def fromfunction(function, shape, *, dtype=float, like=None, **kwargs):
    raise NotImplementedError

def fromregex(file, regexp, dtype, encoding=None):
    raise NotImplementedError



def genfromtxt(fname, dtype=float, comments='#', delimiter=None, skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=" !#$%&'()*+,-./:;<=>?@[\\]^{|}~", replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes', *, like=None):
    raise NotImplementedError

def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    raise NotImplementedError

def get_array_wrap(*args):
    raise NotImplementedError

def get_include():
    raise NotImplementedError

def get_printoptions():
    raise NotImplementedError

def getbufsize():
    raise NotImplementedError

def geterr():
    raise NotImplementedError

def geterrcall():
    raise NotImplementedError

def gradient(f, *varargs, axis=None, edge_order=1):
    raise NotImplementedError

def hamming(M):
    raise NotImplementedError

def hanning(M):
    raise NotImplementedError

def histogram(a, bins=10, range=None, normed=None, weights=None, density=None):
    raise NotImplementedError

def histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None):
    raise NotImplementedError

def histogram_bin_edges(a, bins=10, range=None, weights=None):
    raise NotImplementedError

def histogramdd(sample, bins=10, range=None, normed=None, weights=None, density=None):
    raise NotImplementedError

def hsplit(ary, indices_or_sections):
    raise NotImplementedError

def hstack(tup):
    raise NotImplementedError







def in1d(ar1, ar2, assume_unique=False, invert=False):
    raise NotImplementedError

def indices(dimensions, dtype=int, sparse=False):
    raise NotImplementedError

def inner(a, b, /):
    raise NotImplementedError

def insert(arr, obj, values, axis=None):
    raise NotImplementedError

def interp(x, xp, fp, left=None, right=None, period=None):
    raise NotImplementedError

def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    raise NotImplementedError

def is_busday(dates, weekmask='1111100', holidays=None, busdaycal=None, out=None):
    raise NotImplementedError

def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    raise NotImplementedError


def isfortran(a):
    raise NotImplementedError

def isin(element, test_elements, assume_unique=False, invert=False):
    raise NotImplementedError


def isscalar(element):
    raise NotImplementedError

def issctype(rep):
    raise NotImplementedError

def issubclass_(arg1, arg2):
    raise NotImplementedError

def issubdtype(arg1, arg2):
    raise NotImplementedError

def issubsctype(arg1, arg2):
    raise NotImplementedError

def iterable(y):
    raise NotImplementedError

def ix_(*args):
    raise NotImplementedError

def kaiser(M, beta):
    raise NotImplementedError

def kron(a, b):
    raise NotImplementedError

def lexsort(keys, axis=-1):
    raise NotImplementedError


def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    raise NotImplementedError

def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII', *, max_header_size=10000):
    raise NotImplementedError

def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, quotechar=None, like=None):
    raise NotImplementedError

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    raise NotImplementedError

def lookfor(what, module=None, import_modules=True, regenerate=False, output=None):
    raise NotImplementedError

def mafromtxt(fname, **kwargs):
    raise NotImplementedError

def mask_indices(n, mask_func, k=0):
    raise NotImplementedError

def asmatrix(data, dtype=None):
    raise NotImplementedError

def amax(a, axis=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    raise NotImplementedError

def maximum_sctype(t):
    raise NotImplementedError

def may_share_memory(a, b, /, max_work=None):
    raise NotImplementedError

def mean(a, axis=None, dtype=None, out=None, keepdims=NoValue, *, where=NoValue):
    raise NotImplementedError

def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    raise NotImplementedError

def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    raise NotImplementedError

def amin(a, axis=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    raise NotImplementedError

def min_scalar_type(a, /):
    raise NotImplementedError

def mintypecode(typechars, typeset='GDFgdf', default='d'):
    raise NotImplementedError

def moveaxis(a, source, destination):
    raise NotImplementedError

def msort(a):
    raise NotImplementedError

def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    raise NotImplementedError

def nanargmax(a, axis=None, out=None, *, keepdims=NoValue):
    raise NotImplementedError

def nanargmin(a, axis=None, out=None, *, keepdims=NoValue):
    raise NotImplementedError

def nancumprod(a, axis=None, dtype=None, out=None):
    raise NotImplementedError

def nancumsum(a, axis=None, dtype=None, out=None):
    raise NotImplementedError

def nanmax(a, axis=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    raise NotImplementedError

def nanmean(a, axis=None, dtype=None, out=None, keepdims=NoValue, *, where=NoValue):
    raise NotImplementedError

def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=NoValue):
    raise NotImplementedError

def nanmin(a, axis=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    raise NotImplementedError

def nanpercentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=NoValue, *, interpolation=None):
    raise NotImplementedError

def nanprod(a, axis=None, dtype=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    raise NotImplementedError

def nanquantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=NoValue, *, interpolation=None):
    raise NotImplementedError

def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue):
    raise NotImplementedError

def nansum(a, axis=None, dtype=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    raise NotImplementedError

def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue):
    raise NotImplementedError


def nonzero(a):
    raise NotImplementedError

def obj2sctype(rep, default=None):
    raise NotImplementedError


def outer(a, b, out=None):
    raise NotImplementedError

def packbits(a, /, axis=None, bitorder='big'):
    raise NotImplementedError

def pad(array, pad_width, mode='constant', **kwargs):
    raise NotImplementedError

def partition(a, kth, axis=-1, kind='introselect', order=None):
    raise NotImplementedError

def percentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *, interpolation=None):
    raise NotImplementedError

def piecewise(x, condlist, funclist, *args, **kw):
    raise NotImplementedError

def place(arr, mask, vals):
    raise NotImplementedError

def poly(seq_of_zeros):
    raise NotImplementedError

def polyadd(a1, a2):
    raise NotImplementedError

def polyder(p, m=1):
    raise NotImplementedError

def polydiv(u, v):
    raise NotImplementedError

def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    raise NotImplementedError

def polyint(p, m=1, k=None):
    raise NotImplementedError

def polymul(a1, a2):
    raise NotImplementedError

def polysub(a1, a2):
    raise NotImplementedError

def polyval(p, x):
    raise NotImplementedError

def printoptions(*args, **kwargs):
    raise NotImplementedError

def product(*args, **kwargs):
    raise NotImplementedError

def ptp(a, axis=None, out=None, keepdims=NoValue):
    raise NotImplementedError

def put(a, ind, v, mode='raise'):
    raise NotImplementedError

def put_along_axis(arr, indices, values, axis):
    raise NotImplementedError

def putmask(a, mask, values):
    raise NotImplementedError

def quantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *, interpolation=None):
    raise NotImplementedError

def ravel(a, order='C'):
    raise NotImplementedError




def recfromcsv(fname, **kwargs):
    raise NotImplementedError

def recfromtxt(fname, **kwargs):
    raise NotImplementedError

def repeat(a, repeats, axis=None):
    raise NotImplementedError

def require(a, dtype=None, requirements=None, *, like=None):
    raise NotImplementedError

def resize(a, new_shape):
    raise NotImplementedError

def result_type(*arrays_and_dtypes):
    raise NotImplementedError

def roll(a, shift, axis=None):
    raise NotImplementedError

def rollaxis(a, axis, start=0):
    raise NotImplementedError

def roots(p):
    raise NotImplementedError

def rot90(m, k=1, axes=(0, 1)):
    raise NotImplementedError

def round_(a, decimals=0, out=None):
    raise NotImplementedError

def round_(a, decimals=0, out=None):
    raise NotImplementedError

def vstack(tup):
    raise NotImplementedError

def safe_eval(source):
    raise NotImplementedError

def save(file, arr, allow_pickle=True, fix_imports=True):
    raise NotImplementedError

def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None):
    raise NotImplementedError

def savez(file, *args, **kwds):
    raise NotImplementedError

def savez_compressed(file, *args, **kwds):
    raise NotImplementedError

def sctype2char(sctype):
    raise NotImplementedError

def searchsorted(a, v, side='left', sorter=None):
    raise NotImplementedError

def select(condlist, choicelist, default=0):
    raise NotImplementedError

def set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None, sign=None, floatmode=None, *, legacy=None):
    raise NotImplementedError

def set_string_function(f, repr=True):
    raise NotImplementedError

def setbufsize(size):
    raise NotImplementedError

def setdiff1d(ar1, ar2, assume_unique=False):
    raise NotImplementedError

def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    raise NotImplementedError

def seterrcall(func):
    raise NotImplementedError

def setxor1d(ar1, ar2, assume_unique=False):
    raise NotImplementedError

def shares_memory(a, b, max_work=None):
    raise NotImplementedError

def show():
    raise NotImplementedError

def sinc(x):
    raise NotImplementedError

def sometrue(*args, **kwargs):
    raise NotImplementedError

def sort(a, axis=-1, kind=None, order=None):
    raise NotImplementedError

def sort_complex(a):
    raise NotImplementedError

def split(ary, indices_or_sections, axis=0):
    raise NotImplementedError

def stack(arrays, axis=0, out=None):
    raise NotImplementedError

def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue):
    raise NotImplementedError

def sum(a, axis=None, dtype=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    raise NotImplementedError

def swapaxes(a, axis1, axis2):
    raise NotImplementedError

def take(a, indices, axis=None, out=None, mode='raise'):
    raise NotImplementedError

def take_along_axis(arr, indices, axis):
    raise NotImplementedError

def tensordot(a, b, axes=2):
    raise NotImplementedError

def tile(A, reps):
    raise NotImplementedError

def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    raise NotImplementedError

def transpose(a, axes=None):
    raise NotImplementedError

def trapz(y, x=None, dx=1.0, axis=-1):
    raise NotImplementedError

def tri(N, M=None, k=0, dtype=float, *, like=None):
    raise NotImplementedError

def tril(m, k=0):
    raise NotImplementedError

def tril_indices(n, k=0, m=None):
    raise NotImplementedError

def tril_indices_from(arr, k=0):
    raise NotImplementedError

def trim_zeros(filt, trim='fb'):
    raise NotImplementedError

def triu(m, k=0):
    raise NotImplementedError

def triu_indices(n, k=0, m=None):
    raise NotImplementedError

def triu_indices_from(arr, k=0):
    raise NotImplementedError

def typename(char):
    raise NotImplementedError

def union1d(ar1, ar2):
    raise NotImplementedError

def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True):
    raise NotImplementedError

def unpackbits(a, /, axis=None, count=None, bitorder='big'):
    raise NotImplementedError

def unwrap(p, discont=None, axis=-1, *, period=6.283185307179586):
    raise NotImplementedError

def vander(x, N=None, increasing=False):
    raise NotImplementedError

def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue):
    raise NotImplementedError

def vdot(a, b, /):
    raise NotImplementedError

def vsplit(ary, indices_or_sections):
    raise NotImplementedError

def vstack(tup):
    raise NotImplementedError

def where(condition, x, y, /):
    raise NotImplementedError

def who(vardict=None):
    raise NotImplementedError

