######################################
# Everything here is autogenerated with
# $ python dump_namespace.py
# and lightly edited manually to be importable. Implemented functions are moved
# to wrapper.py and areremoved from the list below.
# Not present: scalars, types, dtypes, `np.r_` etc


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


def fromfile(file, dtype=float, count=-1, sep="", offset=0, *, like=None):
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


def append(arr, values, axis=None):
    raise NotImplementedError


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    raise NotImplementedError


def apply_over_axes(func, a, axes):
    raise NotImplementedError


def argpartition(a, kth, axis=-1, kind="introselect", order=None):
    raise NotImplementedError


def array2string(
    a,
    max_line_width=None,
    precision=None,
    suppress_small=None,
    separator=" ",
    prefix="",
    style=NoValue,
    formatter=None,
    threshold=None,
    edgeitems=None,
    sign=None,
    floatmode=None,
    suffix="",
    *,
    legacy=None,
):
    raise NotImplementedError


def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    raise NotImplementedError


def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    raise NotImplementedError


def asarray_chkfinite(a, dtype=None, order=None):
    raise NotImplementedError


def asfarray(a, dtype="numpy.float64"):
    raise NotImplementedError


def asmatrix(data, dtype=None):
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


def busday_count(
    begindates, enddates, weekmask="1111100", holidays=[], busdaycal=None, out=None
):
    raise NotImplementedError


def busday_offset(
    dates,
    offsets,
    roll="raise",
    weekmask="1111100",
    holidays=None,
    busdaycal=None,
    out=None,
):
    raise NotImplementedError


def byte_bounds(a):
    raise NotImplementedError


def choose(a, choices, out=None, mode="raise"):
    raise NotImplementedError


def clip(a, a_min, a_max, out=None, **kwargs):
    raise NotImplementedError


def common_type(*arrays):
    raise NotImplementedError


def compress(condition, a, axis=None, out=None):
    raise NotImplementedError


def convolve(a, v, mode="full"):
    raise NotImplementedError


def copyto(dst, src, casting="same_kind", where=True):
    raise NotImplementedError


def correlate(a, v, mode="valid"):
    raise NotImplementedError


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    raise NotImplementedError


def datetime_as_string(arr, unit=None, timezone="naive", casting="same_kind"):
    raise NotImplementedError


def delete(arr, obj, axis=None):
    raise NotImplementedError


def deprecate(*args, **kwargs):
    raise NotImplementedError


def deprecate_with_doc(msg):
    raise NotImplementedError


def diag_indices(n, ndim=2):
    raise NotImplementedError


def diag_indices_from(arr):
    raise NotImplementedError


def diagflat(v, k=0):
    raise NotImplementedError


def diagonal(a, offset=0, axis1=0, axis2=1):
    raise NotImplementedError


def digitize(x, bins, right=False):
    raise NotImplementedError


def disp(mesg, device=None, linefeed=True):
    raise NotImplementedError


def dot(a, b, out=None):
    raise NotImplementedError


def ediff1d(ary, to_end=None, to_begin=None):
    raise NotImplementedError


def einsum(*operands, out=None, optimize=False, **kwargs):
    raise NotImplementedError


def einsum_path(*operands, optimize="greedy", einsum_call=False):
    raise NotImplementedError


def extract(condition, arr):
    raise NotImplementedError


def fill_diagonal(a, val, wrap=False):
    raise NotImplementedError


def find_common_type(array_types, scalar_types):
    raise NotImplementedError


def fix(x, out=None):
    raise NotImplementedError


def format_float_positional(
    x,
    precision=None,
    unique=True,
    fractional=True,
    trim="k",
    sign=False,
    pad_left=None,
    pad_right=None,
    min_digits=None,
):
    raise NotImplementedError


def format_float_scientific(
    x,
    precision=None,
    unique=True,
    trim="k",
    sign=False,
    pad_left=None,
    exp_digits=None,
    min_digits=None,
):
    raise NotImplementedError


def fromfunction(function, shape, *, dtype=float, like=None, **kwargs):
    raise NotImplementedError


def fromregex(file, regexp, dtype, encoding=None):
    raise NotImplementedError


def genfromtxt(
    fname,
    dtype=float,
    comments="#",
    delimiter=None,
    skip_header=0,
    skip_footer=0,
    converters=None,
    missing_values=None,
    filling_values=None,
    usecols=None,
    names=None,
    excludelist=None,
    deletechars=" !#$%&'()*+,-./:;<=>?@[\\]^{|}~",
    replace_space="_",
    autostrip=False,
    case_sensitive=True,
    defaultfmt="f%i",
    unpack=None,
    usemask=False,
    loose=True,
    invalid_raise=True,
    max_rows=None,
    encoding="bytes",
    *,
    like=None,
):
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


def is_busday(dates, weekmask="1111100", holidays=None, busdaycal=None, out=None):
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


def lexsort(keys, axis=-1):
    raise NotImplementedError


def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding="ASCII"):
    raise NotImplementedError


def load(
    file,
    mmap_mode=None,
    allow_pickle=False,
    fix_imports=True,
    encoding="ASCII",
    *,
    max_header_size=10000,
):
    raise NotImplementedError


def loadtxt(
    fname,
    dtype=float,
    comments="#",
    delimiter=None,
    converters=None,
    skiprows=0,
    usecols=None,
    unpack=False,
    ndmin=0,
    encoding="bytes",
    max_rows=None,
    *,
    quotechar=None,
    like=None,
):
    raise NotImplementedError


def lookfor(what, module=None, import_modules=True, regenerate=False, output=None):
    raise NotImplementedError


def mafromtxt(fname, **kwargs):
    raise NotImplementedError


def mask_indices(n, mask_func, k=0):
    raise NotImplementedError


def asmatrix(data, dtype=None):
    raise NotImplementedError


def maximum_sctype(t):
    raise NotImplementedError


def may_share_memory(a, b, /, max_work=None):
    raise NotImplementedError


def min_scalar_type(a, /):
    raise NotImplementedError


def mintypecode(typechars, typeset="GDFgdf", default="d"):
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


def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=NoValue):
    raise NotImplementedError


def nanmin(a, axis=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue):
    raise NotImplementedError


def nanpercentile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=NoValue,
    *,
    interpolation=None,
):
    raise NotImplementedError


def nanprod(
    a, axis=None, dtype=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue
):
    raise NotImplementedError


def nanquantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=NoValue,
    *,
    interpolation=None,
):
    raise NotImplementedError


def nanstd(
    a, axis=None, dtype=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue
):
    raise NotImplementedError


def nansum(
    a, axis=None, dtype=None, out=None, keepdims=NoValue, initial=NoValue, where=NoValue
):
    raise NotImplementedError


def nanvar(
    a, axis=None, dtype=None, out=None, ddof=0, keepdims=NoValue, *, where=NoValue
):
    raise NotImplementedError


def obj2sctype(rep, default=None):
    raise NotImplementedError


def outer(a, b, out=None):
    raise NotImplementedError


def packbits(a, /, axis=None, bitorder="big"):
    raise NotImplementedError


def pad(array, pad_width, mode="constant", **kwargs):
    raise NotImplementedError


def partition(a, kth, axis=-1, kind="introselect", order=None):
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


def put(a, ind, v, mode="raise"):
    raise NotImplementedError


def putmask(a, mask, values):
    raise NotImplementedError


def ravel(a, order="C"):
    raise NotImplementedError


def recfromcsv(fname, **kwargs):
    raise NotImplementedError


def recfromtxt(fname, **kwargs):
    raise NotImplementedError


def require(a, dtype=None, requirements=None, *, like=None):
    raise NotImplementedError


def resize(a, new_shape):
    raise NotImplementedError


def roots(p):
    raise NotImplementedError


def safe_eval(source):
    raise NotImplementedError


def save(file, arr, allow_pickle=True, fix_imports=True):
    raise NotImplementedError


def savetxt(
    fname,
    X,
    fmt="%.18e",
    delimiter=" ",
    newline="\n",
    header="",
    footer="",
    comments="# ",
    encoding=None,
):
    raise NotImplementedError


def savez(file, *args, **kwds):
    raise NotImplementedError


def savez_compressed(file, *args, **kwds):
    raise NotImplementedError


def sctype2char(sctype):
    raise NotImplementedError


def searchsorted(a, v, side="left", sorter=None):
    raise NotImplementedError


def select(condlist, choicelist, default=0):
    raise NotImplementedError


def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    suppress=None,
    nanstr=None,
    infstr=None,
    formatter=None,
    sign=None,
    floatmode=None,
    *,
    legacy=None,
):
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


def sort_complex(a):
    raise NotImplementedError


def swapaxes(a, axis1, axis2):
    raise NotImplementedError


def take(a, indices, axis=None, out=None, mode="raise"):
    raise NotImplementedError


def tensordot(a, b, axes=2):
    raise NotImplementedError


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    raise NotImplementedError


def trapz(y, x=None, dx=1.0, axis=-1):
    raise NotImplementedError


def trim_zeros(filt, trim="fb"):
    raise NotImplementedError


def typename(char):
    raise NotImplementedError


def union1d(ar1, ar2):
    raise NotImplementedError


def unique(
    ar,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    *,
    equal_nan=True,
):
    raise NotImplementedError


def unpackbits(a, /, axis=None, count=None, bitorder="big"):
    raise NotImplementedError


def unwrap(p, discont=None, axis=-1, *, period=6.283185307179586):
    raise NotImplementedError


def vdot(a, b, /):
    raise NotImplementedError


def where(condition, x, y, /):
    raise NotImplementedError


def who(vardict=None):
    raise NotImplementedError
