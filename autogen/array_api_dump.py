def abs(x: "Array", /) -> "Array":
    raise NotImplementedError


def acos(x: "Array", /) -> "Array":
    raise NotImplementedError


def acosh(x: "Array", /) -> "Array":
    raise NotImplementedError


def add(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def all(
    x: "Array",
    /,
    *,
    axis: "Optional[Union[int, Tuple[int, ...]]]" = None,
    keepdims: "bool" = False,
) -> "Array":
    raise NotImplementedError


def any(
    x: "Array",
    /,
    *,
    axis: "Optional[Union[int, Tuple[int, ...]]]" = None,
    keepdims: "bool" = False,
) -> "Array":
    raise NotImplementedError


def arange(
    start: "Union[int, float]",
    /,
    stop: "Optional[Union[int, float]]" = None,
    step: "Union[int, float]" = 1,
    *,
    dtype: "Optional[Dtype]" = None,
    device: "Optional[Device]" = None,
) -> "Array":
    raise NotImplementedError


def argmax(
    x: "Array", /, *, axis: "Optional[int]" = None, keepdims: "bool" = False
) -> "Array":
    raise NotImplementedError


def argmin(
    x: "Array", /, *, axis: "Optional[int]" = None, keepdims: "bool" = False
) -> "Array":
    raise NotImplementedError


def argsort(
    x: "Array",
    /,
    *,
    axis: "int" = -1,
    descending: "bool" = False,
    stable: "bool" = True,
) -> "Array":
    raise NotImplementedError


def asarray(
    obj: "Union[Array, bool, int, float, NestedSequence[bool | int | float], SupportsBufferProtocol]",
    /,
    *,
    dtype: "Optional[Dtype]" = None,
    device: "Optional[Device]" = None,
    copy: "Optional[Union[bool, np._CopyMode]]" = None,
) -> "Array":
    raise NotImplementedError


def asin(x: "Array", /) -> "Array":
    raise NotImplementedError


def asinh(x: "Array", /) -> "Array":
    raise NotImplementedError


def astype(x: "Array", dtype: "Dtype", /, *, copy: "bool" = True) -> "Array":
    raise NotImplementedError


def atan(x: "Array", /) -> "Array":
    raise NotImplementedError


def atan2(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def atanh(x: "Array", /) -> "Array":
    raise NotImplementedError


def bitwise_and(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def bitwise_invert(x: "Array", /) -> "Array":
    raise NotImplementedError


def bitwise_left_shift(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def bitwise_or(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def bitwise_right_shift(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def bitwise_xor(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def broadcast_arrays(*arrays: "Array") -> "List[Array]":
    raise NotImplementedError


def broadcast_to(x: "Array", /, shape: "Tuple[int, ...]") -> "Array":
    raise NotImplementedError


def can_cast(from_: "Union[Dtype, Array]", to: "Dtype", /) -> "bool":
    raise NotImplementedError


def ceil(x: "Array", /) -> "Array":
    raise NotImplementedError


def concat(
    arrays: "Union[Tuple[Array, ...], List[Array]]", /, *, axis: "Optional[int]" = 0
) -> "Array":
    raise NotImplementedError


def cos(x: "Array", /) -> "Array":
    raise NotImplementedError


def cosh(x: "Array", /) -> "Array":
    raise NotImplementedError


def divide(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def empty(
    shape: "Union[int, Tuple[int, ...]]",
    *,
    dtype: "Optional[Dtype]" = None,
    device: "Optional[Device]" = None,
) -> "Array":
    raise NotImplementedError


def empty_like(
    x: "Array", /, *, dtype: "Optional[Dtype]" = None, device: "Optional[Device]" = None
) -> "Array":
    raise NotImplementedError


def equal(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def exp(x: "Array", /) -> "Array":
    raise NotImplementedError


def expand_dims(x: "Array", /, *, axis: "int") -> "Array":
    raise NotImplementedError


def expm1(x: "Array", /) -> "Array":
    raise NotImplementedError


def eye(
    n_rows: "int",
    n_cols: "Optional[int]" = None,
    /,
    *,
    k: "int" = 0,
    dtype: "Optional[Dtype]" = None,
    device: "Optional[Device]" = None,
) -> "Array":
    raise NotImplementedError


def finfo(type: "Union[Dtype, Array]", /) -> "finfo_object":
    raise NotImplementedError


def flip(
    x: "Array", /, *, axis: "Optional[Union[int, Tuple[int, ...]]]" = None
) -> "Array":
    raise NotImplementedError


def floor(x: "Array", /) -> "Array":
    raise NotImplementedError


def floor_divide(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def from_dlpack(x: "object", /) -> "Array":
    raise NotImplementedError


def full(
    shape: "Union[int, Tuple[int, ...]]",
    fill_value: "Union[int, float]",
    *,
    dtype: "Optional[Dtype]" = None,
    device: "Optional[Device]" = None,
) -> "Array":
    raise NotImplementedError


def full_like(
    x: "Array",
    /,
    fill_value: "Union[int, float]",
    *,
    dtype: "Optional[Dtype]" = None,
    device: "Optional[Device]" = None,
) -> "Array":
    raise NotImplementedError


def greater(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def greater_equal(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def iinfo(type: "Union[Dtype, Array]", /) -> "iinfo_object":
    raise NotImplementedError


def isfinite(x: "Array", /) -> "Array":
    raise NotImplementedError


def isinf(x: "Array", /) -> "Array":
    raise NotImplementedError


def isnan(x: "Array", /) -> "Array":
    raise NotImplementedError


def less(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def less_equal(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def linspace(
    start: "Union[int, float]",
    stop: "Union[int, float]",
    /,
    num: "int",
    *,
    dtype: "Optional[Dtype]" = None,
    device: "Optional[Device]" = None,
    endpoint: "bool" = True,
) -> "Array":
    raise NotImplementedError


def log(x: "Array", /) -> "Array":
    raise NotImplementedError


def log10(x: "Array", /) -> "Array":
    raise NotImplementedError


def log1p(x: "Array", /) -> "Array":
    raise NotImplementedError


def log2(x: "Array", /) -> "Array":
    raise NotImplementedError


def logaddexp(x1: "Array", x2: "Array") -> "Array":
    raise NotImplementedError


def logical_and(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def logical_not(x: "Array", /) -> "Array":
    raise NotImplementedError


def logical_or(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def logical_xor(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def matmul(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def matrix_transpose(x: "Array", /) -> "Array":
    raise NotImplementedError


def max(
    x: "Array",
    /,
    *,
    axis: "Optional[Union[int, Tuple[int, ...]]]" = None,
    keepdims: "bool" = False,
) -> "Array":
    raise NotImplementedError


def mean(
    x: "Array",
    /,
    *,
    axis: "Optional[Union[int, Tuple[int, ...]]]" = None,
    keepdims: "bool" = False,
) -> "Array":
    raise NotImplementedError


def meshgrid(*arrays: "Array", indexing: "str" = "xy") -> "List[Array]":
    raise NotImplementedError


def min(
    x: "Array",
    /,
    *,
    axis: "Optional[Union[int, Tuple[int, ...]]]" = None,
    keepdims: "bool" = False,
) -> "Array":
    raise NotImplementedError


def multiply(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def negative(x: "Array", /) -> "Array":
    raise NotImplementedError


def nonzero(x: "Array", /) -> "Tuple[Array, ...]":
    raise NotImplementedError


def not_equal(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def ones(
    shape: "Union[int, Tuple[int, ...]]",
    *,
    dtype: "Optional[Dtype]" = None,
    device: "Optional[Device]" = None,
) -> "Array":
    raise NotImplementedError


def ones_like(
    x: "Array", /, *, dtype: "Optional[Dtype]" = None, device: "Optional[Device]" = None
) -> "Array":
    raise NotImplementedError


def permute_dims(x: "Array", /, axes: "Tuple[int, ...]") -> "Array":
    raise NotImplementedError


def positive(x: "Array", /) -> "Array":
    raise NotImplementedError


def pow(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def prod(
    x: "Array",
    /,
    *,
    axis: "Optional[Union[int, Tuple[int, ...]]]" = None,
    dtype: "Optional[Dtype]" = None,
    keepdims: "bool" = False,
) -> "Array":
    raise NotImplementedError


def remainder(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def reshape(x: "Array", /, shape: "Tuple[int, ...]") -> "Array":
    raise NotImplementedError


def result_type(*arrays_and_dtypes: "Union[Array, Dtype]") -> "Dtype":
    raise NotImplementedError


def roll(
    x: "Array",
    /,
    shift: "Union[int, Tuple[int, ...]]",
    *,
    axis: "Optional[Union[int, Tuple[int, ...]]]" = None,
) -> "Array":
    raise NotImplementedError


def round(x: "Array", /) -> "Array":
    raise NotImplementedError


def sign(x: "Array", /) -> "Array":
    raise NotImplementedError


def sin(x: "Array", /) -> "Array":
    raise NotImplementedError


def sinh(x: "Array", /) -> "Array":
    raise NotImplementedError


def sort(
    x: "Array",
    /,
    *,
    axis: "int" = -1,
    descending: "bool" = False,
    stable: "bool" = True,
) -> "Array":
    raise NotImplementedError


def sqrt(x: "Array", /) -> "Array":
    raise NotImplementedError


def square(x: "Array", /) -> "Array":
    raise NotImplementedError


def squeeze(x: "Array", /, axis: "Union[int, Tuple[int, ...]]") -> "Array":
    raise NotImplementedError


def stack(
    arrays: "Union[Tuple[Array, ...], List[Array]]", /, *, axis: "int" = 0
) -> "Array":
    raise NotImplementedError


def std(
    x: "Array",
    /,
    *,
    axis: "Optional[Union[int, Tuple[int, ...]]]" = None,
    correction: "Union[int, float]" = 0.0,
    keepdims: "bool" = False,
) -> "Array":
    raise NotImplementedError


def subtract(x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def sum(
    x: "Array",
    /,
    *,
    axis: "Optional[Union[int, Tuple[int, ...]]]" = None,
    dtype: "Optional[Dtype]" = None,
    keepdims: "bool" = False,
) -> "Array":
    raise NotImplementedError


def tan(x: "Array", /) -> "Array":
    raise NotImplementedError


def tanh(x: "Array", /) -> "Array":
    raise NotImplementedError


def tensordot(
    x1: "Array",
    x2: "Array",
    /,
    *,
    axes: "Union[int, Tuple[Sequence[int], Sequence[int]]]" = 2,
) -> "Array":
    raise NotImplementedError


def tril(x: "Array", /, *, k: "int" = 0) -> "Array":
    raise NotImplementedError


def triu(x: "Array", /, *, k: "int" = 0) -> "Array":
    raise NotImplementedError


def trunc(x: "Array", /) -> "Array":
    raise NotImplementedError


def unique_all(x: "Array", /) -> "UniqueAllResult":
    raise NotImplementedError


def unique_counts(x: "Array", /) -> "UniqueCountsResult":
    raise NotImplementedError


def unique_inverse(x: "Array", /) -> "UniqueInverseResult":
    raise NotImplementedError


def unique_values(x: "Array", /) -> "Array":
    raise NotImplementedError


def var(
    x: "Array",
    /,
    *,
    axis: "Optional[Union[int, Tuple[int, ...]]]" = None,
    correction: "Union[int, float]" = 0.0,
    keepdims: "bool" = False,
) -> "Array":
    raise NotImplementedError


def vecdot(x1: "Array", x2: "Array", /, *, axis: "int" = -1) -> "Array":
    raise NotImplementedError


def where(condition: "Array", x1: "Array", x2: "Array", /) -> "Array":
    raise NotImplementedError


def zeros(
    shape: "Union[int, Tuple[int, ...]]",
    *,
    dtype: "Optional[Dtype]" = None,
    device: "Optional[Device]" = None,
) -> "Array":
    raise NotImplementedError


def zeros_like(
    x: "Array", /, *, dtype: "Optional[Dtype]" = None, device: "Optional[Device]" = None
) -> "Array":
    raise NotImplementedError
