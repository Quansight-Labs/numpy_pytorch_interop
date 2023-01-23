from ._decorators import deco_unary_ufunc_from_impl
from ._detail import _ufunc_impl

__all__ = ['abs', 'absolute', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctanh', 'cbrt', 'ceil', 'conj', 'conjugate', 'cos', 'cosh', 'deg2rad', 'degrees', 'exp', 'exp2', 'expm1', 'fabs', 'floor', 'isfinite', 'isinf', 'isnan', 'log', 'log10', 'log1p', 'log2', 'logical_not', 'negative', 'positive', 'rad2deg', 'radians', 'reciprocal', 'rint', 'sign', 'signbit', 'sin', 'sinh', 'sqrt', 'square', 'tan', 'tanh', 'trunc', 'invert']


absolute = deco_unary_ufunc_from_impl(_ufunc_impl.absolute)
arccos = deco_unary_ufunc_from_impl(_ufunc_impl.arccos)
arccosh = deco_unary_ufunc_from_impl(_ufunc_impl.arccosh)
arcsin = deco_unary_ufunc_from_impl(_ufunc_impl.arcsin)
arcsinh = deco_unary_ufunc_from_impl(_ufunc_impl.arcsinh)
arctan = deco_unary_ufunc_from_impl(_ufunc_impl.arctan)
arctanh = deco_unary_ufunc_from_impl(_ufunc_impl.arctanh)
ceil = deco_unary_ufunc_from_impl(_ufunc_impl.ceil)
conjugate = deco_unary_ufunc_from_impl(_ufunc_impl.conjugate)
cos = deco_unary_ufunc_from_impl(_ufunc_impl.cos)
cosh = deco_unary_ufunc_from_impl(_ufunc_impl.cosh)
deg2rad = deco_unary_ufunc_from_impl(_ufunc_impl.deg2rad)
degrees = deco_unary_ufunc_from_impl(_ufunc_impl.rad2deg)
exp = deco_unary_ufunc_from_impl(_ufunc_impl.exp)
exp2 = deco_unary_ufunc_from_impl(_ufunc_impl.exp2)
expm1 = deco_unary_ufunc_from_impl(_ufunc_impl.expm1)
fabs = deco_unary_ufunc_from_impl(_ufunc_impl.absolute)
floor = deco_unary_ufunc_from_impl(_ufunc_impl.floor)
isfinite = deco_unary_ufunc_from_impl(_ufunc_impl.isfinite)
isinf = deco_unary_ufunc_from_impl(_ufunc_impl.isinf)
isnan = deco_unary_ufunc_from_impl(_ufunc_impl.isnan)
log = deco_unary_ufunc_from_impl(_ufunc_impl.log)
log10 = deco_unary_ufunc_from_impl(_ufunc_impl.log10)
log1p = deco_unary_ufunc_from_impl(_ufunc_impl.log1p)
log2 = deco_unary_ufunc_from_impl(_ufunc_impl.log2)
logical_not = deco_unary_ufunc_from_impl(_ufunc_impl.logical_not)
negative = deco_unary_ufunc_from_impl(_ufunc_impl.negative)
rad2deg = deco_unary_ufunc_from_impl(_ufunc_impl.rad2deg)
radians = deco_unary_ufunc_from_impl(_ufunc_impl.deg2rad)
reciprocal = deco_unary_ufunc_from_impl(_ufunc_impl.reciprocal)
rint = deco_unary_ufunc_from_impl(_ufunc_impl.rint)
sign = deco_unary_ufunc_from_impl(_ufunc_impl.sign)
signbit = deco_unary_ufunc_from_impl(_ufunc_impl.signbit)
sin = deco_unary_ufunc_from_impl(_ufunc_impl.sin)
sinh = deco_unary_ufunc_from_impl(_ufunc_impl.sinh)
sqrt = deco_unary_ufunc_from_impl(_ufunc_impl.sqrt)
square = deco_unary_ufunc_from_impl(_ufunc_impl.square)
tan = deco_unary_ufunc_from_impl(_ufunc_impl.tan)
tanh = deco_unary_ufunc_from_impl(_ufunc_impl.tanh)
trunc = deco_unary_ufunc_from_impl(_ufunc_impl.trunc)

invert = deco_unary_ufunc_from_impl(_ufunc_impl.invert)


cbrt = deco_unary_ufunc_from_impl(_ufunc_impl.cbrt)
positive = deco_unary_ufunc_from_impl(_ufunc_impl.positive)

# numpy has these aliases while torch does not
abs = absolute
conj = conjugate
bitwise_not = invert

