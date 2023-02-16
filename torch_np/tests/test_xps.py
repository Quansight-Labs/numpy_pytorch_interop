"""
Tests which use hypothesis.extra.array_api

These tests aren't specifically for testing Array API adoption!
"""
import cmath
import warnings

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.errors import HypothesisWarning
from hypothesis.extra.array_api import make_strategies_namespace

import torch_np as np

__all__ = ["xps"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=HypothesisWarning)
    np.bool = np.bool_
    xps = make_strategies_namespace(np, api_version="2022.12")


default_dtypes = [np.bool, np.int64, np.float64, np.complex128]
kind_to_strat = {
    "b": xps.boolean_dtypes(),
    "i": xps.integer_dtypes(),
    "u": xps.unsigned_integer_dtypes(sizes=8),
    "f": xps.floating_dtypes(),
    "c": xps.complex_dtypes(),
}
scalar_dtype_strat = st.one_of(kind_to_strat.values()).map(np.dtype)


@given(shape=xps.array_shapes(), data=st.data())
def test_full(shape, data):
    if data.draw(st.booleans(), label="pass kwargs?"):
        dtype = data.draw(st.none() | scalar_dtype_strat, label="dtype")
        kw = {"dtype": dtype}
    else:
        kw = {}
    _dtype = kw.get("dtype", None) or data.draw(scalar_dtype_strat, label="_dtype")
    values_strat = xps.from_dtype(_dtype)
    if _dtype not in default_dtypes or data.draw(
        st.booleans(), label="fill_value is array?"
    ):
        if specified_dtype := kw.get("dtype", None):
            kind = specified_dtype.name[0]
            values_dtypes_strat = kind_to_strat[kind]
        else:
            values_dtypes_strat = st.just(_dtype)
        values_strat = values_dtypes_strat.flatmap(
            lambda d: values_strat.map(lambda v: np.asarray(v, dtype=d))
        )
    fill_value = data.draw(values_strat, label="fill_value")
    out = np.full(shape, fill_value, **kw)
    assert out.dtype == _dtype
    assert out.shape == shape
    if cmath.isnan(fill_value):
        assert np.isnan(out).all()
    else:
        assert (out == fill_value).all()
