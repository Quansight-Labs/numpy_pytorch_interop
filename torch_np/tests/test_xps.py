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


@given(shape=xps.array_shapes(), data=st.data())
def test_full(shape, data):
    if data.draw(st.booleans(), label="pass kwargs?"):
        kw = {}
    else:
        dtype = data.draw(st.none() | xps.scalar_dtypes(), label="dtype")
        kw = {"dtype": dtype}
    _dtype = kw.get("dtype", None) or data.draw(
        st.sampled_from([np.bool, np.int64, np.float64, np.complex128]), label="_dtype"
    )
    values_strat = xps.from_dtype(_dtype)
    fill_value = data.draw(
        values_strat | values_strat.map(lambda v: np.asarray(v, dtype=_dtype)),
        label="fill_value",
    )
    out = np.full(shape, fill_value, **kw)
    if kw.get("dtype", None) is None and not isinstance(fill_value, np.ndarray):
        if isinstance(fill_value, bool):
            assert out.dtype == np.bool
        elif isinstance(fill_value, int):
            assert out.dtype == np.int64
        elif isinstance(fill_value, float):
            assert out.dtype == np.float64
        else:
            assert isinstance(fill_value, complex)  # sanity check
            assert out.dtype == np.complex128
    else:
        assert out.dtype == _dtype
    assert out.shape == shape
    if cmath.isnan(fill_value):
        assert np.isnan(out).all()
    else:
        assert (out == fill_value).all()
