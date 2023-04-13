"""
Tests which use hypothesis.extra.array_api

These tests aren't specifically for testing Array API adoption!
"""
import cmath
import warnings

import pytest

pytest.importorskip("hypothesis")

import numpy as np
import torch
from hypothesis import given, note
from hypothesis import strategies as st
from hypothesis.errors import HypothesisWarning
from hypothesis.extra import numpy as nps
from hypothesis.extra.array_api import make_strategies_namespace

import torch_np as tnp
from torch_np._dtypes import sctypes
from torch_np.testing import assert_array_equal

__all__ = ["xps"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=HypothesisWarning)
    tnp.bool = tnp.bool_
    xps = make_strategies_namespace(tnp, api_version="2022.12")


default_dtypes = [tnp.bool, tnp.int64, tnp.float64, tnp.complex128]
kind_to_strat = {
    "b": xps.boolean_dtypes(),
    "i": xps.integer_dtypes(),
    "u": xps.unsigned_integer_dtypes(sizes=8),
    "f": xps.floating_dtypes(),
    "c": xps.complex_dtypes(),
}
scalar_dtype_strat = st.one_of(kind_to_strat.values()).map(tnp.dtype)


@pytest.mark.skip(reason="flaky")
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
            lambda d: values_strat.map(lambda v: tnp.asarray(v, dtype=d))
        )
    fill_value = data.draw(values_strat, label="fill_value")
    out = tnp.full(shape, fill_value, **kw)
    assert out.dtype == _dtype
    assert out.shape == shape
    if cmath.isnan(fill_value):
        assert tnp.isnan(out).all()
    else:
        assert (out == fill_value).all()


def integer_array_indices(shape, result_shape) -> st.SearchStrategy[tuple]:
    # See hypothesis.extra.numpy.integer_array_indices()
    # n.b. result_shape only accepts a shape, as opposed to only accepting a strategy
    def array_for(index_shape, size):
        return xps.arrays(
            dtype=xps.integer_dtypes(),
            shape=index_shape,
            elements=st.integers(-size, size - 1),
        )

    return st.tuples(*(array_for(result_shape, size) for size in shape))


@given(
    x=xps.arrays(dtype=xps.integer_dtypes(), shape=xps.array_shapes()),
    data=st.data(),
)
def test_integer_indexing(x, data):
    result_shape = data.draw(xps.array_shapes(), label="result_shape")
    idx = data.draw(integer_array_indices(x.shape, result_shape), label="idx")
    result = x[idx]
    assert result.shape == result_shape


@given(
    np_x=nps.arrays(
        # We specifically use namespaced dtypes to prevent non-native byte-order issues
        dtype=scalar_dtype_strat.map(lambda d: getattr(np, d.name)),
        shape=nps.array_shapes(),
    ),
    data=st.data(),
)
def test_put(np_x, data):
    # We cast arrays from torch_np.asarray as currently it doesn't carry over
    # dtypes. XXX: Remove the below sanity check and subsequent casting when
    # this is fixed.
    assert tnp.asarray(np.zeros(5, dtype=np.int16)).dtype != tnp.int16

    tnp_x = tnp.asarray(np_x.copy()).astype(np_x.dtype.name)

    result_shapes = st.shared(nps.array_shapes())
    ind = data.draw(
        nps.integer_array_indices(np_x.shape, result_shape=result_shapes), label="ind"
    )
    v = data.draw(nps.arrays(dtype=np_x.dtype, shape=result_shapes), label="v")

    tnp_x_copy = tnp_x.copy()
    np.put(np_x, ind, v)
    note(f"(after put) {np_x=}")
    assert_array_equal(tnp_x, tnp_x_copy)  # sanity check

    note(f"{tnp_x=}")
    tnp_ind = []
    for np_indices in ind:
        tnp_indices = tnp.asarray(np_indices).astype(np_indices.dtype.name)
        tnp_ind.append(tnp_indices)
    tnp_ind = tuple(tnp_ind)
    note(f"{tnp_ind=}")
    tnp_v = tnp.asarray(v.copy()).astype(v.dtype.name)
    note(f"{tnp_v=}")
    try:
        tnp.put(tnp_x, tnp_ind, tnp_v)
    except NotImplementedError:
        return
    note(f"(after put) {tnp_x=}")

    assert_array_equal(tnp_x, tnp.asarray(np_x).astype(tnp_x.dtype))
