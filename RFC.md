# A PyTorch - NumPy compatibility layer

**Authors:**
* @ev-br
* @lezcano
* @rgommers

## Summary
This RFC describes a proposal for a translation layer from NumPy into PyTorch.
In simple terms, this accounts for implementing most of NumPy's API (`ndarray`,
the `numpy`, `numpy.linalg`, `numpy.fft`  modules, etc) using `torch.Tensor`
and PyTorch ops as backend.


The this project has a main goal as per the
[initial design document](https://docs.google.com/document/d/1gdUDgZNbumFORRcUaZUVw790CtNYweAM20C1fbWMNd8):
1. Make TorchDynamo understand NumPy calls

The work is currently being done at [numpy_pytorch_interop](https://github.com/Quansight-Labs/numpy_pytorch_interop/).


## Motivation

### Introductory examples

Consider the following snippet:
```python
import numpy as np

x = np.random.randn(3, 4)
y = np.random.randn(4, 3)
z = np.dot(x, y)
w = z.sum()
```

When we trace this program with the compat layer, the semantics of the
program would stay the same, but the implementation would be equivalent to

```python
import torch
x = torch.randn(3, 4, dtype=torch.float64)
y = torch.randn(4, 3, dtype=torch.float64)
z = torch.matmul(x, y)
w = z.sum()
```

Here, we can already spot a couple differences between NumPy and PyTorch.
The most obvious one is that the default dtype in NumPy is `float64` rather than
`float32`. The less obvious is very sneakily hiding in the last line.

```python
>>> type(w)
<class 'numpy.float64'>
```

Reductions and similar operations in NumPy return the infamous NumPy scalars.
We'll discuss these and other NumPy quirks and how we dealt with them in the
[design decision section](#design-decisions).


Let's now have a look at a toy example of how this layer would be used.
```python
import torch
import numpy as np
t1 = torch.tensor([1, 3, 5])
t2 = torch.exp(t)
# Now say the user has some code lying around which uses NumPy:
def fn(x, y):
    return np.multiply(x, y).sum()

result = fn(t1, t2)
t_results = torch.empty(5, dtype=torch.float64)
t_results[0] = result  # store the result in a torch.Tensor
```

Note that this code mixing NumPy and PyTorch already works in eager mode with
CPU tensors, as `torch.Tensor` implements the `__array__` method. Now, the
compatibility layer allows us to trace through it. In order to do that, there
would be no necessary changes, other than simply ask `torch.compile` to trace
through it:

```python
@compile
def fn(x, y):
    return np.multiply(x, y).sum()
```

### Design decisions

The two main ideas driving the design of this compatibility layer are the following:

1. The behavior of the layer should be as close to that of NumPy as possible
2. The layer follows the most recent NumPy release

The following design decisions follow from these:

**Default dtypes**. One of the most common issues that bites people when migrating their
codebases from NumPy to JAX is the default dtype changing from `float64` to
`float32`. So much so that this is noted as one of
[JAX's shap edges](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).
Following the spirit of making everything match NumPy by default, we choose the
NumPy defaults whenever the `dtype` was not made explicit in a factory function.

**TODO(Lezcano)**: I just realized that we do not have a clean way to change
the default dtype of `torch_np` to those from PyTorch. We should implement
that utility flag, similar to
[`torch.set_default_dtype`](https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html).
Perhaps call it `torch_np.use_torch_defaults()` and then add a way for users
to be able to set their own int/float/complex defaults.

**NumPy scalars**. NumPy's type system is tricky. At first sight, it looks
like PyTorch's, but with few more dtypes like `np.uint16` or `np.longdouble`.
Upon closer inspection, one finds that it also has
[NumPy scalar](https://numpy.org/doc/stable/reference/arrays.scalars.html) objects.
NumPy scalars are similar to Python scalars but with a set width. NumPy scalars
are NumPy's preferred return class for reductions and other operations that
return just one element. NumPy scalars do not play particularly well with
computations on devices like GPUs, as they live on CPU. Implementing NumPy
scalars would mean that we need to synchronize after every `sum()` call, which
would be terrible performance-wise. In this implementation, we choose to represent
NumPy scalars as 0-D arrays. This may cause small divergences in some cases like

```python
>>> np.int32(2) * [1, 2, 3]       # scalar decays to a python int
[1, 2, 3, 1, 2, 3]

>>> np.asarray(2) * [1, 2, 3]     # zero-dim array is an array-like
array([2, 4, 6])
```

but we don't expect these to pose a big issue in practice. Note that in the
proposed implementation `np.int32(2)` would return the same as `np.asarray(2)`.

**Type promotion**. Another not-so-well-known fact of NumPy's cast system is
that it is data-dependent. Python scalars can be used in pretty much any NumPy
operation, being able to call any operation that accepts a 0-D array with a
Python scalar. If you provide an operation with a Python scalar, these will be
casted to the smallest dtype they can be represented in, and then, they will
participate in type promotion. This allows for for some rather interesting behaviour
```python
>>> np.asarray([1], dtype=np.int8) + 127
array([128], dtype=int8)
>>> np.asarray([1], dtype=np.int8) + 128
array([129], dtype=int16)
```
This dependent type promotion will be deprecated NumPy 2.0, and will be
replaced with [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html).
For simplicity and to be forward-looking, we chose to implement the
type promotion behaviour proposed in NEP 50, which is much closer to that of
Pytorch.

Note that the decision of going with NEP 50 complements the previous one of
returning 0-D arrays in place of NumPy scalars as, currently, 0-D arrays do not
participate in type promotion in NumPy (but will do in NumPy 2.0 under NEP 50):
```python
int64_0d_array = np.array(1, dtype=np.int64)
np.result_type(np.int8, int64_0d_array) == np.int8
```

**Versioning**. It should be clear from the previous points that NumPy has a
fair amount of questionable and legacy pain points. It is for this reason that
we decided that rather than fighting these, we would declare that the compat
layer follows the behavior of Numpy's most recent release (even, in some cases,
of NumPy 2.0). Given the stability of NumPy's API and how battle-tested its
main functions are, we do not expect this to become a big maintenance burden.
If anything, it should make our lives easier, as some parts of NumPy will soon
be simplified, saving us the pain of having to implement all the pre-existing
corner-cases.

For reference NumPy 2.0 is expected to land at the end of this year.

**Randomness**. PyTorch and NumPy use different random number generation methods.
In particular, NumPy recently moved to a [new API](https://numpy.org/doc/stable/reference/random/index.html)
with a `Generator` object which has sampling methods on it. The current compat.
layer does not implement this new API, as the default bit generator in NumPy is a
`PCG64`, while on PyTorch we use a `MT19937` on CPU and a `Philox`. From this, it
follows that this API will not give any reproducibility guarantees when it comes
to randomness.


## The `torch_np` module

The bulk of the work went into implementing a system that allows us to
implement NumPy operations in terms of those of PyTorch. The main design goals
here were

1. Implement *most* of NumPy's API
2. Preserve NumPy semantics as much as possible

We say *most* of NumPy's API, because NumPy's API is not only massive, but also
there are parts of it which cannot be implemented in PyTorch. For example,
NumPy has support for arrays of string, datetime, structured and other dtypes.
Negative strides are other example of a feature that is just not supported in PyTorch.
We put together a list of things that are out of the scope of this project in the
[following issue](https://github.com/Quansight-Labs/numpy_pytorch_interop/issues/73).

For the bulk of the functions, we started by prioritizing the most common
operations. Then, when bringing tests from the NumPy test suite, we would triage
and prioritize how important was to fix each failure we found. Iterating this
process, we ended up with a small list of differences between the NumPy and the
PyTorch API which we prioritized by hand. That list and the prioritization
discussion can be found in [this issue](https://github.com/Quansight-Labs/numpy_pytorch_interop/issues/87).

**Visibility of the module** For simplicity, this RFC assumes that the
`torch_np` module will not be public, as the decision for it to be made public
was met with different opinions.
We discuss these in the section [unresolved questions](#unresolved-questions).

### Annotation-based preprocessing

NumPy accepts virtually anything that smells like an array as an input.

```python
>>> np.add(1, 3)
4
>>> np.add([1., 2., 3.], 5)
array([6., 7., 8.])
>>> np.concatenate([[1, 2, 3], [4, 5, 6]])
array([1, 2, 3, 4, 5, 6])
```

NumPy calls all these objects `array_like` objects.
To implement NumPy in terms of PyTorch, for any operation we would need to map
inputs into tensors, perform the operations, and then wrap the tensor into
a `torch_np.ndarray` (more on this class later).

To avoid all this code repetition, we implement the functions in two steps.

First, we implement functions with the NumPy signature, but assuming that in
place of NumPy-land elements (`np.array`, array-like functions, `np.dtype`s, etc)
they simply accept `torch.Tensor` and PyTorch-land objects and return
`torch.Tensor`s. For example, we would implement `np.diag` as

```python
def diag(v, k=0):
    return torch.diag(v, k)
```

In this layer, if a NumPy function is composite (calls other NumPy functions
internally), we can simply vendor its implementation, and have it call our
PyTorch-land implementations of these functions. In other words, at this level,
functions are composable, as they are simply regular PyTorch functions.
All these implementations are internal, and are not meant to be seen or used
by the final user.

The second step is then done via type annotations and a decorator. Each type
annotation has an associated function from NumPy-land into PyTorch-land. This
function converts the set of inputs accepted by NumPy for that argument into a
PyTorch-land object (think a `torch.Tensor` or a PyTorch dtype). For example,
for `np.diag` we would write

```python
def diag(v: ArrayLike, k=0):
    return torch.diag(v, k)
```

Then, we wrap these Python-land functions with a `normalizer` decorator and
expose them in the `torch_np` module. This decorator is in charge of gathering
all the inputs at runtime and normalizing them (i.e., converting `torch_np`
objects to PyTorch counterparts) according to their annotations.

We currently have four annotations (and small variations of them):
- `ArrayLike`: The input can be a `torch_np.array`, a list of lists, a
  scalar, or anything that NumPy would accept. It returns a `torch.Tensor`.
- `DTypeLike`: Takes a `torch_np` dtype and returns a PyTorch dtype.
- `AxisLike`: Takes anything that can be accepted as an axis (e.g. a tuple or
  an `ndarray`) and returns a tuple.
- `OutArray`: Asserts that the input is a `torch_np.ndarray`. This is used
  to implement the `out` arg.

Note that none of the code in this implementation  makes use of NumPy. We are
writing `torch_np.ndarray` above to make more explicit our intents, but there
shouldn't be any ambiguity.

**OBS(Lezcano)**: `DTypeLike` should be `Optional[DTypeLike]`

**Implmenting out**: In PyTorch, the `out` kwarg is, as the name says, a
keyword-only argument. It is for this reason that, in PrimTorch, we were able
to implement it as [a decorator](https://github.com/pytorch/pytorch/blob/ce4df4cc596aa10534ac6d54912f960238264dfd/torch/_prims_common/wrappers.py#L187-L282).
This is not the case in NumPy. In NumPy `out` is a positional arg that is often
interleaved with other parameters. This is the reason why we use the `OutArray`
annotation to mark these. We then implement the `out` semantics in the `@normalizer`
wrapper in a generic way.

**Ufuncs and reductions**: Ufuncs (unary and binary) and reductions are two
sets of functions that are particularly regular. For these functions, we
implement their args in a generic way as a preprocessing or postprocessing.

**The ndarray class** Once we have all the free functions implemented as
functions form `torch_np.ndarray`s to `torch_np.ndarray`s, implementing the
methods from the `ndarray` class is rather simple. We simply register all the
free functions as methods or dunder methods appropriately. We also forward the
properties to the properties within the PyTorch tensor and we are done.
This creates a circular dependency which we break with a local import.

### Testing

The testing of the framework was done via ~~copying~~ vendoring tests from the
NumPy test suit. Then, we would replace the NumPy imports with `torch_np`
imports. The failures on these tests were then triaged and discussed the
priority of fixing each of them.

In the (near) future, we plan to get some real world examples and run them
through the library, to test its coverage and correctness.

### Limitations

A number of known limitations are tracked in the second part of the
[OP of this issue](https://github.com/Quansight-Labs/numpy_pytorch_interop/issues/73).
There are some more in [this issue](https://github.com/Quansight-Labs/numpy_pytorch_interop/issues/86).
When landing this RFC, we will create a comprehensive document with the differences
between NumPy and `torch_np`.

### Beyond Plain NumPy

**GPU**. The current implementation has just been implemented and tested on
CPU. We expect GPU coverage to be as good as the coverage we have with CPU
matching GPU. If the original tensors are on GPU, the whole execution should
be performed on the GPU.

**TODO(Lezcano)**. We should probably test CUDA on the tests.

**Gradients**. We have not tested gradient tracking either as we are still to
find some good examples on which to test it, but it should be a simple
corollary of all this effort. If the original tensors fed into the function do
have `requires_grad=True`, the tensors will track the gradients of the internal
implementation and then the user could differentiate through the NumPy code.

**TODO(Lezcano)**. Picking up simple NumPy programs from the internet would be good for these autograd tests.

### Bindings to TorchDyamo

The bindings for NumPy at the TorchDynamo level are currently being developed at [#95849](https://github.com/pytorch/pytorch/pull/95849).


## Unresolved Questions

A question was left open in the initial discussion. Should the module
`torch_np` be publicly exposed as `torch.numpy` or not?

A few arguments in favor of making it public:
* People could use it in their NumPy programs just by changing the import to
  `import torch.numpy as np`. This could be a selling point similar to JAX's
  `jax.numpy`, which could incentivize adoption.
* People would not need to use the whole PyTorch 2.0 stack to start using
  PyTorch in their codebases
  * See [this experiment in scikit-learn](https://github.com/scikit-learn/scikit-learn/pull/25956)
    where they got a 7x speed-up on CPU on a layer just by using `torch.linalg`.
* Since the layer is rather thin and in pure Python, if there are bugs,
  external contributors could easily help fixing them or extend the supported
  functionality.

A few arguments against:
* The compat introduces a number of type conversions that may produce somewhat
  slow code when used in eager mode.
  * [Note] Keeping this in mind, we tried to use in the implementations as few
    operators as possible, to make it reasonably fast in eager mode.
* Exposing `torch.numpy` would create a less performant secondary entry point
  to many of the functions in PyTorch. This could be a trap for new users.
