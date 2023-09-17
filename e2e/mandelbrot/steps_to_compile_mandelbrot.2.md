# JIT-compiling NumPy code in PyTorch land

Since version 2.0, PyTorch has a JIT compiler, designed to speed up tensor manipulations
by transforming PyTorch code into efficient C++ kernels, which are then just-in-time
compiled into machine code.

In version 2.1, the `torch.compile` stack gained the ability to understand and
trace through NumPy code, too. See the [[1]](#1) for a basic
demonstration and tutorial, which showcases the `torch.compile` capabilities on
a rather specific application: a core part of the k-means clustering.

We note however the extreme diversity of applications and subject matter topics
which rely on NumPy. This leads to a very large variability of programming styles,
types and kinds of NumPy programs: there is really no _typical_ NumPy program.
Various applications produce anything from fully vectorized linear algebra heavy
workloads, to iterative algorithms with non-trivial data dependencies, and
pretty much anything in between. 

In this tutorial, we look further into capabilities and limitations of the
`torch.compile` framework, concentrating on an example which cannot be fully
vectorized. In our initial experimentation, we found significant speed-ups
compared to the original NumPy programs. In some cases, small code transformations
can help the compiler a lot, and in this post we consider a worked example of
these sorts of transformations.

Before going into any details, a disclaimer: tricks and workarounds we describe
did work on one specific example; while believe some of them can be useful in a
wider context, they may or may not work in other contexts or require additional steps. 


## The Mandelbrot fractal

For demonstration purposes, we consider a well-known problem where multiple
solutions based on various technologies exist: drawing the Mandelbrot fractal.

To recap, the Mandelbrot fractal is a locus of the complex plane of the variable `c`,
for which the iteration

$$
z_0 = 0, \qquad z_{n+1} = z_n^2 + c, \qquad n=0, 1, \dots
$$

remains bounded: $z_n < \infty$ as $n\to\infty$.

A pure-python solution, while straightforward, would be too slow to be practical
on grids of non-minuscule size. Multiple improvements are known, used various
technologies: NumPy [[2]](#2), Mojo [[3]](#3), and Numba [[4]](#4), to name just
a few.

Here we start from the NumPy version of the algorithm which we take from
[2] verbatim. The code below constructs a grid of the size $x_n \times y_n$,
defined by the box $[x_{min}, x_{max}] \times [y_{min}, y_{max}]$, and for each
point of this grid, $c$, performs `maxiter` steps of the  Mandelbrot iteration:


```python
import numpy as np

# from Chap 4.3, https://www.labri.fr/perso/nrougier/from-python-to-numpy/#temporal-vectorization

def mandelbrot_numpy(xmin, xmax, ymin, ymax, nx, ny, maxiter, horizon=2.0):   
    X = np.linspace(xmin, xmax, nx, dtype=np.float32)
    Y = np.linspace(ymin, ymax, ny, dtype=np.float32)
    C = X + Y[:, None]*1j
    N = np.zeros(C.shape, dtype=int)
    Z = np.zeros(C.shape, dtype=np.complex64)
    for n in range(maxiter):
        I = np.abs(Z) < horizon
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == maxiter-1] = 0
    return Z, N
```


Let's perform the iteration


```python
xmin, xmax, nx = -2.25, +0.75, 3000 // 3 
ymin, ymax, ny = -1.25, +1.25, 2500 // 3
maxiter = 200
horizon = 2.0 ** 40

# iterate
Z, N = mandelbrot_numpy(xmin, xmax, ymin, ymax, nx, ny, maxiter, horizon)
```

and visualize the result:

```python
# plot a nice figure (see the original N. Rougier book for details)
def visualize(Z, N, horizon, nx, ny):
    log_horizon = math.log(horizon, 2)

    dpi = 72
    width = 10
    height = 10*ny/nx
                           
    # visualize (see the original version for details)
    M = np.nan_to_num(N + 1 - np.log2(np.log(abs(Z))) + log_horizon)

    dpi = 72
    width = 10
    height = 10*ny/nx

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False, aspect=1)

    light = colors.LightSource(azdeg=315, altdeg=10)

    plt.imshow(light.shade(M, cmap=plt.cm.hot, vert_exag=1.5,
                           norm = colors.PowerNorm(0.3), blend_mode='hsv'),
               extent=[xmin, xmax, ymin, ymax], interpolation="bicubic")
    ax.set_xticks([])
    ax.set_yticks([])
                                   
visualize(Z, N, horizon, nx, ny)
```
    
![png](mandelbrot.png)


Let's now measure the run time of this iteration:  Running

```python
mandelbrot_numpy(mandelbrot_numpy, 'numpy', xmin, xmax, ymin, ymax, nx, ny, maxiter, horizon)
```

takes `2.1` seconds. This gives us a baseline for further comparisons with
`torch.compile`-d versions.


## Torch.compile the iteration

To improve the run time of the algorithm, we will need to apply several identical
transformations. Let us work them out one by one.


### 1. Extract the inner iteration and only compile it

The first thing to note is that torch dynamo aggressively unrolls loops.
Thus compiling the `mandelbrot` function directly would fully unroll the inner loop
of `maxiter` iterations. This leads to very long compile times and extravagant
memory consumption; on consumer grade machines compilation may even run out of
memory. To avoid this, we extract the inner iteration and will only compile it,
not the whole simulation:


```python
def step(n, C, Z, N, horizon):
    I = np.abs(Z) < horizon
    N[I] = n
    Z[I] = Z[I]**2 + C[I]
    return Z, N


# note the additional argument, `step`
def mandelbrot(xmin, xmax, ymin, ymax, nx, ny, maxiter, step, horizon=2.0):   
    X = np.linspace(xmin, xmax, nx, dtype=np.float32)
    Y = np.linspace(ymin, ymax, ny, dtype=np.float32)
    C = X + Y[:, None]*1j
    N = np.zeros(C.shape, dtype=int)
    Z = np.zeros(C.shape, dtype=np.complex64)
    for n in range(maxiter):
        Z, N = step(n, C, Z, N, horizon)       
    N[N == maxiter-1] = 0
    return Z, N
```


The first attempt is to compile the `step` function as is. We will use the
`fullgraph=True` argument to `torch.compile` so that the compilation errors out on
a construct which fails to compile (in torch.Dynamo parlance: on a _graph break_).
Without that argument, the runtime would have silently delegate to original
numpy on a graph break (in torch.Dynamo parlance: _fall back to eager mode_).



```python
import torch
step_1 = torch.compile(fullgraph=True)(step)

# run it once to warm up the JIT
mandelbrot(xmin, xmax, ymin, ymax, nx, ny, maxiter, step_1, horizon)

---------------------------------------------------------------------------

DnyamicOutputShapeException               Traceback (most recent call last)

<... long and messy traceback truncated ...>

Unsupported: dnyamic shape operator: aten.nonzero.default

from user code:
   File "/tmp/ipykernel_196042/4272750822.py", line 4, in step
    Z[I] = Z[I]**2 + C[I]
```


Indeed, the compilations fails due to a graph break. We've truncated the traceback
(unfortunately messy tracebacks are endemic to JIT compilers), but the gist is
that it cannot properly handle the data dependent nature of the boolean indexing.
Consider a very simple example:


```python
a = np.arange(3)
a[a % 2 == 0], a[a % 2 == 1]

# (array([0, 2]), array([1]))
```

Note that the size of the array indexed by a boolean array depends on the data
values in the indexer array. This sort of behavior is too dynamic for the compiler
to efficiently inline into C++ code (a more precise term here is _lowering_, not inlining), so
that the compiler gives up and falls back to the PyTorch eager mode. Had we not
explicitly ask it to fail instead (by using `fullgraph=True` parameter), the
result would have been not any faster than the original NumPy code.

#### TODO: check after <https://github.com/pytorch/pytorch/pull/107844> lands.


### 2. Remove data dependence from boolean indexing

Note that we only use boolean indices, `I`, as a mask into fixed-size arrays
`Z` and `N`, and only assign elements of `Z` and `N` depending on the mask, `I`.
This allows us to identically rewrite our code to use `np.where` instead of the
boolean indexing:


```python
def step_2(n, C, Z, N, horizon):
    I = np.abs(Z) < horizon
    N = np.where(I, n, N)            # N[I] = n
    Z = np.where(I, Z**2 + C, Z)     # Z[I] = Z[I]**2 + C[I]        
    return Z, N


def mandelbrot_2(xmin, xmax, ymin, ymax, nx, ny, maxiter, step, horizon=2.0):   
    X = np.linspace(xmin, xmax, nx, dtype=np.float32)
    Y = np.linspace(ymin, ymax, ny, dtype=np.float32)
    C = X + Y[:, None]*1j
    N = np.zeros(C.shape, dtype=int)
    Z = np.zeros(C.shape, dtype=np.complex64)
    for n in range(maxiter):
        Z, N = step(n, C, Z, N, horizon)       
    N = np.where(N == maxiter-1, 0, N)        # N[N == maxiter-1] = 0
    return Z, N
```


Compile and run the code

```python
step_2 = torch.compile(step_2)
_ = mandelbrot_2(xmin, xmax, ymin, ymax, nx, ny, maxiter, step=step_2, horizon=horizon)


 UserWarning: Torchinductor does not support code generation for complex operators.
 Performance may be worse than eager.
  warnings.warn(

<... long an messy traceback truncated ...>

```

So we hit the next roadbump: the compiler toolchain (_TorchInductor_) does not
understand complex numbers.

A somewhat puzzling turn of phrase in the traceback text might need explainig:
in _performance maybe worse than eager_, _worse_ means _not better_, as expected;
_eager_ however stands for the _eager mode_, i.e. falling back to uncompiled PyTorch
calls.


### 3. Work around the lack of complex numbers support

The next obstacle is that our `C` array has the `complex64` dtype, and the compiler
toolchain which transpiles our python code into efficient C++ does not handle
complex numbers at the moment. The JIT compiler again falls back to the eager
mode, and the performance is still not any better then the original.

To work around this limitation, we expand our arrays to add a length-2 dimension:
instead of an complex-valued array of shape `(n1, n2)` we use an real-valued
array of shape `(n1, n2, 2)`, where the last dimension holds real and imaginary
parts separately.


```python
x = np.linspace(xmin, xmax, nx, dtype=np.float32)
y = np.linspace(ymin, ymax, ny, dtype=np.float32)

# instead of C = X[None, :] + 1j* Y[:, None]
c = np.stack(np.broadcast_arrays(x[None, :], y[:, None]), axis=-1)
c.shape

   # (833, 1000, 2)
```


We then implement the absolute value and squaring manually:


```python
def abs2(a):
    r"""abs(a)**2 replacement."""
    return a[..., 0]**2 + a[..., 1]**2


def sq2(a):
    r"""a**2 replacement."""
    z = np.empty_like(a)
    z[..., 0] = a[..., 0]**2 - a[..., 1]**2
    z[..., 1] = 2 * a[..., 0] * a[..., 1]
    return z
```

Taking this all together, we have


```python
@torch.compile(fullgraph=True)
def step_3(n, c, Z, N, horizon):
    I = abs2(Z) < horizon**2                      # Note: abs2
    N = np.where(I, n, N)                         
    Z = np.where(I[..., None], sq2(Z) + c, Z)     # Note: sq2
    return Z, N


def mandelbrot_3(xmin, xmax, ymin, ymax, nx, ny,  maxiter, step, horizon=2.0):
    x = np.linspace(xmin, xmax, nx, dtype=np.float32)
    y = np.linspace(ymin, ymax, ny, dtype=np.float32)

    c = np.stack(np.broadcast_arrays(x[None, :], y[:, None]), axis=-1)

    N = np.zeros(c.shape[:-1], dtype=int)
    Z = np.zeros_like(c, dtype=np.float32)       # float32, not complex

    for n in range(maxiter):
        Z, N = step(n, c, Z, N, horizon)
    N = np.where(N == maxiter-1, 0, N)

    Zc = Z[..., 0] + 1j*Z[..., 1]  # restore the complex-valued return
    return Zc, N

```

We now run the simulation with the original value of `maxiter=200`:


```python
_ = mandelbrot_3(xmin, xmax, ymin, ymax, nx, ny, maxiter, step=step_3, horizon=horizon)
```

Warming up the JIT and averaging over multiple runs on a 32-core machine, we get
the run time of about 0.05 seconds, which corresponds to about **40x** performance
improvement over the NumPy original. By default, `torch.dynamo` uses OpenMP for
parallelizing workloads onto all available cores. Limiting the parallelism to only
two cores via the `OMP_NUM_THREADS=2` environment variable, the run time becomes
about 0.45 seconds, which corresponds a speedup of about **4.5x** w.r.t. the
original NumPy code.



## Further improvements


### 4. Chunk the inner loop

In our current approach we only compile a single step of the algorithm. Interfacing
between a compiled kernel and a python frontend of course incurs some overhead.
To quantify and possibly offset this overhead we can split the loop of `maxiter`
iterations and compile compile a chunk of several iterations at once:

```python
@torch.compile
def step_4(n0, c, Z, N, horizon, chunksize):
    for j in range(chunksize):
        n = n0 + j                    # update the iteration counter
        I = abs2(Z) < horizon**2
        N = np.where(I, n, N)
        Z = np.where(I[..., None], sq2(Z) + c, Z)
    return Z, N


def mandelbrot_4(xmin, xmax, ymin, ymax, nx, ny,  maxiter, step, horizon=2.0):
    x = np.linspace(xmin, xmax, nx, dtype=np.float32)
    y = np.linspace(ymin, ymax, ny, dtype=np.float32)
    c = np.stack(np.broadcast_arrays(x[None, :], y[:, None]), axis=-1)

    N = np.zeros(c.shape[:-1], dtype=int)
    Z = np.zeros_like(c, dtype=np.float32)

    chunksize=10                                     # compile this many steps
    n_chunks = maxiter // chunksize

    for i_chunk in range(n_chunks):
        n0 = i_chunk*chunksize                       # update the iteration counter
        Z, N = step(n0, c, Z, N, horizon, chunksize)

    N = np.where(N == maxiter-1, 0, N)
    Zc = Z[..., 0] + 1j*Z[..., 1]
    return Zc, N
```

Here `chunksize=1` is equivalent to our previous version, and `chunksize=maxiter`
compiles (hence unrolls) the whole loop.
Depending on the `chunksize` we observe a variation of both the
compilation time (larger chunk sizes lead to longer compile times), and
the run time (see the Table below---note that run-to-run variations of the
values in the table are rather significant, and are of the order of at least
unity in the last digit given).

Notice that for 32 threads, chunking the loop only decreases performance
(presumably, this is due to communication overhead of the autogenerated code).
For two threads, the optimal chunk size is between 10 and 50.
Specific numbers depend on the communication patterns of autogenerated code and
are clearly workload-specific. 
We will not delve into further parallel performance tuning of our particular problem here.


Table I: Performance boost w.r.t. NumPy. The run-to-run variations of the values
in this table are at least a unity in the last decimal digit given.

| `chunksize` | 2 threads | 32 threads  |
|-------------|-----------|-------------|
|     1       |   4.5     |  42         |
|     2       |   5.3     |  25         |
|    10       |   7.4     |  17         |
|    50       |   5.9     |  14         |
|    100      |   5.8     |  16         |
|    200      |   4.3     |  15         |


### 5. Run your NumPy code on CUDA


Since our approach involves automatically converting NumPy calls into equivalent
PyTorch calls, and given that PyTorch tensors can live on either CPU and or GPU,
we can, in fact, _make our NumPy program run on GPU unchanged_. All we need to
do it to set the PyTorch default device to CUDA:

```
import torch
torch.set_default_device("cuda")
```

With that, the `torch.compile`-d code would
- convert NumPy arrays into PyTorch tensors on GPU
- compile the manipulations with tensors into CUDA calls
- convert the GPU tensors into CPU NumPy arrays on exit (or a graph break)

Note that the last point is inescapable: NumPy arrays are always on CPU and do
not have a notion of 'device'. Therefore, data transfer to/from device happens
automatically and there is no user control over it.

Typically, data transfers can be expected to ruin performance. This, however,
can be mitigated by using small trampoline functions to manually convert numpy arrays
to tensors and back before and after computations. See [[1]](#1) for details and
a demonstration.


# Recap

To summarize, we started with a NumPy program which performs the Mandelbrot iteration, and used `torch.compile` to speed it up. En route, we worked around several peculiarities of the `torch.compile` toolchain, including the lack of complex number support, difficulties with compiling the data dependent control flows, and aggressive unrolling of loops during compilation.

With rather mild rewrites of the original code, we got a massive performance increase and parallelization of originally a single-core NumPy code. Note that the specific performance numbers may rather strongly depend on the problem size and other details (for instance, the chunk size for splitting the iterations). Anecdotally, in other programs we saw speedups ranging from 3 to 300 depending on the problem size relative to the cache size of the target machine. Performance tuning remains an experimental activity and the outcomes very much depend on details.

Finally, we note that our mitigation tricks may be equally applicable to NumPy and PyTorch programs.


# Bonus: compare to Numba


TODO


## References
<a id="1">[1]</a> XXX Mario Lezcano, PyTorch blog post

<a id="2">[2]</a> Nicholas Rougier, "From python to NumPy", Chapter 4.3
https://www.labri.fr/perso/nrougier/from-python-to-numpy/


<a id="3">[3]</a> Shashank Prasanna, "Benchmarking Modular Mojo🔥 and PyTorch torch.compile() on Mandelbrot function"
https://shashankprasanna.com/benchmarking-modular-mojo-and-pytorch-torch.compile-on-mandelbrot-function/index.html#benchmarking-pytorch-cpu-with-torchcompile

<a id="4">[4]</a> Vishwas Saini, "Numba + Cuda Mandelbrot",
https://www.kaggle.com/code/landlord/numba-cuda-mandelbrot/notebook

