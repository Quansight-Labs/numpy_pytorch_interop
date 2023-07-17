To test our wrapper, we use two strategies:

- port parts of the numpy test suite
- run several small examples which use NumPy and check that the results are identical to original NumPy.

We only run tests in the eager mode by replacing `import numpy as np` by `import torch_np as np`.
Examples we run in both eager and JIT modes. 


For numpy tests, see `torch_np/testing/numpy_tests` folder.

`e2e` folder contains examples we run our wrapper on:

- A toy NN from scratch using numpy
- Build a random maze and find a path in it
- Simulate a diffusion/advection process
- Construct and visualize the Mandelbrot fractal
- Inner operation of the k-means clustering

# JIT compiled mode

The main observation is that `torch.dynamo` unrolls python-level loops. For
iterative algorithms this leads to very long compile times. We therefore
often only compile the inner loop.

## Maze path-finding

The Bellman-Ford algorithm simply does not compile because it contains a
data-dependent loop `while point != start`.


## CFD diffusion/advecton process

We compile the inner loop of the diffusion-advection simulation. While the code
compiles, the performance is on par or slightly worse than the original NumPy.

## Mandelbrot fractal

Results strongly depend on an implementation: a straighforward NumPy implementation
uses complex-valued arrays which are not supported by triton.
Working around this and several other dynamo issues, leads to speedups of about x3 to x5.

The implementation based on the [Mojo benchmark](https://shashankprasanna.com/benchmarking-modular-mojo-and-pytorch-torch.compile-on-mandelbrot-function/index.html#benchmarking-pytorch-cpu-with-torchcompile) allows to compile the inner loop. The performance
increase relative to numpy is substantial and strongly data size and machine
dependent: x8 for smaller inputs and up to x50 for unputs larger than the cache size of the machine.


## K-means clustering

The internal loop of the k-means algorithm compiles into a straighforward
C++ loop and offers up to x30 speedups versus NumPy.


# Eager mode

In short, the main changes to examples are:

- With random number generators, our `random` module is a drop-in replacement to NumPy's, but exact streams of random variates is different. Therefore, to preserve exact bit-to-bit identity, one needs to use NumPy's `random` numbers.
- Interaction with matplotlib: for plotting, we need to convert our wrapper ndarrays to PyTorch tensors or original NumPy arrays. In practice, this will be done automatically by TorchDynamo.


We checked that our examples run on both CPU and GPU by setting the PyTorch global state

```
torch.set_default_device("cuda")    # or "cpu"
```

More specifics for the examples we run:


## A toy NN from scratch using numpy

Origin: https://www.geeksforgeeks.org/implementation-of-neural-network-from-scratch-using-numpy/

Source: `e2e/nn_from_scratch`.


### Tweaks

  - Use the original numpy random stream in both cases to initialize the NN weights


### Results

Results with numpy and torch_np are identical modulo different scalar vs 0D array reprs:
    
    `epochs: 100 ======== acc: 98.64865709132037` with NumPy vs 
    `epochs: 100 ======== acc: array_w(98.6487, dtype=float64)`  with torch_np



## Build a maze and find a path in it

Origin: N. Rougier, From python to numpy, 
https://github.com/rougier/from-python-to-numpy/blob/master/code/maze_numpy.py

Source: `e2e/maze`


### Tweaks


Seed the numpy random generator, use the same random stream.

For plotting with matplotlib, convert torch_numpy arrays to numpy via
`Z = Z.tensor.numpy()`.



## Diffusion/advection

Origin: N. Rougier, From python to numpy,
https://github.com/rougier/from-python-to-numpy/blob/master/code/{smoke_solver,smoke_1,smoke_2}.py

Source: `e2e/smoke`

### Tweaks

 - fix a bug in bool array minus bool array (fails on numpy 1.24)
 - inline np.fromfunction into a call to np.indices


## Mandelbrot fractal

Origin: N. Rougier, From python to numpy, 
https://github.com/rougier/from-python-to-numpy/blob/master/code/mandelbrot.py
https://github.com/rougier/from-python-to-numpy/blob/master/code/mandelbrot_numpy_1.py

source: `e2e/mandelbrot.py`

### Tweaks

  - use `mandelbrot_numpy_1.py` version (slightly slower, but no mgrid)
  - complex abs in float32 runs into https://github.com/pytorch/pytorch/pull/99550

