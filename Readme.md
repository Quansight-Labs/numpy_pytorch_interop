To test our wrapper, we use two strategies:

- port parts of the numpy test suite
- run several small examples which use NumPy and check that the results are identical to original NumPy.

We only run tests and examples in the eager mode by replacing `import numpy as np` by `import torch_np as np`.

For numpy tests, see `torch_np/testing/numpy_tests` folder.

`e2e` folder contains examples we run our wrapper on:

- A toy NN from scratch using numpy
- Build a random maze and find a path in it
- Simulate a diffusion/advection process
- Construct and visualize the Mandelbrot fractal

In short, the main changes to examples are:

- With random number generators, our `random` module is a drop-in replacement to NumPy's, but exact streams of random variates is different. Therefore, to preserve exact bit-to-bit identity, one needs to use NumPy's `random` numbers.
- Interaction with matplotlib: for plotting, we need to convert our wrapper ndarrays to PyTorch tensors or original NumPy arrays. This of course is expected, as extrenal libraries do not know about our wrapper out of the box.

Specific details of our tests can be seen in https://github.com/Quansight-Labs/numpy_pytorch_interop/blob/main/e2e/tests.md
