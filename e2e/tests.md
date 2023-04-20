A toy NN from scratch using numpy
=================================

Origin: https://www.geeksforgeeks.org/implementation-of-neural-network-from-scratch-using-numpy/

Source: `e2e/nn_from_scratch`.


Tweaks
------

  - Use the original numpy random stream in both cases to initialize the NN weights


Results
-------

Results with numpy and torch_np are identical modulo different scalar vs 0D array reprs:
    
    `epochs: 100 ======== acc: 98.64865709132037` with NumPy vs 
    `epochs: 100 ======== acc: array_w(98.6487, dtype=float64)`  with torch_np



Build maze
==========

Origin: N. Rougier, From python to numpy, 
https://github.com/rougier/from-python-to-numpy/blob/master/code/maze_numpy.py

Source: `e2e/maze`


Tweaks
------

Seed the numpy random generator, use the same random stream.

For plotting with matplotlib, convert torch_numpy arrays to numpy via
`Z = Z.tensor.numpy()`.



Diffusion/advection
-------------------

Origin: N. Rougier, From python to numpy,
https://github.com/rougier/from-python-to-numpy/blob/master/code/{smoke_solver,smoke_1,smoke_2}.py

Source: `e2e/smoke`

Tweaks
------

 - fix a bug in bool array minus bool array (fails on numpy 1.24)
 - inline np.fromfunction into a call to np.indices


Mandelbrot fractal
------------------

Origin: N. Rougier, From python to numpy, 
https://github.com/rougier/from-python-to-numpy/blob/master/code/mandelbrot.py
https://github.com/rougier/from-python-to-numpy/blob/master/code/mandelbrot_numpy_1.py

source: `e2e/mandelbrot.py`

Tweaks
------

  - use `mandelbrot_numpy_1.py` version (slightly slower, but no mgrid)
  - complex abs in float32 runs into https://github.com/pytorch/pytorch/pull/99550

