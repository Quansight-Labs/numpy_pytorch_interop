A toy NN from scratch using numpy
=================================

Origin: https://www.geeksforgeeks.org/implementation-of-neural-network-from-scratch-using-numpy/

Source: `e2e/nn_from_scratch`.

Results with numpy and torch_np are identical:

  - Use the original numpy random stream in both cases to initialize the NN weights
  - scalar vs 0D array repr differs:
    
    `epochs: 100 ======== acc: 98.64865709132037` with NumPy vs 
    `epochs: 100 ======== acc: array_w(98.6487, dtype=float64)`  with torch_np


