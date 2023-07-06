import numpy as np      # otherwise @compile is ImportError
import torch._dynamo.config as cfg
import torch
import torch_np

cfg.numpy_ndarray_as_tensor = True
device="cpu"

@torch.compile
def mandelbrot_numpy(max_iter=200):
    # Define the boundaries of the complex plane
    xn = 450
    yn = 375
    xmin = -2.25
    xmax = 0.75
    ymin = -1.25
    ymax = 1.25

    # Create the grid of complex numbers
    x_values = np.linspace(xmin, xmax, xn, dtype='float64')
    y_values = np.linspace(ymin, ymax, yn, dtype='float64')


    rx, iy = np.meshgrid(x_values, y_values, indexing='xy')

    x = rx.copy()
    y = iy.copy()
    mask = np.zeros_like(x)
    for _ in range(max_iter):
        x_prev = x
        y_prev = y
        x = x_prev**2 - y_prev**2 + rx
        y = 2*x_prev*y_prev + iy
        inside = np.sqrt(x**2 + y**2) <= 2      # sqrt is from the OP
        mask += inside
    return mask

print(mandelbrot_numpy(3))
