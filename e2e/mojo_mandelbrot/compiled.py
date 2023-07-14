import numpy as np      # otherwise @compile is ImportError
import torch._dynamo.config as cfg
import torch

cfg.numpy_ndarray_as_tensor = True
device="cpu"


def init():
    # Define the boundaries of the complex plane
    xn = 450
    yn = 375
    xmin = -2.25
    xmax = 0.75
    ymin = -1.25
    ymax = 1.25

    # Create the grid of complex numbers
    x_values = np.linspace(xmin, xmax, xn, dtype='float32')
    y_values = np.linspace(ymin, ymax, yn, dtype='float32')

    rx, iy = np.meshgrid(x_values, y_values, indexing='xy')

    return rx, iy


def simulate(rx, iy, step, max_iter=200):
    x = rx.copy()
    y = iy.copy()
    mask = np.zeros_like(x)
    for _ in range(max_iter):
        mask = step(x, y, rx, iy, mask)
    return mask


def step(x, y, rx, iy, mask):
    x_prev = x
    y_prev = y
    x = x_prev**2 - y_prev**2 + rx
    y = 2*x_prev*y_prev + iy
    inside = np.sqrt(x**2 + y**2) <= 2      # sqrt is from the OP
    mask += inside
    return mask



rx, iy = init()
mask = simulate(rx, iy, step, max_iter=3)

print(np.count_nonzero(mask))


################ benchmark #####################
import time

# ### numpy ###


rx, iy = init()
start_time = time.time()
mask = simulate(rx, iy, step, max_iter=1000)
end_time = time.time()
numpy_time = end_time - start_time
print("\n\nnumpy:    elapsed=", numpy_time)


# ### compile ###

step_c = torch.compile(step)

# ### warm up ###
rx, iy = init()
for _ in range(50):
    simulate(rx, iy, step_c, max_iter=3)


rx, iy = init()
start_time = time.time()
mask = simulate(rx, iy, step_c, max_iter=1000)
end_time = time.time()


compiled_time = end_time - start_time
print("compiled: elapsed=", compiled_time, '  speedup = ', numpy_time / compiled_time)

