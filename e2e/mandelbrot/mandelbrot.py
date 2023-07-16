# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import math
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

import torch
torch.set_default_device("cpu")
import torch._dynamo.config as cfg
cfg.numpy_ndarray_as_tensor = True


def abs2(a):
    r"""abs(a) replacement."""
    return a[..., 0]**2 + a[..., 1]**2


def sq2(a):
    """a**2 replacement."""
    z = np.empty_like(a)
    z[..., 0] = a[..., 0]**2 - a[..., 1]**2
    z[..., 1] = 2 * a[..., 0] * a[..., 1]
    return z


@torch.compile
def step(n, c, Z, N, horizon):
    I = abs2(Z) < horizon**2
    N = np.where(I, n, N)                         # N[I] = n
    Z = np.where(I[..., None], sq2(Z) + c, Z)     # Z[I] = Z[I]**2 + C[I]
    return Z, N


def mandelbrot_c(xmin, xmax, ymin, ymax, xn, yn, horizon=2**10, maxiter=5):
    x = np.linspace(xmin, xmax, xn, dtype='float32')
    y = np.linspace(ymin, ymax, yn, dtype='float32')
    c = np.stack(np.broadcast_arrays(x[None, :], y[:, None]), axis=-1)

    N = np.zeros(c.shape[:-1], dtype='int')
    Z = np.zeros_like(c, dtype='float32')

    for n in range(maxiter):
        Z, N = step(n, c, Z, N, horizon)

    N = np.where(N == maxiter-1, 0, N)     # N[N == maxiter-1] = 0
    return Z, N


if __name__ == '__main__':
    # start up
    xmax, xmin, xn = -2.25, 0.75, 3000 // 2
    ymax, ymin, yn = -1.25, 1.25, 2500 // 2

    maxiter = 200
    horizon = 2**40
    log_horizon = math.log(horizon, 2)

    # simulate
    Z, N = mandelbrot_c(xmin, xmax, ymin, ymax, xn, yn, maxiter=maxiter)

    # Visualization
    Z = Z[..., 0] + 1j*Z[..., 1]
    M = np.nan_to_num(N + 1 - np.log(np.log(abs(Z)))/np.log(2) + log_horizon)

    dpi = 72
    width = 10
    height = 10*yn/xn

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False, aspect=1)

    light = colors.LightSource(azdeg=315, altdeg=10)

    plt.imshow(light.shade(M, cmap=plt.cm.hot, vert_exag=1.5,
                           norm = colors.PowerNorm(0.3), blend_mode='hsv'),
               extent=[xmin, xmax, ymin, ymax], interpolation="bicubic")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("mandelbrot.png")
    plt.show()

