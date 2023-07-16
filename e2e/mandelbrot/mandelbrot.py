# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np

from matplotlib import colors
import matplotlib.pyplot as plt


# To run on CUDA, change "cpu" to "cuda" below.
import torch
torch.set_default_device("cpu")


def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    # Adapted from https://www.ibm.com/developerworks/community/blogs/jfp/...
    #              .../entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en
    X = np.linspace(xmin, xmax, xn, dtype=np.float32)
    Y = np.linspace(ymin, ymax, yn, dtype=np.float32)
    C = X + Y[:,None]*1j
    N = np.zeros(C.shape, dtype=int)
    Z = np.zeros(C.shape, np.complex64)
    for n in range(maxiter):
        I = np.abs(Z) < horizon
        N = np.where(I, n, N)  # N[I] = n
        Z = np.where(I, Z**2 + C, Z)  # Z[I] = Z[I]**2 + C[I]
    N = np.where(N == maxiter -1, 0, N)  # N[N == maxiter-1] = 0
    return Z, N


if __name__ == '__main__':
    # Benchmark
    xmin, xmax, xn = -2.25, +0.75, int(3000/3)
    ymin, ymax, yn = -1.25, +1.25, int(2500/3)
    maxiter = 200

    # Visualization
    xmin, xmax, xn = -2.25, +0.75, int(3000/2)
    ymin, ymax, yn = -1.25, +1.25, int(2500/2)
    maxiter = 200
    horizon = 2.0 ** 40
    log_horizon = np.log(np.log(horizon))/np.log(2)
    Z, N = mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon)

    # Normalized recount as explained in:
    # http://linas.org/art-gallery/escape/smooth.html
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

