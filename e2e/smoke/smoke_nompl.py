# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
from smoke_solver import vel_step, dens_step

N = 128
size = N + 2
dt = 0.1
diff = 0.0
visc = 0.0
force = 5.0
source = 100.0

u = np.zeros((size, size), np.float32)  # velocity
u_prev = np.zeros((size, size), np.float32)

v = np.zeros((size, size), np.float32)  # velocity
v_prev = np.zeros((size, size), np.float32)

dens = np.zeros((size, size), np.float32)  # density
dens_prev = np.zeros((size, size), np.float32)

# initialization

u[:, :] = 0.0
v[:, :] = 0.0
u_prev[:, :] = 0.0
v_prev[:, :] = 0.0
dens[:, :] = 0.0
dens_prev[:, :] = 0.0

def disc(shape=(size, size), center=(size/2, size/2), radius=10):
    def distance(x, y):
        return np.sqrt((x-center[0])**2+(y-center[1])**2)

    args = np.indices(shape, dtype=float)
    D = distance(*args)
    return np.where(D <= radius, True, False)

D = disc(radius=32) ^ disc(radius=16)
dens[...] = D*source/10

u[:, :] = force * 0.1 * np.random.uniform(-1, 1, u.shape)
v[:, :] = force * 0.1 * np.random.uniform(-1, 1, u.shape)


# update
def update_single(N, dens, u, v, u_prev, v_prev, visc, diff, dt):
    vel_step(N, u, v, u_prev, v_prev, visc, dt)
    dens_step(N, dens, dens_prev, u, v, diff, dt)


for _ in range(28):
    update_single(N, dens, u, v, u_prev, v_prev, visc, diff, dt)

