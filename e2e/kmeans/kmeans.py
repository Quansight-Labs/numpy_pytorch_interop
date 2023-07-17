# k-means step, given data, `X` and centroids
# https://realpython.com/numpy-array-programming/#clustering-algorithms
import numpy as np
import torch
torch.set_default_device("cpu")
import torch._dynamo.config as cfg
cfg.numpy_ndarray_as_tensor = True


# np.linalg.norm replacement (2-norm only), https://github.com/pytorch/pytorch/issues/105269
def norm(a, axis):
    s = (a.conj() * a).real
    return np.sqrt(s.sum(axis=axis))


#@torch.compile
def get_labels(X, centroids) -> np.ndarray:
    return np.argmin(norm(X - centroids[:, None], axis=2),
                     axis=0)


def init(npts):
    np.random.seed(12345)
    X = np.repeat([[5, 5], [10, 10]], [npts, npts], axis=0)
    X = X + np.random.randn(*X.shape)  # 2 distinct "blobs"
    centroids = np.array([[5, 5], [10, 10]])
    return X, centroids


################ benchmark #####################
import time

# ### numpy ###
npts = int(2e7)
X, centroids = init(npts)

start_time = time.time()
labels = get_labels(X, centroids)
end_time = time.time()
numpy_time = end_time - start_time
print("\n\nnumpy:    elapsed=", numpy_time)


# ### compile ###
get_labels_c = torch.compile(get_labels)

# ### warm up ###
for _ in range(5):
    get_labels_c(X, centroids)


# ### measure ###
start_time = time.time()
labels = get_labels_c(X, centroids)
end_time = time.time()
compiled_time = end_time - start_time
print("compiled: elapsed=", compiled_time, '  speedup = ', numpy_time / compiled_time)

