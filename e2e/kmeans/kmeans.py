# k-means step, given data, `X` and centroids
# https://realpython.com/numpy-array-programming/#clustering-algorithms
import numpy as np
import torch
torch.set_default_device("cuda")
import torch._dynamo.config as cfg
cfg.numpy_ndarray_as_tensor = True


# this will be compiled
def get_labels(X, centroids) -> np.ndarray:
    return np.argmin(np.linalg.norm(X - centroids[:, None, :], ord=2, axis=2),
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
npts = int(1e8)
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
torch.cuda.synchronize()
compiled_time = end_time - start_time
print("compiled: elapsed=", compiled_time, '  speedup = ', numpy_time / compiled_time)


