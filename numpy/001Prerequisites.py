import numpy as np

ndarray_a = np.arange(15).reshape(3, 5)
print("ndarray_a =", ndarray_a)

print("ndarray_a.shape =", ndarray_a.shape)
print("ndarray_a.ndim =", ndarray_a.ndim)
print("ndarray_a.dtype.name =", ndarray_a.dtype.name)
print("ndarray_a.size =", ndarray_a.size)
print("type(ndarray_a) =", type(ndarray_a))


ndarray_b = np.array([6,7,8])
print("ndarray_b =", ndarray_b)
print("type(ndarray_b) =", type(ndarray_b))

