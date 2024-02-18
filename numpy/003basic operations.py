import numpy as np

a = np.array([20, 30, 40, 50])
b = np.arange(4)

print("b =", b)

c = a - b
print("c =", c)

# matrix operation
A = np.array([[1, 1],
              [0, 1]])

B = np.array([[2, 0],
              [3, 4]])

print("A * B =", A * B)

print("A @ B =", A @ B)

print("A.dot(B) =", A.dot(B))

rg = np.random.default_rng(1)  # create instance of default random number generator
brg = rg.random((2, 3))

print("brg =", brg)

b = np.arange(12).reshape(3, 4)

sum_b = b.sum(axis=0)
print("sum_b =", sum_b)


