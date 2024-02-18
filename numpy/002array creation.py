import numpy as np

a = np.array([2, 3, 4], dtype=np.int64)

print("a.dtype =", a.dtype)

b = np.array([1.2, 3.5, 5.1])
print("b.dtype =", b.dtype)

b = np.array([[1.5, 2, 3], (4, 5, 6)])
print("b =", b)


# 虚数
c = np.array([[1, 2], [3, 4]], dtype=complex)
print("c =", c)


d = np.zeros((3, 4))
print("d =", d)

e = np.ones((2, 3, 4), dtype=np.int16)
print("e =", e)

f = np.empty((2, 3))
print("f =", f)

a1 = np.arange(10, 30, 5)
print("a1 =", a1)

a2 = np.arange(0, 2, 0.3)  # it accepts float arguments
print("a2 =", a2)

from numpy import pi

a3 = np.linspace(0, 2, 9)
print("a3 =", a3)
x = np.linspace(0, 2 * pi, 100)
f = np.sin(x)
print("f =", f)

b1 = np.arange(24).reshape(4, 6)
print("b1 =", b1)
