import matplotlib
import matplotlib.pyplot as pl
import numpy as np

def kernel(a, b):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2 , 1) - 2 *np.dot(a, b.T)
    return np.exp(-.5 * sqdist)


matplotlib.interactive(True)
n = 50
x = np.linspace(-5,5, n).reshape(-1,1)
K_ = kernel(x,x)
L = np.linalg.cholesky(K_ + 1e-6 * np.eye(n))
print(L)
a = np.random.normal(size = (n,1))
f = np.dot(L, a)
pl.plot(x,a)
pl.plot(x,f)
print(a)