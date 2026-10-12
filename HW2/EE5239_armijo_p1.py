import numpy as np

f = lambda x: 3*x[0]**2 + x[1]**4

x = np.array([1., -2.])
d = np.array([-1., 2/3])
g = np.array([6., -32.])

s, beta, sigma = 1, 0.5, 0.1

for m in range(100):
    alpha = s * beta**m
    if f(x + alpha*d) <= f(x) + sigma*alpha*(g @ d):
        break

print("m =", m)
print("alpha =", alpha)
print("x1 =", x + alpha*d)
print("f(x1) =", f(x + alpha*d))