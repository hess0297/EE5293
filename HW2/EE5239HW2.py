import numpy as np
import matplotlib.pyplot as plt

# Generate problem
n = 10

A = np.random.randn(n, n)
Q = A.T @ A + np.eye(n)
b = np.random.randn(n)

# Initial point
x = np.zeros(n)

# Constant step size
L = np.max(np.linalg.eigvalsh(Q))
s = 1 / L

# Steepest descent
max_iter = 1000
tol = 1e-6

grad_norms = []

for k in range(max_iter):

    # Gradient
    grad = Q @ x + b

    # Store gradient norm
    grad_norms.append(np.linalg.norm(grad))

    # Stopping condition
    if np.linalg.norm(grad) < tol:
        break

    # Steepest descent update
    x = x - s * grad

print("Iterations:", k)
print("Solution:", x)
print("Gradient norm:", np.linalg.norm(Q @ x + b))

x_star = np.linalg.solve(Q, -b)

print("Exact solution:", x_star)
print("Error:", np.linalg.norm(x - x_star))