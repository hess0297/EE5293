

import numpy as np
import matplotlib.pyplot as plt
import time

# Settings
problem_sizes = [100, 500]
condition_numbers = [10, 5000]

max_iter = 100000
tol = 1e-6

# Plot formatting
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7
})

# Create 8 plots
fig, axes = plt.subplots(
    4, 2, figsize=(14, 13),
    constrained_layout=True
)

plot_num = 0

for n in problem_sizes:
    for c in condition_numbers:

        # Q (condition number)
        A = np.random.randn(n, n) # random matrix
        U, _ = np.linalg.qr(A) # orthoganal matrix to A

        eigenvalues = np.linspace(1, c, n) # equaly spaced numbers between 1 and n, 
        Q = U @ np.diag(eigenvalues) @ U.T # create Q
        b = np.random.randn(n) # random linear coefficient

        # exact solution
        # gradient f(x) = Qx* + b = 0
        # Qx* = - b
        def f(x):
            return 0.5 * x @ Q @ x + b @ x

        # Exact solution and minimum
        x_star = np.linalg.solve(Q, -b)
        f_star = f(x_star)

        # Constant step size
        L = c
        s = 1 / L

        # start point
        x = np.zeros(n)

        error = []
        cpu_times = []

        start_time = time.perf_counter()

        # steepest descent
        for k in range(max_iter):

            grad = Q @ x + b

            # Store objective error and CPU time
            error.append(max(f(x) - f_star, 1e-15))
            cpu_times.append(
                1000 * (time.perf_counter() - start_time)
            )

            # Stopping condition
            if np.linalg.norm(grad) < tol:
                break

            # Update x
            x = x - s * grad

        # Objective error vs iteration
        ax1 = axes[plot_num, 0]

        ax1.semilogy(error, linewidth=1.5)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel(r"$f(x^k)-f(x^*)$")
        ax1.set_title(f"Constant Step: n = {n}, c = {c}")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.margins(x=0.02)

        # Objective error vs CPU time
        ax2 = axes[plot_num, 1]

        ax2.semilogy(cpu_times, error, linewidth=1.5)
        ax2.set_xlabel("CPU Time (ms)")
        ax2.set_ylabel(r"$f(x^k)-f(x^*)$")
        ax2.set_title(f"Constant Step: n = {n}, c = {c}")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.margins(x=0.02)

        print(
            f"n = {n}, c = {c}, "
            f"Iterations = {k}, "
            f"CPU Time = {cpu_times[-1]:.2f} ms, "
            f"Error = {error[-1]:.2e}"
        )

        plot_num += 1

# Logarithmic x-axis scaling
for ax in axes.flat:
    ax.set_xscale("symlog", linthresh=1)

# Same x-axis scale for plots with matching units
iteration_max = max(ax.get_xlim()[1] for ax in axes[:, 0])
time_max = max(ax.get_xlim()[1] for ax in axes[:, 1])

for ax in axes[:, 0]:
    ax.set_xlim(0, iteration_max)

for ax in axes[:, 1]:
    ax.set_xlim(0, time_max)

plt.savefig("Constant_Step_Convergence.png", dpi=300)
plt.show()