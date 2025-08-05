import numpy as np


def direct(objective_func, n_iter, n_divisions=20, threshold=1e-4):
    """
    Univariate DIRECT algorithm implementation.

    :param objective_func: The objective function to minimize.
    :param n_iter: Number of iterations to perform.
    :param n_divisions: Number of divisions for each interval.
    :param threshold: Convergence threshold.
    :return: The best point and its function value.
    """
    # Initial interval
    lower, upper = 0.1, 4  # You can set this to your specific interval
    best_point = None
    best_value = np.inf

    for _ in range(n_iter):
        # Divide the interval into n_divisions subintervals
        subintervals = np.linspace(lower, upper, n_divisions + 1)
        values = np.array([objective_func(x) for x in subintervals[:-1]])

        # Find the subinterval with the best (lowest) function value
        best_idx = np.argmin(values)
        best_subinterval = subintervals[best_idx:best_idx + 2]

        # Check for convergence
        if (best_subinterval[1] - best_subinterval[0] < threshold):
            best_point = best_subinterval.mean()
            break

        # Refine the best subinterval
        lower, upper = best_subinterval[0], best_subinterval[1]

    # Evaluate the objective function at the best point
    best_value = objective_func(best_point)

    return best_point, best_value


# Example usage with a simple objective function
def objective_func(x):
    return np.log(x+1) - np.log(x)  # A simple quadratic function



# Run the DIRECT algorithm
best_point, best_value = direct(objective_func, n_iter=10)
print(f"Best point: {best_point}, Best value: {best_value}")
