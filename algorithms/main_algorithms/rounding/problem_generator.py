import cvxpy as cp
import numpy as np
from cvxpy import installed_solvers


def simulated_problem(c, B_array, B_tot, lb):
    """
    Optimize a linear problem:
    max_x min_i c_i x_i
    s.t. B_array * x <= B_tot
    x >= lb
    :param c: numpy array, the objective coefficients
    :param B_array: numpy array, inequality matrix
    :param B_tot: scalar, B_tot
    :return: a numpy represents the optimal solution and a scalar of the optimal value
    """
    K = len(c)
    x = cp.Variable(K + 1)
    A = np.diag(c) * (-1)
    C = np.ones((K, 1))
    D = np.zeros((1, 1))
    a = np.vstack((
        np.hstack((B_array.reshape(1, -1), D)),
        np.hstack((A, C))
    ))
    b = np.zeros(K + 1)
    b[0] = B_tot
   # b = b.reshape((K+1, -1))
    # define constraints
    constraints = [a @ x <= b, x >= np.hstack((lb, [0]))]
    # define objective
    objective = cp.Maximize(x[K])
    prob = cp.Problem(objective, constraints)
    prob.solve(solver='SCIPY')
    return x.value[:K], x.value[K]


if __name__ == '__main__':
    print(installed_solvers())