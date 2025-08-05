import copy

import numpy as np
import cvxpy as cp
from scipy.optimize import Bounds, milp, LinearConstraint

def rounding_by_mip(c, B_array, B_tot, lb):
    """
    Optimize a linear integer problem:
    max_x min_i c_i x_i
    s.t. B_array * x <= B_tot
    x >= lb
    :param c: numpy array, the objective coefficients
    :param B_array: numpy array, inequality matrix
    :param B_tot: scalar, B_tot
    :param lb: numpy array, must be integer
    :return: a numpy represents the optimal solution and a scalar of the optimal value
    """
    K = len(c)
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
    b_u = b
    b_l = np.full_like(b, -np.inf, dtype=float)
    constraints = LinearConstraint(a, b_l, b_u)
    # define objective
    LB = np.hstack((lb, [-np.inf]))
    bounds = Bounds(lb=LB)
    integrality = np.ones_like(LB)
    integrality[-1] = 0
    obj_c = np.zeros_like(LB)
    obj_c[-1] = -1
    res = milp(c=obj_c, constraints=constraints, integrality=integrality, bounds=bounds)
    if res.status == 1 or res.status == 0:
        return res.x, -res.fun
    else:
        print(f"lb = {LB[:-1]}, lb@B_array = {LB[:-1]@B_array}, B_tot = {B_tot}")
        raise ValueError(f"Infeasible or Unbounded Problem: status = {res.status}")


def rounding_by_knapsack(a, c, y, r):
    """
    Round y to integers to maximize min_i c_i*y_i such that a^T y <= r.
    :param a: B_i
    :param c: coefficient of objective
    :param y: the variables to be rounded
    :param r: B_tot
    :return: the rounded y
    """
    v = c
    wt = a
    B_tot = int(r - a @ np.floor(y))
    I, W = len(v) + 1, B_tot + 1
    if I <= 0 or W <= 0:
        print(f'I = {I}, W = {W}')
        print(f'a @ np.floor(y) = {a @ np.floor(y)}, r = {r}')
    M = 1e8
    # base cases
    m = np.zeros((I, W))
    for i in range(1, I):
        m[i][0] = - np.max(v[:i])
    for w in range(W):
        m[0][w] = M

    for w in range(1, W):
        for i in range(1, I):
            if wt[i - 1] > w:
                m[i][w] = min(m[i - 1][w], -v[i - 1])
            else:
                m[i][w] = max(min(m[i - 1][w], -v[i - 1]), min(m[i - 1][w - wt[i - 1]], v[i - 1]))
    #print(m)
    f_opt = m[I - 1][W - 1]
    #print(f_opt)
    selected_items = np.zeros(len(v))
    q = (f_opt / v + 1) / 2
    for i in range(len(v)):
        if q[i] > 0:
            selected_items[i] = 1
    B_tot -= selected_items @ wt
    for i in range(len(v)):
        if q[i] == 0 and B_tot - wt[i] >= 0:
            selected_items[i] = 1
            B_tot -= wt[i]
    return np.floor(y) + selected_items


def rounding_by_opt_condition(a, c, y, r):
    """
    Round y to integers to maximize c^T y such that a^T y <= r.
    :param a: B_i
    :param c: coefficient of objective
    :param y: the variables to be rounded
    :param r: B_tot
    :return: the rounded y
    """
    y_int = np.floor(y)

    u = c*np.floor(y)
    r -= a@y_int
    sorted_idx = np.argsort(u)
    for i in sorted_idx:
        if r - a[i] >= 0 and abs(np.round(y[i]) - y[i]) > 1e-6:
            y_int[i] += 1
            r -= a[i]
    return y_int


def rounding_by_worst_condition(a, c, y, r):
    """
    Round y to integers to maximize c^T y such that a^T y <= r.
    :param a: B_i
    :param c: coefficient of objective
    :param y: the variables to be rounded
    :param r: B_tot
    :return:
    """
    y_int = np.floor(y)

    u = c*np.ceil(y)
    r -= a@y_int
    sorted_idx = np.argsort(-u)
    for i in sorted_idx:
        if r - a[i] >= 0:
            y_int[i] += 1
            r -= a[i]
    return y_int


def rounding_by_random_order(a, c, y, r):
    """
    Round y to integers to maximize c^T y such that a^T y <= r.
    :param a: B_i
    :param c: coefficient of objective
    :param y: the variables to be rounded
    :param r: B_tot
    :return:
    """
    y_int = np.floor(y)
    order = np.arange(len(y))
    r -= a@y_int
    np.random.shuffle(order)
    for i in order:
        if r - a[i] > 0 and abs(np.round(y[i]) - y[i]) > 1e-6:
            y[i] = y_int[i] + 1
            r -= a[i]
        else:
            y[i] = y_int[i]
    return y


def rounding_nearest(a, c, y, r):
    """
    Round y to its nearest integers such that a^T y <= r.
    :param a: B_i
    :param c: coefficient of objective
    :param y: the variables to be rounded
    :param r: B_tot
    :return:
    """
    y_int = np.floor(y)
    r -= a @ y_int

    for i in range(len(y)):
        if r - a[i] > 0 and abs(np.round(y[i]) - y[i]) > 1e-6:
            y[i] = y_int[i] + 1
            r -= a[i]
        else:
            y[i] = y_int[i]
    return y


if __name__ == '__main__':
    x_con = np.array([5, 10.9176737, 5.29377029, 9.5979652, 5.18494595])
    c = np.array([15.47370115, 6.6546328, 13.72426561, 7.56963668, 14.01231761])
    B_array = np.array([0.5, 0.1, 0.5, 0.5, 2.5])
    B_tot = 24
    lb = np.array([5, 2, 2, 0, 5])
    x_opt = rounding_by_opt_condition(B_array, c, x_con, B_tot)
    x_near = rounding_nearest(B_array, c, copy.deepcopy(x_con), B_tot)
    x_random = rounding_by_random_order(B_array, c, copy.deepcopy(x_con), B_tot)
    x_worst = rounding_by_worst_condition(B_array, c, copy.deepcopy(x_con), B_tot)
    print(min(x_opt * c), min(x_near * c), min(x_random * c), min(x_worst * c))

