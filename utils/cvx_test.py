import matplotlib.pyplot as plt
import numpy as np




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
    B_tot = r
    I, W = len(v) + 1, B_tot + 1
    M = 1e8
    # base cases
    m = np.zeros((I, W))
    for i in range(1, I):
        m[i][0] = - np.max(v[:i])
    for w in range(W):
        m[0][w] = M
    s = np.zeros((I, W))

    for w in range(1, W):
        for i in range(1, I):
            if wt[i - 1] > w:
                m[i][w] = min(m[i - 1][w], -v[i - 1])
            else:
                m[i][w] = max(min(m[i - 1][w], -v[i - 1]), min(m[i - 1][w - wt[i - 1]], v[i - 1]))
    print(m)
    f_opt = m[I - 1][W - 1]
    print(f_opt)
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
    return y + selected_items