import math
import numpy as np
from scipy.optimize import bisect
import cvxpy as cp


def optimize_p_kkt_with_lb(sc, M=1e8):
    """
    Optimize the subproblem_p with KKT conditions
    :param sc: The scenario
    :param M: a very large upper bound for the optimal value
    :param eps: precision used in the bisection search
    :return: f_opt, p_opt are scalar and numpy array, which are
    the optimal value and optimal solution of sub_p, respectively.
    """
    theta_h = max(sc.pn.theta_min ** 2, (math.atan(sc.pn.radius / (math.sqrt(sc.uav.h)))) ** 2)
    a, b, l = [], [], []
    pn = sc.pn
    P_max = pn.p_max
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(sc.slices[i].UEs)):
            u = sc.slices[i].UEs[k]
            a.append(u.x * s.b_width)
            b.append(
                pn.g_0 * u.tilde_g / (
                            pn.sigma * (theta_h ** 2) * (u.loc_x ** 2 + u.loc_y ** 2 + sc.uav.h) ** (pn.alpha / 2)))
            l.append((np.exp(s.r_sla/s.b_width) - 1)/b[i])
    a, b, l = np.array(a), np.array(b), np.array(l)
    v = np.array([a[i]*np.log(1 + b[i]*l[i]) for i in range(len(a))])
    idx = np.argsort(-v)  # sort v in descending order
    a_sort, b_sort, l_sort, v_sort = a[idx], b[idx], l[idx], v[idx]

    def f(z, K):
        """
        function for the bisection search
        :param z: input value
        :param K: an index that is used to represent the set of index for which p_i^* > l_i.
        The set is given by {K, K+1, len(a)-1}
        :return res:
        """
        res = 0
        for k in range(K, len(a)):
            res += (np.exp(z/a_sort[k])-1)/b_sort[k]
        return res - P_max + np.sum(l_sort[:K])
    k_idx = -1  # the index that separates the two sets
    for i in range(len(v_sort)):
        if f(v_sort[i], i) <= 0:
            k_idx = i
            break
    func = lambda z: f(z, k_idx)
    if k_idx < 0:
        raise ValueError(f"Subproblem p is infeasible: sum(l) = {np.sum(l)}")
    if k_idx == 0:
        arr_M = [a_sort[j]*np.log(1+b_sort[j]*P_max) for j in range(len(a))]
        f_opt = bisect(func, np.max(arr_M), v_sort[k_idx])
    else:
        print(f" the index is not 0, k_idx = {k_idx}")
        f_opt = bisect(func, v_sort[k_idx-1], v_sort[k_idx])
    #f_opt = bisect(func, v_sort[k_idx], v_sort[k_idx + 1])
    p_opt = np.zeros(len(a))
    for i in range(len(a)):
        p_opt[i] = max(l[i], (np.exp(f_opt/a[i])-1)/b[i])
    return f_opt, p_opt


def optimize_p_cvx_with_lb(sc, M=1e8):
    theta_h = max(sc.pn.theta_min ** 2, (math.atan(sc.pn.radius / (math.sqrt(sc.uav.h)))) ** 2)
    a, b, l = [], [], []
    pn = sc.pn
    P_max = pn.p_max
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(sc.slices[i].UEs)):
            u = sc.slices[i].UEs[k]
            a.append(u.x * s.b_width)
            b.append(
                pn.g_0 * u.tilde_g / (
                        pn.sigma * (theta_h ** 2) * (u.loc_x ** 2 + u.loc_y ** 2 + sc.uav.h) ** (pn.alpha / 2)))
            l.append((np.exp(s.r_sla / s.b_width) - 1) / b[i])
    a, b, l = np.array(a), np.array(b), np.array(l)
    n = len(a)
    p = cp.Variable(shape=n)
    z = cp.Variable(shape=1)
    objective = cp.Maximize(z)
    constraints = [p >= l,
                   cp.sum(p) - P_max == 0.0]
    for i in range(n):
        constraints.append(z <= a[i]*cp.log1p(b[i]*p[i]))

    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status != cp.OPTIMAL:
        raise ValueError(f" the problem is infeasible or unbounded.")

    return prob.value, p.value
