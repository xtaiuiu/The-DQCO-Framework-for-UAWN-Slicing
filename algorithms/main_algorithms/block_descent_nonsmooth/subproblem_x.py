import math
import numpy as np
import cvxpy as cp
def optimize_x_kkt_with_lb(sc, M=1e8):
    pn = sc.pn
    theta_h = max(pn.theta_min ** 2, (math.atan(pn.radius / (math.sqrt(sc.uav.h)))) ** 2)
    B_tot = pn.b_tot
    C, B, L = [], [], []  # which are the coefficients, B_array and the lower bound of the problem, respectively
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(sc.slices[i].UEs)):
            u = sc.slices[i].UEs[k]
            C.append(s.b_width * np.log(1 + pn.g_0 * u.tilde_g * u.p / (
                    pn.sigma * (theta_h ** 2) * (u.loc_x ** 2 + u.loc_y ** 2 + sc.uav.h) ** (pn.alpha / 2))))
            B.append(s.b_width)
            L.append(u.tilde_r)
    C, B, L = np.array(C), np.array(B), np.array(L)
    v = np.array([C[i]*L[i] for i in range(len(C))])
    idx = np.argsort(-v)
    C_sort, B_sort, L_sort, v_sort = C[idx], B[idx], L[idx], v[idx]

    k_idx, t = -1, 1e-8
    bl, bc, cl = B_sort*L_sort, B_sort/C_sort, C_sort*L_sort
    for i in range(len(C)):
        t = (B_tot - np.sum(bl[:i])) / (np.sum(bc[i:]))
        if t >= cl[i]:
            k_idx = i
            break
    if k_idx < 0:
        raise ValueError(f"Subproblem x is infeasible: sum(l_x) = {np.sum(L)}, B_tot = {B_tot}")
    x_opt = t/C
    for i in range(len(x_opt)):
        if x_opt[i] < L[i]:
            x_opt[i] = L[i]
    return t, x_opt


def optimize_x_cvx_with_lb(sc, M=1e8):
    pn = sc.pn
    theta_h = max(pn.theta_min ** 2, (math.atan(pn.radius / (math.sqrt(sc.uav.h)))) ** 2)
    B_tot = pn.b_tot
    C, B, L = [], [], []  # which are the coefficients, B_array and the lower bound of the problem, respectively
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(sc.slices[i].UEs)):
            u = sc.slices[i].UEs[k]
            C.append(s.b_width * np.log(1 + pn.g_0 * u.tilde_g * u.p / (
                    pn.sigma * (theta_h ** 2) * (u.loc_x ** 2 + u.loc_y ** 2 + sc.uav.h) ** (pn.alpha / 2))))
            B.append(s.b_width)
            L.append(u.tilde_r)
    C, B, L = np.array(C), np.array(B), np.array(L)
    n = len(C)
    x = cp.Variable(shape=n)
    z = cp.Variable(shape=1)
    objective = cp.Maximize(z)
    constraints = [x >= L,
                   cp.sum(cp.multiply(B, x)) - B_tot == 0.0]
    for i in range(n):
        constraints.append(z <= C[i]*x[i])

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return prob.value, x.value
