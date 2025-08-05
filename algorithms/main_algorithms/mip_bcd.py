import warnings

import cvxpy as cp
import numpy as np
import math

import scipy.optimize
from scipy import optimize
from scipy.optimize import minimize, root_scalar

from algorithms.main_algorithms.func_val_gradient import func_val_x, grad_h, func_val_h, func_val_h_wrapper, func_val_p, \
    grad_p
from algorithms.main_algorithms.projection import proj_b
from scenarios.scenario_creators import create_scenario, save_scenario, load_scenario
import operator
from functools import reduce


# from utils.test_func_grad import test_func_h_plot


def mip_bcd(sc):
    eps = 1e-6
    pn = sc.pn
    n = 0  # the number of UEs in the system
    band_max = -1
    for i in range(len(sc.slices)):
        band_max = max(band_max, sc.slices[i].b_width)
        for k in range(len(sc.slices[i].UEs)):
            n += 1
    #  Set initial value for p and h
    p = np.ones(n) * pn.p_max / n
    x = np.ones(n) * pn.b_tot / n / band_max
    h = max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c * sc.pn.t_s)) ** 2, sc.pn.h_min ** 2)
    sc.set_x(x)
    sc.set_p(p)
    sc.set_h(h)
    opt_old, opt = -1e8, 0

    cnt, max_iter = 0, 100
    while cnt < max_iter and np.abs(opt - opt_old) > eps:
        #  Step 1: optimize x
        f_x, x = optimize_x_continuous_kkt(sc)
        print(f" optimize x in BCD loop cnt = {cnt: 04}, f_x = {f_x: .12f}")
        sc.set_x(x)
        #  Step 2: optimize p
        f_p, p = optimize_p_kkt(sc)
        print(f" optimize p in BCD loop cnt = {cnt: 04}, f_p = {-f_p: .12f}, p = {p}")
        sc.set_p(p)
        #  Step 3: optimize h
        f_h, h = minimize_h_golden(sc)
        print(f" optimize h in BCD loop cnt = {cnt: 04}, f_h = {-f_h: .12f}")
        sc.set_h(h)
        sc.set_theta(h)

        # print(f"*********************** BCD loop cnt: {cnt: 04} *********************** ")

        opt_old = opt
        opt = -f_h
        cnt += 1

    #  Return the outcomes
    vars = sc.scenario_variables()
    print(f"x = {vars.x}")
    print(f"p = {vars.p}")
    print(f"h = {vars.h}")
    return opt, vars


def optimize_x_continuous(scenario):
    # define variables
    K = 0
    for s in scenario.slices:
        for k in range(len(s.UEs)):
            K += 1
    x = cp.Variable(K + 1)
    A = np.eye(K)
    B = np.zeros((1, K))
    C = np.ones((K, 1))
    D = np.zeros((1, 1))
    theta_h = max(scenario.pn.theta_min ** 2, (math.atan(scenario.pn.radius / (math.sqrt(scenario.uav.h)))) ** 2)
    idx = 0
    for i in range(len(scenario.slices)):
        s = scenario.slices[i]
        for k in range(len(scenario.slices[i].UEs)):
            u = scenario.slices[i].UEs[k]
            A[idx][idx] = -(s.b_width * np.log(1 + scenario.pn.g_0 * u.tilde_g * u.p / (
                    scenario.pn.sigma * theta_h * (u.loc_x ** 2 + u.loc_y ** 2 + scenario.uav.h) ** (
                    scenario.pn.alpha / 2))) / u.tilde_r)
            B[0][idx] = s.b_width
            idx += 1
    a = np.vstack((
        np.hstack((A, C)),
        np.hstack((B, D))
    ))
    b = np.zeros(K + 1)
    b[K] = scenario.pn.b_tot
    print(a)

    # define constraints
    constraints = [a @ x <= b, x >= 0]
    # define objective
    objective = cp.Maximize(x[K])
    prob = cp.Problem(objective, constraints)

    print("Optimal value", prob.solve())
    print("Optimal var")
    print(x.value)  # A numpy ndarray.
    print(-a[:K, :K] @ x.value[:K])
    idx = 0
    for i in range(len(scenario.slices)):
        s = scenario.slices[i]
        for k in range(len(scenario.slices[i].UEs)):
            u = s.UEs[k]
            u.x = x.value[idx]
            idx += 1
    return x.value


def optimize_x_continuous_kkt(sc):
    pn = sc.pn
    theta_h = max(pn.theta_min ** 2, (math.atan(pn.radius / (math.sqrt(sc.uav.h)))) ** 2)
    B_tot = pn.b_tot
    A, B = [], []
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(sc.slices[i].UEs)):
            u = sc.slices[i].UEs[k]
            A.append(s.b_width * np.log(1 + pn.g_0 * u.tilde_g * u.p / (
                    pn.sigma * (theta_h**2) * (u.loc_x ** 2 + u.loc_y ** 2 + sc.uav.h) ** (pn.alpha / 2))))
            B.append(s.b_width)
    den = [B[i] / A[i] for i in range(len(A))]
    x_opt = [B_tot / A[i] / np.sum(den) for i in range(len(A))]
    f_opt = B_tot / np.sum(den)
    return f_opt, x_opt


def optimize_x_discrete(sc):
    pass


def optimize_p(sc):
    n, max_iter = 0, 1000
    eps = 1e-8
    delta, beta, theta = 0.01, 0.5, 1
    delta_k = delta
    min_f = 1e8
    func_val = 1e8
    len_p = len(np.concatenate(sc.scenario_variables().p))
    p = np.ones(len_p) * (sc.pn.p_max / len_p) / 2
    p_old = p
    func_val_new, _, _ = func_val_p(sc, p)
    func_val_new = -func_val_new
    print(f"Iteration started:  h_init = {p}, f_init = {func_val_new: .4f}")
    while (n < max_iter) and (math.fabs(func_val_new - func_val) > eps):
        grad, _, _ = grad_p(sc, p)
        func_val, _, _ = func_val_p(sc, p)
        grad, func_val = -grad, -func_val
        min_f = min(func_val, min_f)
        f_k = min_f - delta_k
        alpha_k = (func_val - f_k) / ((np.linalg.norm(grad)) ** 2)
        # subgradient descent
        p_old = p
        # p = p - alpha_k*grad
        p = p - grad
        # projection
        p = proj_b(p / sc.pn.p_max) * sc.pn.p_max

        # update delta_k
        func_val_new, _, _ = func_val_p(sc, p)
        func_val_new = -func_val_new
        print(
            f"n = {n: d}, p_old = {p_old}, p = {p}, grad = {grad}, min_f = {min_f: .4f}, f_val = {func_val_new: .4f}, alpha_k = {alpha_k: .4f}")
        if func_val_new <= f_k:
            delta_k *= theta
        else:
            delta_k = max(beta * delta_k, delta)
        n += 1
    fval, _, _ = func_val_p(sc, p)
    return fval, p


def optimize_p_kkt(sc):
    theta_h = max(sc.pn.theta_min ** 2, (math.atan(sc.pn.radius / (math.sqrt(sc.uav.h)))) ** 2)
    C, D = [], []
    pn = sc.pn
    P_max = pn.p_max
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(sc.slices[i].UEs)):
            u = sc.slices[i].UEs[k]
            C.append(u.x * s.b_width)
            D.append(
                pn.g_0 * u.tilde_g / (pn.sigma * (theta_h**2) * (u.loc_x ** 2 + u.loc_y ** 2 + sc.uav.h) ** (pn.alpha / 2)))

    def f_q(q):
        res = .0
        for i in range(len(C)):
            res += (np.exp(-q / C[i]) - 1) / D[i]
        return res - P_max

    def f_q_prime(q):
        return np.sum([-np.exp(-q / C[i]) / C[i] / D[i] for i in range(len(C))])

    # compute the zeros
    init_guess = -1.0
    # q_star = root_scalar(f_q, method='newton', x0=init_guess, fprime=f_q_prime).root
    q_star = root_scalar(f_q, method='bisect', bracket=[-1e8, 0]).root
    p_star = [(np.exp(-q_star / C[i]) - 1) / D[i] for i in range(len(C))]
    return -q_star, p_star


def optimize_h(sc):
    # lower bound of h
    lb = max(sc.pn.h_min ** 2, (max(0, np.sqrt(sc.uav.h_bar) - sc.pn.t_s)) ** 2)
    # upper bound of h
    ub = min(sc.pn.h_max ** 2, (np.sqrt(sc.uav.h_bar) + sc.pn.t_s) ** 2)
    if ub < lb:
        raise ValueError("lb is larger than ub. lb = {:.4f}, ub = {:.4f}".format(lb, ub))
    # set initial value of h
    h = (ub + lb) / 2
    h_old = -1e8
    n, max_iter = 0, 1000
    tolerance = 1e-10
    delta, beta, theta = 0.001, 0.5, 1
    delta_k = delta
    min_f = 1e8
    func_val = 1e8
    func_val_new, _, _ = func_val_h(sc, h)
    func_val_new = -func_val_new
    print(f"Iteration started: lb = {lb: .4f}, ub = {ub: .4f}, h_init = {h: .4f}, f_init = {func_val_new: .4f}")
    while (n < max_iter) and (math.fabs(func_val_new - func_val) > tolerance):
        grad, _, _ = grad_h(sc, h)
        func_val, _, _ = func_val_h(sc, h)
        grad, func_val = -grad, -func_val
        min_f = min(func_val, min_f)
        f_k = min_f - delta_k
        alpha_k = (func_val - f_k) / (grad ** 2)

        print(
            f"n = {n: d}, h_old = {h_old: .4f}, h = {h: .4f}, grad = {grad: .8f}, min_f = {min_f: .8f}, f_val = {func_val: .4f}, alpha_k = {alpha_k: .4f}")
        # gradient ascent
        h_old = h
        h = h - alpha_k * grad

        # projection
        if h > ub:
            h = ub
        if h < lb:
            h = lb
        # update delta_k
        func_val_new, _, _ = func_val_h(sc, h)
        func_val_new = -func_val_new
        if func_val_new <= f_k:
            delta_k *= theta
        else:
            delta_k = max(beta * delta_k, delta)
        n += 1
    return h


def minimize_h_golden(sc):
    # lower bound of h
    lb = max(sc.pn.h_min ** 2, (max(0, np.sqrt(sc.uav.h_bar) - sc.pn.t_s)) ** 2)
    # upper bound of h
    ub = min(sc.pn.h_max ** 2, (np.sqrt(sc.uav.h_bar) + sc.pn.t_s) ** 2)
    if ub < lb:
        raise ValueError("lb is larger than ub. lb = {:.4f}, ub = {:.4f}".format(lb, ub))
    # set initial value of h
    result = scipy.optimize.minimize_scalar(func_val_h_wrapper, bounds=(lb, ub), args=(sc,), method='bounded')
    # print("Optimization result: x =", result.x, "with f(x) =", result.fun)
    return result.fun, result.x


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    # scenario = create_scenario(100, 1)
    scenario = load_scenario('scenario_2_slices.pickle')
    # p_star, f_star = optimize_p(scenario)
    # print(f"p_star = {p_star}, f_star = {f_star}")
    # f_kkt, p_kkt = optimize_p_kkt(scenario)
    # print(f"f_kkt = {f_kkt}, p_kkt = {p_kkt}")
    # scenario = create_scenario(np.random.randint(2, 10), np.random.randint(1, 100))

    # x = scenario.scenario_variables().x
    # x = reduce(operator.concat, x)
    # func, k, i = func_val_x(scenario, x)
    # print(f"func = {func: .4f}, k = {k}, i = {i}")
    # x_value = optimize_x_continuous(scenario)
    # f_kkt, x_kkt = optimize_x_continuous_kkt(scenario)
    # print(f"f_kkt = {f_kkt}, x_kkt = {x_kkt}")
    # func, k, i = func_val_x(scenario, x_value)
    # print(f"func = {func: .4f}, k = {k}, i = {i}")
    # h_star = optimize_h(scenario)
    # f_star, _, _ = func_val_h(scenario, h_star)
    # print(f"h_star = {h_star: .8f}, f_star = {f_star: .8f}")
    # h_min, f_min = minimize_h_golden(scenario)
    # print(f"h_min = {h_min: .8f}, f_min = {f_min: .8f}")
    # test_func_h_plot(scenario)
    # save_scenario(scenario, 'problem.pickle')

    val, var = mip_bcd(scenario)
    rates, obj = scenario.get_UE_rate()
    print(rates)
    print(obj)
