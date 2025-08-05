import itertools
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from algorithms.main_algorithms.relax_quasi_subgradient import gradient, function, gradient_no_height
from scenarios.scenario_creators import create_scenario, load_scenario
from utils.projction import proj_generalized_simplex, rounding_by_proj_QP, rounding_by_opt_condition, \
    proj_generalized_simplex_QP, proj_generalized_simplex_lb
import logging

def qusi_subgradient_no_QoS_dynamic(sc, projector=proj_generalized_simplex_lb, alpha_k=0.5, plot=True):
    logging.basicConfig(level=logging.INFO)
    pn = sc.pn
    n = 0  # the number of UEs in the system
    band_max = -1
    h = max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c * sc.pn.t_s)) ** 2, sc.pn.h_min ** 2)
    sc.set_h(h)
    theta_h = max(sc.pn.theta_min ** 2, (math.atan(sc.pn.radius / (math.sqrt(sc.uav.h)))) ** 2)
    B_array, lb_x, lb_p = [], [], []

    for i in range(len(sc.slices)):
        band_max = max(band_max, sc.slices[i].b_width)
        slice = sc.slices[i]
        for k in range(len(slice.UEs)):
            ue = slice.UEs[k]
            n += 1
            B_array.append(sc.slices[i].b_width)
            lb_x.append(ue.tilde_r)
            nu = pn.g_0 * ue.tilde_g / (
                            pn.sigma * (theta_h ** 2) * (ue.loc_x ** 2 + ue.loc_y ** 2 + sc.uav.h) ** (pn.alpha / 2))
            lb_p.append((np.exp(slice.r_sla/slice.b_width) - 1)/nu)
    B_array = np.array(B_array)
    #  Set initial value for x, p and h
    p = np.ones(n) * pn.p_max / n
    x = np.ones(n) * pn.b_tot / n / band_max

    u = np.concatenate((x, p))
    sc.set_x(u[:n])
    sc.set_p(u[n:2 * n])

    opt_old, opt = -1e8, 0

    # algorithm parameters
    cnt, max_iter = 0, 100000
    v_seq = map(lambda k: 1 / (int(k / 1) + 1), itertools.count(1))
    #v_seq = map(lambda k: 1, itertools.count(1))
    a_seq = map(lambda k: 0.5, itertools.count(1))
    gamma, delta, beta, theta = 2, 100, 0.9, 2
    f_t, delta_k = function(sc), delta
    #初始化绘图
    if plot:
        plt.ion()  # 交互模式开启，允许动态绘图
        plt.figure()  # 创建一个新图形
        plt.plot([], [], 'b-', label='Function Value')
        plt.title('Objective Function Value per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.legend()
        plt.xlim(0, int(max_iter / 100))
        plt.ylim(-650, 0)
    function_values = []

    for k in range(max_iter):
        f_t = min(f_t, function(sc))
        f_k = f_t - delta_k
        g = gradient_no_height(sc, 2 * n)
        #print(f"g = {g}")
        v = (function(sc) - f_k) / (max(gamma, np.linalg.norm(g) ** 2))
        n_delta_change = 1000
        if not k % n_delta_change:
            delta = max(0.001, delta * 0.8)
        u = u - alpha_k * v * g  # quasi-subgradient descent
        # project u onto the feasible set
        u[:n] = projector(B_array, u[:n], pn.b_tot, lb_x)
        u[n:2 * n] = projector(np.ones_like(B_array), u[n:2 * n], pn.p_max, lb_p)
        sc.set_vars_no_h(u)
        if function(sc) > f_k:
            # print(f"f(k+1) > f(k)")
            delta_k = max(beta * delta_k, delta)
        else:
            print(f"f(k+1) <= f(k)")
            delta_k = theta * delta_k
        n_plots = 1000
        # if np.linalg.norm(g)**2 >= 10000:
        #     print("start debug")
        if not k % n_plots:
            logging.info(
                f"k = {k}, f = {function(sc):.8f}, f_k = {f_k:.8f} v = {v}, delta_k = {delta_k}, norm_g^2 = {np.linalg.norm(g) ** 2}, nonzeros_g = {np.sum(g != 0)}")
            #function_values.append(function(sc))
            function_values.append(f_t)
            #function_values.append(z[-1])
            if plot:
                plt.plot(range(int(k / n_plots) + 1), function_values, 'b-', label='Function Value')
                plt.draw()
                plt.pause(0.05)
        if abs((function(sc) - f_k) / f_k) < 0.001 and k > 100:
            df = pd.DataFrame(function_values)
            df.to_excel('convergence_function_values.xlsx')
            break
    if plot:
        plt.ioff()
        plt.show()
    logging.info(f"B_array = {B_array}, u = {u}, b_tot = {pn.b_tot}")
    # x_int = rounding_by_proj_QP(B_array, u[:n], pn.b_tot)
    # p_int = rounding_by_proj_QP(np.ones(n), u[n:2*n], pn.p_max)
    return -function(sc), u, function_values


def rounded_solution(sc, u, method=rounding_by_opt_condition):
    """
    return the rounded solution
    :param sc: the Scenario
    :param u: the continuous variables
    :param method: the rounding method
    :return: u_rounded, f_rounded
    """
    pn = sc.pn
    n = 0  # the number of UEs in the system
    band_max = -1
    B_array = []
    for i in range(len(sc.slices)):
        band_max = max(band_max, sc.slices[i].b_width)
        for k in range(len(sc.slices[i].UEs)):
            n += 1
            B_array.append(sc.slices[i].b_width)
    B_array = np.array(B_array)
    c = []
    for s in sc.slices:
        for ue in s.UEs:
            A, B, C, beta = pn.g_0 * ue.tilde_g, pn.sigma * sc.uav.theta ** 2, sc.uav.h + ue.loc_x ** 2 + ue.loc_y ** 2, pn.alpha / 2
            c.append(s.b_width * np.log(1 + A * ue.p / (B * C ** beta)))
    x_int = method(B_array, np.array(c), u[:n], pn.b_tot)
    u_rounded = np.concatenate((x_int, u[n:2 * n]))
    sc.set_vars_no_h(u_rounded)
    f_rounded = function(sc)
    return u_rounded, f_rounded


def qusi_subgradient_no_QoS_fixed_stepsize(sc, plot=True):
    pn = sc.pn
    n = 0  # the number of UEs in the system
    band_max = -1
    B_array = []
    for i in range(len(sc.slices)):
        band_max = max(band_max, sc.slices[i].b_width)
        for k in range(len(sc.slices[i].UEs)):
            n += 1
            B_array.append(sc.slices[i].b_width)
    B_array = np.array(B_array)
    #  Set initial value for x, p and h
    p = np.ones(n) * pn.p_max / n
    x = np.ones(n) * pn.b_tot / n / band_max
    h = max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c * sc.pn.t_s)) ** 2, sc.pn.h_min ** 2)
    u = np.concatenate((x, p))
    sc.set_x(x)
    sc.set_p(p)
    sc.set_h(h)
    opt_incumbent, opt = 1e8, function(sc)
    var_incumbent = np.zeros(len(u))
    cnt, max_iter, eps = 0, 100000, 1e-6
    v = 0.01
    while cnt < max_iter and abs(opt_incumbent - opt) > eps:

        g = gradient_no_height(sc, 2 * n)
        u = u - v * g  # quasi-subgradient descent
        # project u onto the feasible set
        u[:n] = proj_generalized_simplex(B_array, u[:n], pn.b_tot)
        u[n:2 * n] = proj_generalized_simplex(np.ones_like(B_array), u[n:2 * n], pn.p_max)
        sc.set_vars_no_h(u)
        opt = function(sc)
        if opt < opt_incumbent:
            opt_incumbent = opt
            var_incumbent = u
        n_plots = 100
        if not cnt % n_plots:
            print(f"cnt = {cnt}, opt_incumbent = {opt_incumbent:.8f}, opt = {opt: .8f}")
            #function_values.append(function(sc))
        cnt += 1
    print(f"B_array = {B_array}, u = {u}, b_tot = {pn.b_tot}")
    # x_int = rounding_by_proj_QP(B_array, u[:n], pn.b_tot)
    # p_int = rounding_by_proj_QP(np.ones(n), u[n:2*n], pn.p_max)
    c = []
    for s in sc.slices:
        for ue in s.UEs:
            A, B, C, beta = pn.g_0 * ue.tilde_g, pn.sigma * sc.uav.theta ** 2, sc.uav.h + ue.loc_x ** 2 + ue.loc_y ** 2, pn.alpha / 2
            c.append(s.b_width * np.log(1 + A * ue.p / (B * C ** beta)))
    x_int = rounding_by_opt_condition(B_array, np.array(c), var_incumbent[:n], pn.b_tot)
    u_opt = np.concatenate((x_int, var_incumbent[n:2 * n]))
    sc.set_vars_no_h(u_opt)
    f_opt = function(sc)
    return u_opt, f_opt


if __name__ == '__main__':
    from pyinstrument import Profiler
    import time

    profiler = Profiler()
    profiler.start()
    np.random.seed(0)
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    sc = create_scenario(10, 10)
    sc.pn.b_tot = 1000
    sc.pn.p_max = 1000
    sc = load_scenario('scenario_2_slices.pickle')
    h = max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c * sc.pn.t_s)) ** 2, sc.pn.h_min ** 2)
    sc.set_h(h)
    sc.set_theta(h)
    #opt_var, opt_fval = qusi_subgradient_no_QoS_dynamic(sc, plot=True)
    start_time = time.perf_counter()  # 记录开始时间

    opt_var, opt_fval, _ = qusi_subgradient_no_QoS_dynamic(sc, projector=proj_generalized_simplex_QP, plot=True)
    end_time = time.perf_counter()  # 记录结束时间
    runtime = end_time - start_time  # 计算运行时间
    print(f"Runtime: {runtime} seconds")
    print(f"opt_var = {opt_var}, opt_fval = {-opt_fval}, n_UEs = {len(opt_var)}")
    profiler.stop()
    profiler.print()
