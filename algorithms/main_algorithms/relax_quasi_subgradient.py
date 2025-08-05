import concurrent.futures
import itertools

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import bisect

from algorithms.main_algorithms.func_val_gradient import func_val
from scenarios.scenario_creators import load_scenario, create_scenario


def quasi_subgradient(scenario):
    pass


def affine_proj(x0, a, b):
    """
    project x0 to the affine half-space defined by ax <= b
    :param x0: numpy array. the point which will be projected
    :param a: numpy array
    :param b: scalar
    :return: the projection
    """
    if np.dot(a, x0) <= b:
        return x0
    else:
        return x0 + (b - np.dot(a, x0)) / (np.linalg.norm(a) ** 2) * a


def nonlinear_proj(t0, p0, beta):
    t = cp.Variable()
    p = cp.Variable()

    objective = cp.Minimize(((t - t0) ** 2 + (p - p0) ** 2) / 2)

    # 线性约束
    constraints = [t ** beta - p <= 0]

    # 创建问题并求解
    prob = cp.Problem(objective, constraints)
    # print("prob1 is DCP:", prob.is_dcp())
    result = prob.solve()

    # 打印结果
    print("Status:", prob.status)
    print("The optimal value is", result)
    print("A solution t is", t.value)
    print("A solution p is", p.value)


def nonlinear_proj_kkt(t0, p0, beta):
    if t0 ** beta <= p0:
        print(f"kkt, t = {t0}, p= {p0}")
        return t0, p0
    else:
        def func(t):
            return beta * t ** (2 * beta - 1) - beta * p0 * t ** (beta - 1) + t - t0

        t_min, t_max = p0 ** (1 / beta), 1e8
        root = bisect(func, t_min, t_max, full_output=False)
        print(f"kkt, t = {root}, p= {root ** beta}")
        return root, root ** beta


def gradient(sc, len_var):
    """
    Compute the gradient of the objective at x = (n, p, h)
    :param sc: Scenario
    :return: a numpy array
    """
    pn, uav = sc.pn, sc.uav
    g = np.zeros(len_var)
    n_UE = int(len_var/2)
    id_cnt, cur = 0, 0
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(s.UEs)):
            u = s.UEs[k]
            A, B, C, beta = pn.g_0 * u.tilde_g, pn.sigma * uav.theta ** 2, uav.h + u.loc_x ** 2 + u.loc_y ** 2, pn.alpha / 2
            if abs(function(sc) + u.x * s.b_width * np.log(1 + A * u.p / (B * C ** beta))) < 1e-8:
                id_cnt += 1
                partial_x = s.b_width * np.log(1 + A * u.p / (B * C ** beta))
                partial_p = u.x * s.b_width * A / (B * C ** beta + A * u.p)
                partial_h = -beta * u.x * s.b_width * A * u.p / (B * C ** beta + A * u.p) / C
                #partial_h = -beta * u.x * s.b_width * A * u.p / (B * C ** beta + A * u.p)
                g[cur], g[n_UE+cur] = partial_x, partial_p
                g[-1] += partial_h
            cur += 1
    return g / (-id_cnt)


def gradient_no_height(sc, len_var):
    """
    Compute the gradient of the objective at x = (n, p, h)
    :param sc: Scenario
    :return: a numpy array
    """
    pn, uav = sc.pn, sc.uav
    g = np.zeros(len_var)
    n_UE = int(len_var/2)
    id_cnt, cur = 0, 0
    f = function(sc)
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(s.UEs)):
            u = s.UEs[k]
            A, B, C, beta = pn.g_0 * u.tilde_g, pn.sigma * uav.theta ** 2, uav.h + u.loc_x ** 2 + u.loc_y ** 2, pn.alpha / 2
            if abs(f + u.x * s.b_width * np.log(1 + A * u.p / (B * C ** beta))) < 1e-8:
                id_cnt += 1
                g[cur], g[n_UE+cur] = s.b_width * np.log(1 + A * u.p / (B * C ** beta)), u.x * s.b_width * A / (B * C ** beta + A * u.p)
            cur += 1
    return g / (-id_cnt)


def function(sc):
    """
    Compute the function value at the current settings
    :param sc: Scenario
    :return: float
    """
    pn, uav = sc.pn, sc.uav
    func_val = -1e8
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(s.UEs)):
            u = s.UEs[k]
            A, B, C, beta = pn.g_0 * u.tilde_g, pn.sigma * uav.theta ** 2, uav.h + u.loc_x ** 2 + u.loc_y ** 2, pn.alpha / 2
            f_tmp = -u.x*s.b_width * np.log(1 + A * u.p / (B * C ** beta))
            if func_val < f_tmp:
                func_val = f_tmp
    return func_val

def proj_linear_p(sc, z):
    """
    Project p onto the linear constraints (i.e., (2)) of the problem
    :param sc: Scenario
    :param z: numpy array
    :return: numpy array
    """
    cursor = 0
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(s.UEs)):
            cursor += 1
    a, b = np.ones_like(z[cursor:2*cursor]), sc.pn.p_max
    p_proj = affine_proj(z[cursor:2*cursor], a, b)
    return np.concatenate((z[:cursor], p_proj, np.array([z[-1]])))


def proj_linear_x(sc, z):
    """
    Project x onto the linear constraints (i.e., (3)) of the problem
    :param sc: Scenario
    :param z: numpy array
    :return: numpy array
    """
    a, b = [], sc.pn.b_tot
    cursor = 0
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(s.UEs)):
            a.append(s.b_width)
            cursor += 1
    x_proj = affine_proj(z[:cursor], np.array(a), b)
    return np.concatenate((x_proj, z[cursor:]))


def proj_bounds(sc, z) -> np.ndarray:
    """
    Project z onto the bounds defined by lb and ub
    :param sc: The Scenario
    :param z: numpy array represents the variable z = (n, p, h)
    :return: numpy array
    """
    lb, ub = np.zeros_like(z), np.ones_like(z)*1e8
    lb[-1] = max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c*sc.pn.t_s))**2, sc.pn.h_min**2)
    ub[-1] = (min(sc.pn.h_max, np.sqrt(sc.uav.h_bar) + sc.uav.c*sc.pn.t_s))**2
    return np.clip(z, lb, ub)


def proj_nonlinear(sc, z0, i) -> np.ndarray:
    """
    Project z onto a nonlinear constraint
    :param sc: the Scenario
    :param z0: numpy array
    :param i: the index of the constraints
    :return: numpy array
    """
    n_UE = int(len(z0)/2)
    cur, ue = 0, None
    for j in range(len(sc.slices)):
        s = sc.slices[j]
        for k in range(len(s.UEs)):
            if cur == i:
                ue = s.UEs[k]
            cur += 1

    p0, h0 = z0[n_UE+i], z0[-1]
    beta, d = sc.pn.alpha/2, ue.loc_x**2 + ue.loc_y**2
    B = sc.pn.g_0*ue.tilde_g/(sc.pn.sigma*(sc.uav.theta**2)*ue.tilde_r)
    if (h0+d)**beta <= B*p0:
        #print("not projected **************** nolinear****")
        return z0
    else:
        #print("projected **************** nolinear****")
        if beta == 1:
            z0[-1] = (h0*B**2 + p0*B - d)/(B**2 + 1)
            z0[n_UE + i] = (z0[-1]+d)/B
            return z0
        else:
            def func(h):
                return beta*(h+d)**(2*beta-1)/(B**2) - beta*p0*(h+d)**(beta-1)/B + h - h0
            t_min, t_max = (p0*B) ** (1 / beta) - d, 1e8
            h = bisect(func, t_min, t_max, full_output=False)
            p = (h+d)**beta/B
            z0[n_UE+i] = p,
            z0[-1] = h
            #print(f"kkt, t = {root}, p= {root ** beta}")
            return z0


def proj_nonlinear_alpha2(sc, z0) -> np.ndarray:
    """
    Project z onto a nonlinear constraint provided that alpha=2
    :param sc: the Scenario
    :param z0: numpy array
    :return: numpy ndarray
    """
    z = np.zeros_like(z0)
    n = int(len(z)/2)
    idx = 0
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(s.UEs)):
            ue = s.UEs[k]
            beta, d = sc.pn.alpha / 2, ue.loc_x ** 2 + ue.loc_y ** 2
            B = sc.pn.g_0 * ue.tilde_g / (sc.pn.sigma * (sc.uav.theta ** 2) * ue.tilde_r)
            if (z0[-1]+d)**beta <= B*z0[n+idx]:
                z[idx+n] = z0[idx+n]
                z[-1] += z0[-1]
            else:
                z[idx+n] = (z0[-1] * B ** 2 + z0[n+idx] * B - d) / (B ** 2 + 1)
                z[-1] += (z0[-1] + d) / B
            idx += 1
    return z


def average_operator(sc, z):
    """
    Compute the averaged operator, which is non-expansive
    :return: a ndarray
    """
    u = np.zeros_like(z)
    u += proj_linear_x(sc, z)
    u += proj_linear_p(sc, z)
    # n_UE = int(len(z)/2)
    # u += proj_nonlinear_alpha2(sc, z)
    return u/(2)


def average_operator_parallel(sc, z):
    """
    :param sc:
    :param z:
    :return:
    """
    u = np.zeros_like(z)
    u += proj_linear_x(sc, z)
    u += proj_linear_p(sc, z)
    n_UE = int(len(z) / 2)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(proj_nonlinear, sc, z, i) for i in range(n_UE)]
        for future in concurrent.futures.as_completed(futures):
            u += future.result()

    return u/(n_UE+2)

def firm_up_operator(sc, z):
    """
    Compute the firmly non-expansive operator
    :return: a ndarray
    """
    return (z + average_operator(sc, z))/2
    #return (z + average_operator_parallel(sc, z)) / 2


def qusi_subgradient_alg(sc):
    eps = 1e-6
    pn = sc.pn
    n = 0  # the number of UEs in the system
    band_max = -1
    for i in range(len(sc.slices)):
        band_max = max(band_max, sc.slices[i].b_width)
        for k in range(len(sc.slices[i].UEs)):
            n += 1
    #  Set initial value for x, p and h
    p = np.ones(n) * pn.p_max / n
    x = np.ones(n) * pn.b_tot / n / band_max
    z = np.concatenate((x, p, np.array([(pn.h_min**2)])))
    sc.set_x(z[:n])
    sc.set_p(z[n:2*n])
    sc.set_h(z[-1])
    opt_old, opt = -1e8, 0

    # algorithm parameters
    cnt, max_iter = 0, 10000
    v_seq = map(lambda k: 1/(int(k/100)+1), itertools.count(1))
    #v_seq = map(lambda k: 0.001, itertools.count(1))
    a_seq = map(lambda k: 0.5, itertools.count(1))

    plt.ion()  # 交互模式开启，允许动态绘图
    plt.figure()  # 创建一个新图形

    #初始化绘图
    plt.plot([], [], 'b-', label='Function Value')
    plt.title('Objective Function Value per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    plt.xlim(0, int(max_iter/100))
    plt.ylim(-650, 0)
    function_values = []

    for k, v, a in zip(range(max_iter), v_seq, a_seq):
        g = gradient(sc, 2*n+1)
        g /= np.linalg.norm(g)
        u = firm_up_operator(sc, z - v*g)
        u = a*z + (1-a)*u
        z = proj_bounds(sc, u)
        sc.set_vars(z)
        n_plots = 10
        if not k%n_plots:
            print(f"k = {k}, f = {function(sc):.8f}")
            function_values.append(function(sc))
            #function_values.append(z[-1])
            plt.plot(range(int(k/n_plots) + 1), function_values, 'b-', label='Function Value')
            plt.draw()
            plt.pause(0.1)
    plt.ioff()
    plt.show()
    return z



if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    #sc = create_scenario(10, 10)
    sc = load_scenario('scenario_2_slices.pickle')
    z = qusi_subgradient_alg(sc)
    print(z)
