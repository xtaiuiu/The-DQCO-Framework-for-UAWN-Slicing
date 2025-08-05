import itertools

import numpy as np
from matplotlib import pyplot as plt

from algorithms.main_algorithms.relax_quasi_subgradient import gradient, function
from scenarios.scenario_creators import create_scenario, load_scenario
from utils.projction import proj_generalized_simplex


def qusi_subgradient_no_QoS(sc):
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
    u = np.concatenate((x, p, np.array([(pn.h_min**2+100)])))
    sc.set_x(u[:n])
    sc.set_p(u[n:2*n])
    sc.set_h(u[-1])
    opt_old, opt = -1e8, 0

    # algorithm parameters
    cnt, max_iter = 0, 50000
    #v_seq = map(lambda k: 1/(int(k/100)+1), itertools.count(1))
    v_seq = map(lambda k: 0.5, itertools.count(1))
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
        u = u - v*g  # quasi-subgradient descent
        # project u onto the feasible set
        u[:n] = proj_generalized_simplex(B_array, u[:n], pn.b_tot)
        u[n:2*n] = proj_generalized_simplex(np.ones_like(B_array), u[n:2*n], pn.p_max)
        u[-1] = np.clip(u[-1], max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c*sc.pn.t_s))**2, sc.pn.h_min**2),
                        (min(sc.pn.h_max, np.sqrt(sc.uav.h_bar) + sc.uav.c*sc.pn.t_s))**2)
        sc.set_vars(u)
        n_plots = 100
        if not k%n_plots:
            print(f"k = {k}, f = {function(sc):.8f}, h = {u[-1]}")
            function_values.append(function(sc))
            #function_values.append(z[-1])
            plt.plot(range(int(k/n_plots) + 1), function_values, 'b-', label='Function Value')
            plt.draw()
            plt.pause(0.05)
    plt.ioff()
    plt.show()
    return u



if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    sc = create_scenario(10, 10)
    #sc = load_scenario('scenario_2_slices.pickle')
    z = qusi_subgradient_no_QoS(sc)
    print(z)