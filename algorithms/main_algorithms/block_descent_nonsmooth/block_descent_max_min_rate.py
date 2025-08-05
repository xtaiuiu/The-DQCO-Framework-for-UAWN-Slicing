import math

import numpy as np

from algorithms.main_algorithms.block_descent_nonsmooth.subproblem_p import optimize_p_kkt_with_lb
from algorithms.main_algorithms.block_descent_nonsmooth.subproblem_x import optimize_x_kkt_with_lb
from algorithms.main_algorithms.mip_bcd import optimize_x_continuous_kkt, optimize_p_kkt, optimize_p
from algorithms.main_algorithms.quasi_subgradient_no_QoS_dynamic import qusi_subgradient_no_QoS_dynamic
from algorithms.main_algorithms.relax_quasi_subgradient import function
from scenarios.scenario_creators import load_scenario, create_scenario
import logging


def block_coordinate_descent(sc, first_block=None, x_optimizer=optimize_x_kkt_with_lb, p_optimizer=optimize_p_kkt_with_lb):
    """
    Optimize the SP by using block coordinate descent algorithm
    :param sc: the scenario
    :param first_block: 0 or 1, where 0 represents x is optimized first, otherwise p is the first
    :return: f_opt, x_opt, n_iter. n_iter is the number of iterations
    """
    pn = sc.pn
    n = 0
    h = max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c * sc.pn.t_s)) ** 2, sc.pn.h_min ** 2)
    sc.set_h(h)
    sc.set_theta(h)
    B_array = []
    for i in range(len(sc.slices)):
        for k in range(len(sc.slices[i].UEs)):
            n += 1
            B_array.append(sc.slices[i].b_width)
    B_array = np.array(B_array)
    if first_block is None:
        # generate the order randomly
        first_block = np.random.choice([0, 1])
    #  Set initial value for p and h
    p = np.ones(n) * pn.p_max / n
    x = np.ones(n) * pn.b_tot / n
    x /= B_array
    # sc.set_x(np.array([33.321, 33.324, 2.779]))
    # sc.set_p(np.array([4, 4, 4]))
    sc.set_x(x)
    sc.set_p(p)

    eps = 1e-4
    opt_old, opt = -1e8, 0
    f_val = opt

    cnt, max_iter = 0, 100
    while cnt < max_iter and np.abs((opt - opt_old)) > eps:
        logging.info(f"*********************** BCD loop cnt: {cnt: 04} *********************** ")
        if first_block == 0:
            #  Step 1: optimize x
            f_val, x = x_optimizer(sc)
            logging.info(f" optimize x, f_x = {f_val: .12f}")
            sc.set_x(x)
            #  Step 2: optimize p
            f_val, p = p_optimizer(sc)
            logging.info(f" optimize p, f_p = {f_val: .12f}")
            sc.set_p(p)
        else:
            #  Step 1: optimize p
            f_val, p = p_optimizer(sc)
            logging.info(f" optimize p, f_p = {f_val: .12f}")
            sc.set_p(p)
            #  Step 2: optimize x
            f_val, x = x_optimizer(sc)
            logging.info(f" optimize x, f_x = {f_val: .12f}")
            sc.set_x(x)
        opt_old = opt
        opt = f_val
        cnt += 1
    #  Return the outcomes
    vars = sc.scenario_variables()
    # print(f"x = {vars.x}")
    # print(f"p = {vars.p}")
    # print(f"h = {vars.h}")
    return opt, x, p, h


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    sc = create_scenario(12, 1)

    # sc = load_scenario('scenario_2_slices.pickle')
    pn = sc.pn
    h = max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c * sc.pn.t_s)) ** 2, sc.pn.h_min ** 2)
    sc.set_h(h)
    sc.set_theta(h)
    rates, obj = sc.get_UE_rate()
    print(rates)
    print(obj)
    val, x, p, h = block_coordinate_descent(sc, first_block=0,
        x_optimizer=optimize_x_kkt_with_lb, p_optimizer=optimize_p_kkt_with_lb)
    opt_bcd = -function(sc)
    sc.reset_scenario()
    _, f_subgradient, _ = qusi_subgradient_no_QoS_dynamic(sc, plot=True)
    print(f'opt_bcd_no_lb = {opt_bcd}, opt_subgradient = {-f_subgradient}, diff = {opt_bcd + f_subgradient}')
