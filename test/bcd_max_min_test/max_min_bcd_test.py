import unittest

import numpy as np

from algorithms.main_algorithms.block_descent_nonsmooth.block_descent_max_min_rate import block_coordinate_descent
from algorithms.main_algorithms.block_descent_nonsmooth.subproblem_p import optimize_p_kkt_with_lb
from algorithms.main_algorithms.block_descent_nonsmooth.subproblem_x import optimize_x_kkt_with_lb
from algorithms.main_algorithms.run_simulations import qusi_subgradient_no_QoS_dynamic
from scenarios.scenario_creators import create_scenario


class MyTestCase(unittest.TestCase):
    def test_max_min_bcd(self):
        sc = create_scenario(12, 1)
        # sc = load_scenario('scenario_2_slices.pickle')
        pn = sc.pn
        h = max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c * sc.pn.t_s)) ** 2, sc.pn.h_min ** 2)
        sc.set_h(h)
        sc.set_theta(h)
        rates, obj = sc.get_UE_rate()
        val, x, p, h = block_coordinate_descent(sc, first_block=0,
                                            x_optimizer=optimize_x_kkt_with_lb, p_optimizer=optimize_p_kkt_with_lb)
        sc.reset_scenario()
        opt_fval, x_2, p_2, h_2, _ = qusi_subgradient_no_QoS_dynamic(sc, plot=False)
        self.assertGreaterEqual(val, opt_fval)
        self.assertAlmostEqual(h, h_2)
        # print(f"val = {val}, h = {h}, x = {x}, p = {p}")
        # print(f"val_qpa = {opt_fval}, var = {opt_var}")

if __name__ == '__main__':
    unittest.main()
