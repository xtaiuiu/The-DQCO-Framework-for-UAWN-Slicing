import time
import unittest

import numpy as np

from algorithms.main_algorithms.block_descent_nonsmooth.subproblem_p import optimize_p_kkt_with_lb, \
    optimize_p_cvx_with_lb
from scenarios.scenario_creators import create_scenario


class MyTestCase(unittest.TestCase):
    def test_optimize_p_kkt_with_lb(self):
        sc = create_scenario(50, 10)
        h = max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c * sc.pn.t_s)) ** 2, sc.pn.h_min ** 2)
        sc.set_h(h)
        sc.set_theta(h)
        # for s in sc.slices:
        #     s.set_sla(1)
        sc.slices[0].set_sla(0.5)
        t1 = time.perf_counter()
        f_opt, p_opt = optimize_p_kkt_with_lb(sc)
        print(f"kkt runtime = {time.perf_counter() - t1}")
        print(f"f_opt = {f_opt}, p = {p_opt}")
        t1 = time.perf_counter()
        f_cvx, p_cvx = optimize_p_cvx_with_lb(sc)
        print(f"cvx runtime = {time.perf_counter() - t1}")
        print(f"f_value = {f_cvx}, p = {p_cvx},")

    def test_optimize_p_cvx(self):
        sc = create_scenario(10, 10)
        h = max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c * sc.pn.t_s)) ** 2, sc.pn.h_min ** 2)
        sc.set_h(h)
        sc.set_theta(h)
        # for s in sc.slices:
        #     s.set_sla(1)
        sc.slices[0].set_sla(2)



if __name__ == '__main__':
    unittest.main()
