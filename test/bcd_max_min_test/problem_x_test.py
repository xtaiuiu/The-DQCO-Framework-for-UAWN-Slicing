import time
import unittest

from algorithms.main_algorithms.block_descent_nonsmooth.subproblem_x import optimize_x_kkt_with_lb, \
    optimize_x_cvx_with_lb
from scenarios.scenario_creators import create_scenario
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_ptimize_p_kkt_with_lb(self):
        np.random.seed(2000)
        sc = create_scenario(5, 10)
        h = max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c * sc.pn.t_s)) ** 2, sc.pn.h_min ** 2)
        sc.set_h(h)
        sc.set_theta(h)
        # for s in sc.slices:
        #     s.set_sla(1)
        # sc.slices[0].set_sla(2)
        t1 = time.perf_counter()
        f_opt, x_opt = optimize_x_kkt_with_lb(sc)
        print(f"kkt runtime = {time.perf_counter() - t1}")
        print(f"f_opt = {f_opt}, x = {x_opt}")
        t1 = time.perf_counter()
        status, value, z, x = optimize_x_cvx_with_lb(sc)
        print(f"cvx runtime = {time.perf_counter() - t1}")
        print(f"status = {status}, value = {value}, x = {x}, z = {z}")


if __name__ == '__main__':
    unittest.main()
