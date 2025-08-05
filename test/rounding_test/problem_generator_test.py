import unittest
import numpy as np
import copy

from algorithms.main_algorithms.rounding.problem_generator import simulated_problem
from algorithms.main_algorithms.rounding.rounding_algorithms import rounding_by_mip, rounding_by_knapsack, \
    rounding_by_opt_condition, rounding_nearest, rounding_by_random_order, rounding_by_worst_condition


class MyTestCase(unittest.TestCase):
    def SimulatedProblem(self):
        n_repeats = 400
        for i in range(n_repeats):
            n = np.random.choice([5, 10, 20])
            B_array = np.random.choice([0.1, 0.5, 2.5], n)
            c = (0.1 + np.random.rand((n)) * 0.5) * np.random.randint(20, 31)
            x = (0.1 + np.random.rand((n)) * 0.5) * np.random.randint(10, 21)
            lb = np.zeros(n)
            for k in range(n):
                lb[k] = max(0, int(x[k]) - 1)

            B_tot = int(x @ B_array) + 1
            x_con, f_con = simulated_problem(c, B_array, B_tot, lb)
            self.assertAlmostEqual(x_con @ B_array, B_tot, delta=1e-4)
            for j in range(n):
                self.assertGreaterEqual(x_con[j], lb[j], msg=f"x_con[j] = {x_con[j]}, lb[j] = {lb[j]}")
                self.assertGreaterEqual(x_con[j] * c[j] + 1e-8, f_con,
                                        msg=f"rate of i = {x_con[j] * c[j]}, f_con = {f_con}")
            self.assertTrue(np.any(x_con % 1 != 0), "The array does not contain a fraction")
        print(f" test_simulated_problem is over")

    def RoundingMIP(self):
        n_repeats = 300
        for i in range(n_repeats):
            n = np.random.choice([5, 10, 20])
            B_array = np.random.choice([0.1, 0.5, 2.5], n)
            c = (0.1 + np.random.rand((n)) * 0.5) * np.random.randint(20, 31)
            x = (0.1 + np.random.rand((n)) * 0.5) * np.random.randint(10, 21)
            lb = np.zeros(n)
            for k in range(n):
                lb[k] = max(0, int(x[k]) - 1)
            B_tot = int(x @ B_array) + 1
            x_int, f_int = rounding_by_mip(c, B_array, B_tot, lb)
            self.assertAlmostEqual(x_int[-1], f_int)
            for k in range(len(x_int) - 1):
                self.assertAlmostEqual(x_int[k], np.round(x_int[k]), delta=1e-6)
                self.assertGreaterEqual(x_int[k] * c[k] + 1e-6, f_int)
        print(f" test_rounding_by_mip is over")

    def testRoundingByOpt(self):
        n_repeats = 300
        for i in range(n_repeats):
            n = np.random.choice([5, 15, 30])
            B_array = np.random.choice([0.1, 0.5, 2.5], n)
            c = (0.1 + np.random.rand((n)) * 0.5) * np.random.randint(20, 31)
            x = (0.1 + np.random.rand((n)) * 0.5) * np.random.randint(10, 21)
            lb = np.zeros(n)
            for k in range(n):
                lb[k] = max(0, int(x[k]) - 1)
            B_tot = int(x @ B_array) + 1
            x_con, f_con = simulated_problem(c, B_array, B_tot, lb)  # generate the continuous variables
            x_opt = rounding_by_opt_condition(B_array, c, x_con, B_tot)
            x_near = rounding_nearest(B_array, c, copy.deepcopy(x_con), B_tot)
            x_random = rounding_by_random_order(B_array, c, copy.deepcopy(x_con), B_tot)
            x_worst = rounding_by_worst_condition(B_array, c, copy.deepcopy(x_con), B_tot)
            self.assertGreaterEqual(f_con, min(x_opt * c))
            self.assertGreaterEqual(min(x_opt * c), min(x_near * c),
                                    msg=f"x_con = {x_con}, x_opt = {x_opt}, x_near = {x_near}, c = {c}, B_array = {B_array}, B_tot = {B_tot}, lb = {lb}")
            self.assertGreaterEqual(min(x_opt * c), min(x_random * c),
                                    msg=f"x_con = {x_con}, x_opt = {x_opt}, x_near = {x_random}, c = {c}, B_array = {B_array}, B_tot = {B_tot}, lb = {lb}")
            self.assertGreaterEqual(min(x_opt * c), min(x_worst * c),
                                    msg=f"x_con = {x_con}, x_opt = {x_opt}, x_near = {x_worst}, c = {c}, B_array = {B_array}, B_tot = {B_tot}, lb = {lb}")
            #print(f"f_near = {min(x_near * c)}, f_opt = {min(x_opt * c)}")

        print(f"test rounding by opt is over")


if __name__ == '__main__':
    unittest.main()
