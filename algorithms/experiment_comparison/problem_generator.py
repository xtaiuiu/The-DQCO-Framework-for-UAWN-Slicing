# generate problem instance
import numpy as np
from mealpy import PSO, FloatVar
from mealpy.utils.problem import Problem

from algorithms.main_algorithms.relax_quasi_subgradient import function


def violate(value):
    return 0 if value <= 0 else value


class NsProblem(Problem):
    def __init__(self, sc, bounds, minmax, **kwargs):
        self.sc = sc
        self.n = len(sc.scenario_variables().x)  # n is the number of UEs
        super().__init__(bounds, minmax, **kwargs)
        # for key, value in kwargs.items():
        #     if key == 'sc':
        #         self.sc = value
        #self.sc = sc  # sc is the network Scenario

    # constraints
    def cons_x(self, x):
        band_max = -1
        B_array = []
        for i in range(len(self.sc.slices)):
            band_max = max(band_max, self.sc.slices[i].b_width)
            for k in range(len(self.sc.slices[i].UEs)):
                B_array.append(self.sc.slices[i].b_width)
        B_array = np.array(B_array)

        return B_array @ x[:self.n] - self.sc.pn.b_tot

    def cons_p(self, x):
        return np.sum(x[self.n:2 * self.n]) - self.sc.pn.p_max

    def obj_func(self, solution):
        pn, uav = self.sc.pn, self.sc.uav
        h = max((max(0, np.sqrt(uav.h_bar) - uav.c * pn.t_s)) ** 2, pn.h_min ** 2)
        self.sc.set_h(h)
        self.sc.set_x(solution[:self.n])
        self.sc.set_p(solution[self.n: self.n * 2])
        fx = function(self.sc)
        fx += violate(self.cons_x(solution[:self.n])) * 200000 + violate(
            self.cons_p(solution[self.n: self.n * 2])) * 200000
        return fx


    def reset_scenario(self):
        pn, uav = self.sc.pn, self.sc.uav
        h = max((max(0, np.sqrt(uav.h_bar) - uav.c * pn.t_s)) ** 2, pn.h_min ** 2)
        self.sc.set_h(h)
        self.sc.set_x(np.zeros(self.n))
        self.sc.set_p(np.zeros(self.n))
