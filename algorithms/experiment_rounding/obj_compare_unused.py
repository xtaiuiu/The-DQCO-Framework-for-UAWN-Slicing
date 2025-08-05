import numpy as np

from algorithms.main_algorithms.run_simulations import qusi_subgradient_no_QoS_dynamic, \
    rounded_solution
from algorithms.main_algorithms.rounding.rounding_algorithms import rounding_by_worst_condition
from scenarios.scenario_creators import create_scenario, load_scenario
from utils.projction import proj_generalized_simplex, rounding_by_opt_condition


def run(n_repeats=1):
    #np.random.seed(0)
    n_slices = np.arange(4, 6) * 10
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    for n in n_slices:
        sc = create_scenario(10, 10)
        sc.pn.b_tot = 1000
        sc.pn.p_max = 1000
        #sc = load_scenario('scenario_2_slices.pickle')
        # opt_var, opt_fval = qusi_subgradient_no_QoS_dynamic(sc, plot=True)

        opt_var, opt_fval, _ = qusi_subgradient_no_QoS_dynamic(sc, projector=proj_generalized_simplex, plot=False)
        opt_round_var, opt_round_val = rounded_solution(sc, opt_var, method=rounding_by_opt_condition)
        near_round_var, near_round_val = rounded_solution(sc, opt_var, method=rounding_by_worst_condition)
        print(f'opt_round_val = {opt_round_val}, near_round_val = {near_round_val}')


if __name__ == '__main__':
     run()
