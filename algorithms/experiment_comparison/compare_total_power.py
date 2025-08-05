import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mealpy import PSO, FloatVar
from mealpy.math_based import CircleSA, SCA, SHIO, PSS, CEM
from mealpy.utils.problem import Problem
import time

from algorithms.experiment_comparison.problem_generator import NsProblem
from algorithms.main_algorithms.block_descent_nonsmooth.block_descent_max_min_rate import block_coordinate_descent
from algorithms.main_algorithms.quasi_subgradient_no_QoS_dynamic import qusi_subgradient_no_QoS_dynamic
from scenarios.scenario_creators import create_scenario
from utils.projction import proj_generalized_simplex


def run(n_repeats=20):
    powers = np.arange(1, 11) * 10

    df_rate_avg = pd.DataFrame(
        {'BCD': np.zeros(len(powers)), 'PSO': np.zeros(len(powers)), 'SHIO': np.zeros(len(powers))})
    for repeat in range(n_repeats):
        print(f" ####################### repeat = {repeat} ##############################")

        rate_pso_arr, rate_shio_arr, rate_bcd_arr = [], [], []
        for power in powers:
            print(f" ####################### repeat = {repeat}, bandwidth = {power} ##############################")
            logging.basicConfig(level=logging.WARNING)
            np.set_printoptions(formatter={'float': '{:0.4f}'.format})
            sc = create_scenario(50, 10)
            sc.pn.b_tot = 100
            sc.pn.p_max = power
            lb_x, lb_p = sc.get_xp_lb()
            lb = np.concatenate((lb_x, lb_p))
            # 1. BCD
            f_bcd, x, p, _ = block_coordinate_descent(sc, first_block=0)
            ub = np.concatenate((x, p))
            sc.reset_scenario()

            problem_cop = NsProblem(sc, FloatVar(lb=lb, ub=ub * 1.1), minmax="min")

            # 3. PSO
            problem_cop.reset_scenario()
            model_pso = PSO.OriginalPSO(epoch=1000, pop_size=100)
            #model_pso = CEM.OriginalCEM(epoch=1000, pop_size=50, n_best = 20, alpha = 0.7)
            model_pso.solve(problem=problem_cop)
            f_pso = -model_pso.g_best.target.fitness

            # 4. SHIO
            problem_cop.reset_scenario()
            model_shio = SHIO.OriginalSHIO(epoch=1000, pop_size=100)
            model_shio.solve(problem_cop)
            f_shio = -model_shio.g_best.target.fitness

            # append results
            rate_pso_arr.append(f_pso)
            rate_shio_arr.append(f_shio)
            rate_bcd_arr.append(f_bcd)

        df_rate = pd.DataFrame({'BCD': rate_bcd_arr, 'PSO': rate_pso_arr, 'SHIO': rate_shio_arr})
        df_rate_avg += df_rate
    df_rate_avg /= n_repeats
    df_rate_avg.to_excel('objective_value_powers_jnl.xlsx')
    df_rate_avg.plot()
    plt.show()


def load_and_plot():
    fontsize = 20
    df_obj_avg = pd.read_excel('objective_value_powers_jnl.xlsx', index_col=0)
    df_obj_avg.columns = [r'$BCD^2$', 'AIW-PSO', 'SHIO']
    G = np.arange(df_obj_avg.shape[0])
    plt.rcParams.update({'font.size': fontsize})

    ax = df_obj_avg.plot(legend=True, lw=2, xlabel=r'$P_{max}$ (MHz)', ylabel='Max-min user rate (Mbps)',
                         fontsize=fontsize, style=["r-s", "m-.d", "c-->"], markersize=10, grid=True,
                         markerfacecolor='none')
    ax.set_xticks(G)
    # plt.ylim((0.295, 0.5))
    ax.set_xticklabels([str((i + 1) * 10) for i in G], fontsize=fontsize)
    plt.legend(loc='upper right', bbox_to_anchor=(0.99, 0.9))
    plt.show()


if __name__ == '__main__':
    # run(20)
    load_and_plot()
