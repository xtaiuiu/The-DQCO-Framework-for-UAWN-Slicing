import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mealpy import PSO, FloatVar
from mealpy.math_based import CircleSA, SCA, SHIO
from mealpy.utils.problem import Problem
import time

from algorithms.experiment_comparison.problem_generator import NsProblem
from algorithms.main_algorithms.block_descent_nonsmooth.block_descent_max_min_rate import block_coordinate_descent
from algorithms.main_algorithms.quasi_subgradient_no_QoS_dynamic import qusi_subgradient_no_QoS_dynamic
from scenarios.scenario_creators import create_scenario
from utils.projction import proj_generalized_simplex



def run(n_repeats=1):
    n_UEs = np.arange(2, 9) * 10

    df_rate_avg = pd.DataFrame(
        {'BCD': np.zeros(len(n_UEs)), 'AIW-PSO': np.zeros(len(n_UEs)), 'SHIO': np.zeros(len(n_UEs))})
    for repeat in range(n_repeats):
        print(f" ####################### repeat = {repeat} ##############################")

        rate_qpa_arr, rate_pso_arr, rate_shio_arr = [], [], []
        for n in n_UEs:
            print(f" ####################### repeat = {repeat}, n_UE = {n} ##############################")
            logging.basicConfig(level=logging.WARNING)
            np.set_printoptions(formatter={'float': '{:0.4f}'.format})
            sc = create_scenario(n, 10)
            sc.pn.b_tot = 100
            sc.pn.p_max = 50
            len_x = len(sc.scenario_variables().x)

            opt_val, x, p, _ = block_coordinate_descent(sc)
            ub = np.concatenate((x, p))
            print(f'f_qpa = {opt_val}')
            problem_cop = NsProblem(sc, FloatVar(lb=np.zeros(len_x * 2), ub=ub * 1.1), minmax="min")

            problem_cop.reset_scenario()
            model_pso = PSO.OriginalPSO(epoch=1000, pop_size=100)
            model_pso.solve(problem=problem_cop)
            f_pso = model_pso.g_best.target.fitness

            problem_cop.reset_scenario()
            model_shio = SHIO.OriginalSHIO(epoch=1000, pop_size=100)
            model_shio.solve(problem_cop)
            f_shio = model_shio.g_best.target.fitness

            # append results
            rate_qpa_arr.append(-opt_val)
            rate_pso_arr.append(f_pso)
            rate_shio_arr.append(f_shio)
        df_rate = pd.DataFrame({'BCD': rate_qpa_arr, 'AIW-PSO': rate_pso_arr, 'SHIO': rate_shio_arr})
        df_rate_avg += df_rate
    df_rate_avg /= (n_repeats * (-1))
    df_rate_avg.to_excel('objective_value_num_UEs_jnl.xlsx')
    df_rate_avg.plot()
    plt.show()


def load_and_plot():
    fontsize = 20
    df_obj_avg = pd.read_excel('objective_value_num_UEs_jnl.xlsx', index_col=0)
    df_obj_avg.columns = [r'$DQCO$', 'SCA', 'MADDPG']
    G = np.arange(df_obj_avg.shape[0])
    plt.rcParams.update({'font.size': fontsize})

    ax = df_obj_avg.plot(legend=True, lw=2, xlabel='No. of UEs', ylabel='Max-min user rate (Mbsp)',
                         fontsize=fontsize, style=["r-s", "m-.d", "c-->"], markersize=10, grid=True,
                         markerfacecolor='none')
    ax.set_xticks(G)
    #plt.ylim((0.295, 0.5))
    ax.set_xticklabels([str((i + 2) * 10) for i in G], fontsize=fontsize)
    plt.show()


if __name__ == '__main__':
    #run(20)
    load_and_plot()