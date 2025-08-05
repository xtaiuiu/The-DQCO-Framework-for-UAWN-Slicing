from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mealpy import FloatVar
from mealpy.math_based import SHIO
from mealpy.swarm_based import PSO
from pandas import DataFrame

from algorithms.experiment_comparison.problem_generator import NsProblem
from algorithms.main_algorithms.block_descent_nonsmooth.block_descent_max_min_rate import block_coordinate_descent
from algorithms.main_algorithms.run_simulations import qusi_subgradient_no_QoS_dynamic
from scenarios.scenario_creators import create_scenario
from utils.projction import proj_generalized_simplex_QP, proj_generalized_simplex


# We test two problem with 50 and 100 UEs, respectively
def run_num_iterations(filename='num_iterations.xlsx'):
    np.random.seed(0)
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    n_UEs = np.array([20])
    alpha_arr = np.array([0.025, 0.05, 0.1])
    df = DataFrame()
    val_list, columns = [], []
    max_len = 0
    for n in n_UEs:
        sc = create_scenario(n, 10)
        sc.pn.b_tot = 1000
        sc.pn.p_max = 1000
        for alpha in alpha_arr:
            sc.reset_scenario()
            _, _, func_val_kkt = qusi_subgradient_no_QoS_dynamic(sc, projector=proj_generalized_simplex, alpha_k=alpha, plot=False)
            val_list.append(func_val_kkt)
            columns.append(f'{n*10} UEs, alpha_k = {alpha}')
            max_len = max(len(func_val_kkt), max_len)
    for i in range(len(val_list)):
        df[columns[i]] = np.array(val_list[i] + [val_list[i][-1]] * (max_len - len(val_list[i])))
    df.to_excel(filename)
    df.plot()
    plt.show()
    return df

def run_num_iterations_heu(filename='num_iterations_heu_jnl.xlsx'):
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    sc = create_scenario(20, 10)
    sc.pn.b_tot = 1000
    sc.pn.p_max = 1000
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
    # model_pso = CEM.OriginalCEM(epoch=1000, pop_size=50, n_best = 20, alpha = 0.7)
    model_pso.solve(problem=problem_cop)
    model_pso.history.save_global_objectives_chart(filename="pso_iterations/goc")
    f_pso = -model_pso.g_best.target.fitness

    # 4. SHIO
    problem_cop.reset_scenario()
    model_shio = SHIO.OriginalSHIO(epoch=1000, pop_size=100)
    model_shio.solve(problem_cop)
    model_shio.history.save_global_objectives_chart(filename="shio_iterations/goc")
    f_shio = -model_shio.g_best.target.fitness


def plot_num_iterations(filename='num_iterations.xlsx'):
    df = pd.read_excel(filename, index_col=0)[:100]
    df = df.iloc[:, [0, 3, 4, 5]]
    plt.rcParams.update({'font.size': 18})

    ax = df.plot(legend=True, lw=2, xlabel=r'Number of iterations', ylabel='Function value',
                          fontsize=16, style=["r-s", "k:^", "m-.d", "c-->"], markersize=4, grid=True,
                          markerfacecolor='none')
    # ax.set_xticks(G)
    # ax.set_xticklabels([str(i + 6) for i in G], fontsize=16)
    plt.rcParams['legend.markerscale'] = 2
    ax.legend(['BCD', 'QPA', 'AIW-PSO', 'SHIO'])
    #plt.legend()

    df = df.iloc[:, [0]]
    df = df[:5]
    plt.rcParams['legend.markerscale'] = 2
    ax = df.plot(legend=False, lw=2,
                 fontsize=32, style=["r-s", "k:^", "m-.d", "c-->"], markersize=10, grid=True,
                 markerfacecolor='none')
    #ax.legend()
    plt.show()



if __name__ == '__main__':
    #run_num_iterations()
    # run_num_iterations_heu()
    plot_num_iterations()
