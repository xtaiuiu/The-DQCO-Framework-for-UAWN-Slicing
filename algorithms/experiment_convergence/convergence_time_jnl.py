import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mealpy import FloatVar
from mealpy.math_based import CEM, SHIO
from mealpy.swarm_based import PSO

from algorithms.experiment_comparison.problem_generator import NsProblem
from algorithms.main_algorithms.block_descent_nonsmooth.block_descent_max_min_rate import block_coordinate_descent
from algorithms.main_algorithms.block_descent_nonsmooth.subproblem_p import optimize_p_kkt_with_lb, \
    optimize_p_cvx_with_lb
from algorithms.main_algorithms.block_descent_nonsmooth.subproblem_x import optimize_x_kkt_with_lb, \
    optimize_x_cvx_with_lb
from algorithms.main_algorithms.run_simulations import qusi_subgradient_no_QoS_dynamic
from scenarios.scenario_creators import create_scenario


def run_bcd_runtime(n_repeats=1):
    n_UEs = np.arange(1, 9) * 500

    df_avg_time = pd.DataFrame(
        {'KKT-KKT': np.zeros(len(n_UEs)), 'LP-KKT': np.zeros(len(n_UEs)), 'KKT-CVX': np.zeros(len(n_UEs)), 'LP-CVX': np.zeros(len(n_UEs))})
    for repeat in range(n_repeats):
        print(f" ####################### repeat = {repeat} ##############################")
        bcd_arr, lp_kkt_arr, kkt_cvx_arr, lp_cvx_arr = [], [], [], []
        for n in n_UEs:
            print(f" ####################### repeat = {repeat}, n_UE = {n} ##############################")
            logging.basicConfig(level=logging.WARNING)
            np.set_printoptions(formatter={'float': '{:0.4f}'.format})
            sc = create_scenario(n, 10)
            sc.pn.b_tot = 8000
            sc.pn.p_max = 8000

            t = time.perf_counter()
            val, x, p, h = block_coordinate_descent(sc, first_block=0,
                                                    x_optimizer=optimize_x_kkt_with_lb,
                                                    p_optimizer=optimize_p_kkt_with_lb)
            bcd_arr.append(time.perf_counter() - t)

            t = time.perf_counter()
            val, x, p, h = block_coordinate_descent(sc, first_block=0,
                                                    x_optimizer=optimize_x_cvx_with_lb,
                                                    p_optimizer=optimize_p_kkt_with_lb)
            lp_kkt_arr.append(time.perf_counter() - t)

            t = time.perf_counter()
            val, x, p, h = block_coordinate_descent(sc, first_block=0,
                                                    x_optimizer=optimize_x_cvx_with_lb,
                                                    p_optimizer=optimize_p_cvx_with_lb)
            lp_cvx_arr.append(time.perf_counter() - t)

            t = time.perf_counter()
            val, x, p, h = block_coordinate_descent(sc, first_block=0,
                                                    x_optimizer=optimize_x_kkt_with_lb,
                                                    p_optimizer=optimize_p_cvx_with_lb)
            kkt_cvx_arr.append(time.perf_counter() - t)

        df_time = pd.DataFrame({'KKT-KKT': bcd_arr,'LP-KKT': lp_kkt_arr, 'KKT-CVX': kkt_cvx_arr, 'LP-CVX': lp_cvx_arr})
        df_avg_time += df_time
    df_avg_time /= n_repeats
    df_avg_time.to_excel('runtime_bcd_compare_jnl.xlsx')
    df_avg_time.plot()
    plt.show()


def run_benchmark_runtime(n_repeats=1):
    n_UEs = np.arange(1, 6) * 20

    df_avg_time = pd.DataFrame({'BCD': np.zeros(len(n_UEs)), 'QPA': np.zeros(len(n_UEs))})
    df_avg_obj = pd.DataFrame({'BCD': np.zeros(len(n_UEs)), 'QPA': np.zeros(len(n_UEs))})
    for repeat in range(n_repeats):
        print(f" ####################### repeat = {repeat} ##############################")
        qpa_time_arr, bcd_time_arr = [], []
        qpa_obj_arr, bcd_obj_arr = [], []
        for n in n_UEs:
            print(f" ####################### repeat = {repeat}, n_UE = {n} ##############################")
            logging.basicConfig(level=logging.INFO)
            np.set_printoptions(formatter={'float': '{:0.4f}'.format})
            sc = create_scenario(n, 10)
            sc.pn.b_tot = 300
            sc.pn.p_max = 200

            # 2. BCD
            sc.reset_scenario()
            t = time.perf_counter()
            f_bcd, x, p, _ = block_coordinate_descent(sc)
            bcd_time_arr.append(time.perf_counter() - t)
            bcd_obj_arr.append(f_bcd)

            # 1. QPA
            sc.reset_scenario()
            t = time.perf_counter()
            f_qpa, opt_var, _ = qusi_subgradient_no_QoS_dynamic(sc, plot=False)
            qpa_time_arr.append(time.perf_counter() - t)
            qpa_obj_arr.append(f_qpa)

        df_time = pd.DataFrame({'BCD': bcd_time_arr, 'QPA': qpa_time_arr})
        df_obj = pd.DataFrame({'BCD': bcd_obj_arr, 'QPA': qpa_obj_arr})
        df_avg_time += df_time
        df_avg_obj += df_obj
    df_avg_time /= n_repeats
    df_avg_obj /= n_repeats
    df_avg_time.to_excel('runtime_time_compare_jnl.xlsx')
    df_avg_obj.to_excel('runtime_obj_compare_jnl.xlsx')
    df_avg_obj.plot()
    df_avg_time.plot()
    plt.show()


def load_plot_bcd():
    fontsize = 20
    df_obj_avg = pd.read_excel('runtime_bcd_compare_jnl.xlsx', index_col=0)
    df_obj_avg.columns = ['BCD', 'LP-KKT', 'KKT-CVX', 'LP-CVX']
    G = np.arange(df_obj_avg.shape[0])
    plt.rcParams.update({'font.size': fontsize})

    ax = df_obj_avg.plot(legend=True, lw=2, xlabel=r'Number of UEs ($\times 10^2$)', ylabel='Convergence time (s)',
                         fontsize=fontsize, style=["r-s", "m-.d", "c-->", "b:o"], markersize=10, grid=True,
                         markerfacecolor='none')
    ax.set_xticks(G)
    plt.ylim((0, 25))
    ax.set_xticklabels([str((i + 1) * 5) for i in G], fontsize=fontsize)

    df = df_obj_avg.iloc[:, [0]]
    df = df[len(G) - 4:]
    ax2 = df.plot(legend=False, lw=2,
                         fontsize=32, style=["r-s", "m-.d", "c-->", "b:o"], markersize=10, grid=True,
                         markerfacecolor='none')
    ax2.set_xticklabels([str((i + 1) * 5) for i in np.arange(len(G)-5, len(G))], fontsize=32)
    plt.ylim((0.18, 0.35))

    plt.show()


if __name__ == '__main__':
    # run_bcd_runtime(2)
    load_plot_bcd()
    # run_benchmark_runtime(10)