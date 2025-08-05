import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker

from algorithms.main_algorithms.rounding.problem_generator import simulated_problem
from algorithms.main_algorithms.rounding.rounding_algorithms import rounding_by_opt_condition, rounding_nearest, \
    rounding_by_random_order, rounding_by_worst_condition
from utils.fairness import Jain_fairness


def single_run(variance, n=100):
    B_array = np.random.choice([1, 5, 10], n)
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

    # print(f'f_opt = {min(x_int_opt * c)}, f_nearest = {min(x_int_near * c)}')
    rate_opt, rate_near, rate_worst, rate_random = x_opt * c, x_near * c, x_worst * c, x_random * c
    return ((f_con - min(rate_opt))/f_con, (f_con - min(rate_near))/f_con, (f_con - min(rate_worst))/f_con, (f_con - min(rate_random))/f_con, Jain_fairness(rate_opt),
            Jain_fairness(rate_near), Jain_fairness(rate_worst), Jain_fairness(rate_random))



def run(n_repeats=10):
    n_UEs = np.arange(2, 11) * 10
    df_obj_avg = pd.DataFrame(
        {'Opt': np.zeros(len(n_UEs)), 'Nearest': np.zeros(len(n_UEs)), 'Worst': np.zeros(len(n_UEs)), 'Random': np.zeros(len(n_UEs))})
    df_fair_avg = pd.DataFrame(
        {'Opt': np.zeros(len(n_UEs)), 'Nearest': np.zeros(len(n_UEs)), 'Worst': np.zeros(len(n_UEs)), 'Random': np.zeros(len(n_UEs))})

    for repeat in range(n_repeats):
        if repeat % 100 == 0:
            print(f" ####################### repeat = {repeat} ##############################")
        obj_opt_arr, obj_near_arr, obj_worst_arr, obj_random_arr = [], [], [], []
        fair_opt_arr, fair_near_arr, fair_worst_arr, fair_random_arr = [], [], [], []
        for n in n_UEs:
            obj_opt, obj_near, obj_worst, obj_random, fair_opt, fair_near, fair_worst, fair_random = single_run(n)

            obj_opt_arr.append(obj_opt)
            obj_near_arr.append(obj_near)
            obj_worst_arr.append(obj_worst)
            obj_random_arr.append(obj_random)

            fair_opt_arr.append(fair_opt)
            fair_near_arr.append(fair_near)
            fair_worst_arr.append(fair_worst)
            fair_random_arr.append(fair_random)

        df_obj = pd.DataFrame({'Opt': obj_opt_arr, 'Nearest': obj_near_arr, 'Worst': obj_worst_arr, 'Random': obj_random_arr})
        df_fair = pd.DataFrame({'Opt': fair_opt_arr, 'Nearest': fair_near_arr, 'Worst': fair_worst_arr, 'Random': fair_random_arr})
        df_obj_avg += df_obj
        df_fair_avg += df_fair
    df_obj_avg /= n_repeats
    df_fair_avg /= n_repeats

    df_obj_avg.to_excel('rounding_objective_ton.xlsx')
    df_fair_avg.to_excel('rounding_fairness_ton.xlsx')

    df_obj_avg.plot()
    df_fair_avg.plot()
    plt.show()

def load_and_plot():
    fontsize = 20
    df_obj_avg = pd.read_excel('rounding_objective_ton.xlsx', index_col=0)
    df_fair_avg = pd.read_excel('rounding_fairness_ton.xlsx', index_col=0)
    df_obj_avg.columns = ['ORA', 'Nearest', 'LBF', 'Random']
    df_fair_avg.columns = ['ORA', 'Nearest', 'LBF', 'Random']
    G = np.arange(df_obj_avg.shape[0])
    plt.rcParams.update({'font.size': fontsize})
    percent_formatter = ticker.FuncFormatter(lambda x, pos: f"{100 * x:.0f}%")


    ax = df_obj_avg.plot(legend=True, lw=2, xlabel='No. of UEs', ylabel='Relative integer gap',
                          fontsize=fontsize, style=["r-s", "m-.d", "c-->", "k:x"], markersize=10, grid=True,
                          markerfacecolor='none')
    ax.set_xticks(G)
    #plt.ylim((0.295, 0.5))
    ax.set_xticklabels([str((i + 2) * 10) for i in G], fontsize=fontsize)
    plt.gca().yaxis.set_major_formatter(percent_formatter)

    ax = df_fair_avg.plot(legend=True, lw=2, xlabel='No. of UEs', ylabel='Fairness',
                         fontsize=fontsize, style=["r-s", "m-.d", "c-->", "k:x"], markersize=10, grid=True,
                         markerfacecolor='none')
    ax.set_xticks(G)
    #plt.ylim((0.795, 1.0))
    ax.set_xticklabels([str((i + 2) * 10) for i in G], fontsize=fontsize)

    plt.legend()



    plt.show()
if __name__ == '__main__':
    # run(n_repeats=2000)
    load_and_plot()