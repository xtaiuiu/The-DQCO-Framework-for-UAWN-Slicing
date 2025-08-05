import timeit

import pandas as pd
import seaborn as sns
import numpy as np
import time
from matplotlib import pyplot as plt

from utils.projction import proj_generalized_simplex, proj_generalized_simplex_QP


def run():
    df_KKT = pd.DataFrame(columns=['No. of UEs', 'runs', 'CPU time (s)', 'algorithm'])
    n_UEs = np.arange(1, 11) * 100
    n_runs = 100
    for n_UE in n_UEs:
        for i in range(n_runs):
            a = np.random.randint(1, 10, n_UE)
            y = (0.1 + np.random.random(n_UE) * 0.9) * n_UE
            r = np.random.randint(5000, 10000)
            start_time1 = time.perf_counter()
            #testfunc()
            proj_generalized_simplex(a, y, r)
            runtime1 = time.perf_counter() - start_time1

            df_KKT.loc[len(df_KKT)] = [n_UE, i, runtime1*25000*6, 'KKT']

            start_time2 = time.perf_counter()
            #testfunc2()
            proj_generalized_simplex_QP(a, y, r)
            runtime2 = time.perf_counter() - start_time2
            df_KKT.loc[len(df_KKT)] = [n_UE, i, runtime2*25000*6, 'CVX']
    df_KKT.to_excel('kkt_runtime.xlsx')

def testfunc():
    time.sleep(0.05)
    return np.array([1, 2, 3])

def testfunc2():
    time.sleep(0.07)
    return np.array([1, 2, 3])

def load_and_plot():
    sns.set_theme(style="darkgrid")
    df_KKT = pd.read_excel('kkt_runtime.xlsx')
    sns.lineplot(data=df_KKT, x="No. of UEs", y="CPU time (s)", hue='algorithm')
    plt.xlabel('Number of UEs', fontsize=18)
    plt.ylabel('CPU time (s)', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.show()


if __name__ == '__main__':
    # run()
    load_and_plot()