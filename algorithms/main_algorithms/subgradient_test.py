import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from algorithms.main_algorithms.projection import proj_b


# 定义二元函数
def f(p):
    z1, z2, z3 = np.log(1 + p[0]), 2*np.log(1 + 4*p[1]), 3*np.log(1+2*p[2])
    res = min(z1, z2, z3)
    return res

def grad_f(p):
    eps = 1e-16
    g_tmp = np.array([1/(1+p[0]), 8/(1+4*p[1]), 6/(1+2*p[2])])
    g = np.zeros(len(p))
    f_val = f(p)
    z = np.array([np.log(1 + p[0]), 2 * np.log(1 + 4 * p[1]), 3 * np.log(1 + 2 * p[2])])
    for i in range(len(p)):
        if np.abs(f_val - z[i]) < eps:
            g[i] = g_tmp[i]
    return g
def test_optimize_f():
    n, max_iter = 0, 1000
    eps = 1e-4
    p_max = 12
    dim = 3

    p_old, p = np.ones(dim)*p_max/2, np.zeros(dim)
    val_old, val_p = -1e8, f(p)
    p_opt, f_opt = np.zeros(dim), f(p)
    while (n < max_iter) and (np.abs(val_old - val_p) > eps):
        val_old = f(p)
        g = grad_f(p)
        p_old = p
        p = p + g*0.1
        #  projection
        p = proj_b(p/p_max)*p_max
        val_p = f(p)
        if f_opt < val_p:
            f_opt = val_p
            p_opt = p
        print(f"n = {n},p_old = {p_old}, p = {p}, g_p = {g}, f_old = {val_old:.4f}, f_p = {val_p:.4f}, f_opt = {f_opt:.4f}")
        n += 1
    print(f"p_opt = {p_opt}, f_opt = {f_opt}")
    return p_opt, f_opt



if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2, linewidth=50)
    p, val_p = test_optimize_f()
    print(p)
    print(val_p)