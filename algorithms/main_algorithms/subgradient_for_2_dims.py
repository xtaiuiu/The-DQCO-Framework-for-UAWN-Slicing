import numpy as np
from scipy.optimize import root_scalar
from algorithms.main_algorithms.projection import proj_b


def f(p):
    #z1, z2, z3 = np.log(1 + p[0]), 2*np.log(1 + 4*p[1]), 3*np.log(1+2*p[2])
    z = np.array([np.log(1 + p[0]), 2*np.log(1 + 4*p[1])])
    res = min(z)
    return res

def grad_f(p):
    eps = 1e-16
    g_tmp = np.array([1/(1+p[0]), 8/(1+4*p[1])])
    g = np.zeros(len(p))
    f_val = f(p)
    z = np.array([np.log(1 + p[0]), 2 * np.log(1 + 4 * p[1])])
    n = 0
    for i in range(len(p)):
        if np.abs(f_val - z[i]) < eps:
            g[i] = g_tmp[i]
            n += 1
    return g/n

def optimize_2d():
    n, max_iter = 0, 5000
    eps = 1e-6
    p_max = 1
    dim = 2

    p_old, p = np.ones(dim) * p_max / 2, np.zeros(dim)
    #p = np.array([11.3664, 0.6336])
    val_old, val_p = -1e8, f(p)
    p_opt, f_opt = np.zeros(dim), f(p)
    while (n < max_iter) and (np.abs(val_old - val_p) > eps):
        val_old = f(p)
        g = grad_f(p)
        p_old = p
        p = p + g * 0.2
        #  projection
        p = proj_b(p / p_max) * p_max
        val_p = f(p)
        if f_opt < val_p:
            f_opt = val_p
            p_opt = p
        print(
            f"n = {n:04d},p_old = {p_old}, p = {p}, g_p = {g}, f_old = {val_old:.4f}, f_p = {val_p:.4f}, f_opt = {f_opt:.4f}")
        n += 1
    print(f"p_opt = {p_opt}, f_opt = {f_opt}")
    return p_opt, f_opt


def f_root_finding(z):
    return np.exp(-z) + 0.25*np.exp(-z/2) - 9/4


def f_root_finding_prime(z):
    return -np.exp(-z) - np.exp(-z/2)/8


def optimize_root_finding():
    init_guess = 1.0
    z = root_scalar(f_root_finding, method='newton', x0=init_guess, fprime=f_root_finding_prime)
    z = z.root
    x = np.array([np.exp(-z)-1, (np.exp(-z/2)-1)/4])
    print(f"x = {x}, opt = {-z}")



if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    #print(grad_f([0, 1]))
    optimize_2d()
    optimize_root_finding()