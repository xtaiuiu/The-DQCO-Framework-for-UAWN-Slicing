import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scenarios.scenario_creators import load_scenario


def func_val(sc):
    """
    Compute the function value at current settings of sc
    :param sc: Scenario object
    :return: function value
    """
    fval, _, _ = func_val_h(sc, sc.uav.h)
    return fval

# compute the function value at x
def func_val_x(sc, x):
    """
    :param sc: the scenario
    :param x: a list or numpy array
    :return func_val, i_min, k_min:
    func_val: the function value at x for fixed p and h
    i_min: the slice index at which the minimum is achieved
    k_min: the UE index at which the minimum is achieved
    """
    func_val = 1e6
    i_min, k_min = 0, 0
    idx = 0
    theta_h = max(sc.pn.theta_min ** 2, (math.atan(sc.pn.radius / (math.sqrt(sc.uav.h)))) ** 2)
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(sc.slices[i].UEs)):
            u = sc.slices[i].UEs[k]
            f_i_k = x[idx] * s.b_width * np.log(1 + sc.pn.g_0 * u.tilde_g * u.p / (
                        sc.pn.sigma * theta_h * (u.loc_x ** 2 + u.loc_y ** 2 + sc.uav.h) ** (
                            sc.pn.alpha / 2))) / u.tilde_r
            if func_val > f_i_k:
                func_val = f_i_k
                i_min = i
                k_min = k
            idx += 1
    return func_val, i_min, k_min


# compute the function value at p
def func_val_p(sc, p):
    """
    :param sc:
    :param p: a list or numpy array
    :return:
    """
    func_val = 1e6
    i_min, k_min = 0, 0
    idx = 0
    theta_h = max(sc.pn.theta_min ** 2, (math.atan(sc.pn.radius / (math.sqrt(sc.uav.h)))) ** 2)
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(sc.slices[i].UEs)):
            u = sc.slices[i].UEs[k]
            C = u.x * s.b_width / u.tilde_r
            D = sc.pn.g_0 * u.tilde_g / (
                        sc.pn.sigma * theta_h * (u.loc_x ** 2 + u.loc_y ** 2 + sc.uav.h) ** (sc.pn.alpha / 2))
            f_i_k = C * np.log(1 + D * p[idx])
            if func_val > f_i_k:
                func_val = f_i_k
                i_min = i
                k_min = k
            idx += 1
    return func_val, i_min, k_min


# compute the function value at h
def func_val_h(sc, h):
    func_val = 1e6
    i_min, k_min = 0, 0
    idx = 0
    theta_h = max(sc.pn.theta_min ** 2, (math.atan(sc.pn.radius / (math.sqrt(h)))) ** 2)
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(sc.slices[i].UEs)):
            u = sc.slices[i].UEs[k]
            C = u.x * s.b_width / u.tilde_r
            E = sc.pn.g_0 * u.tilde_g * u.p / sc.pn.sigma
            f_i_k = C * np.log(1 + E / (theta_h * ((u.loc_x ** 2 + u.loc_y ** 2 + h) ** (sc.pn.alpha / 2))))
            if func_val > f_i_k:
                func_val = f_i_k
                i_min = i
                k_min = k
            idx += 1
    return func_val, i_min, k_min


def func_val_h_wrapper(h, sc):
    res, _, _ = func_val_h(sc, h)
    return -res


# compute the gradient at p
def grad_p(sc, p):
    """
    :param sc:
    :param p: a list or numpy array
    :return: grad, i_min, k_min
    grad: a numpy array that represents the gradient of f at p
    """
    func_val = 1e6
    grad_tmp = -1e8
    i_min, k_min = 0, 0
    idx = 0
    idx_min = -1
    theta_h = max(sc.pn.theta_min ** 2, (math.atan(sc.pn.radius / (math.sqrt(sc.uav.h)))) ** 2)
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(sc.slices[i].UEs)):
            u = sc.slices[i].UEs[k]
            C = u.x * s.b_width / u.tilde_r
            D = sc.pn.g_0 * u.tilde_g / (
                        sc.pn.sigma * theta_h * (u.loc_x ** 2 + u.loc_y ** 2 + sc.uav.h) ** (sc.pn.alpha / 2))
            f_i_k = C * np.log(1 + D * p[idx])
            if func_val > f_i_k:
                func_val = f_i_k
                i_min = i
                k_min = k
                grad_tmp = C * D / (1 + D * p[idx])
                idx_min = idx
            idx += 1
    grad = np.zeros(len(p))
    grad[idx_min] = grad_tmp
    return grad, i_min, k_min


# compute the gradient at h
def grad_h(sc, h):
    h_0 = (sc.pn.radius / np.tan(sc.pn.theta_min)) ** 2
    s_idx, u_idx, min_val = 0, 0, 1e8
    if h <= h_0:
        # min_s_idx and min_u_idx are the indices of slice and UEs such that the inner minimization reaches it minimum
        theta_h = (math.atan(sc.pn.radius / (math.sqrt(h)))) ** 2
        for i in range(len(sc.slices)):
            s = sc.slices[i]
            for k in range(len(sc.slices[i].UEs)):
                u = sc.slices[i].UEs[k]
                C = u.x * s.b_width / u.tilde_r
                E = sc.pn.g_0 * u.tilde_g * u.p / sc.pn.sigma
                f_i_k = C * np.log(1 + E / (theta_h * ((u.loc_x ** 2 + u.loc_y ** 2 + h) ** (sc.pn.alpha / 2))))
                if f_i_k < min_val:
                    s_idx, u_idx, min_val = i, k, f_i_k
        s, u = sc.slices[s_idx], sc.slices[s_idx].UEs[u_idx]
        pn = sc.pn
        C = u.x * s.b_width / u.tilde_r
        E = pn.g_0 * u.tilde_g * u.p / pn.sigma
        den2 = h + u.loc_x ** 2 + u.loc_y ** 2
        grad = C * E / (2 * theta_h * (E + theta_h * den2 ** (pn.alpha / 2))) * (
                    pn.radius * math.sqrt(h) / (h ** 2 + h * pn.radius) - pn.alpha * theta_h / den2)
    else:
        theta_h = sc.pn.theta_min ** 2
        for i in range(len(sc.slices)):
            s = sc.slices[i]
            for k in range(len(sc.slices[i].UEs)):
                u = sc.slices[i].UEs[k]
                C = u.x * s.b_width / u.tilde_r
                E = sc.pn.g_0 * u.tilde_g * u.p / sc.pn.sigma
                f_i_k = C * np.log(1 + E / (theta_h * ((u.loc_x ** 2 + u.loc_y ** 2 + h) ** (sc.pn.alpha / 2))))
                if f_i_k < min_val:
                    s_idx, u_idx, min_val = i, k, f_i_k
        s, u = sc.slices[s_idx], sc.slices[s_idx].UEs[u_idx]
        pn = sc.pn
        C = u.x * s.b_width / u.tilde_r
        E = pn.g_0 * u.tilde_g * u.p / pn.sigma
        den2 = h + u.loc_x ** 2 + u.loc_y ** 2
        grad = -pn.alpha * C * E / (2 * den2 * ((pn.theta_min ** 2) * (den2 ** (pn.alpha / 2)) + E))
    return grad, s_idx, u_idx

def func_p_wrapper(sc, p1, p2):
    val, _, _ = func_val_p(sc, [p1, p2])
    return val

def grad_p_wapper(sc, p1, p2):
    grad, _, _ = grad_p(sc, [p1, p2])
    return grad

def func_test_p(sc):
    # 创建 x 和 y 的网格
    x = np.linspace(0, 2, 10)
    y = np.linspace(1, 2, 10)
    X, Y = np.meshgrid(x, y)
    # 计算 Z 轴上的函数值
    Z = np.zeros_like(X)
    grad_X, grad_Y = np.zeros_like(X), np.zeros_like(Y)
    for i in range(len(X)):
        for j in range(len(X[i])):
            Z[i, j] = func_p_wrapper(sc, X[i, j], Y[i, j])
            grad_tmp = grad_p_wapper(sc, X[i, j], Y[i, j])
            grad_X[i, j] = grad_tmp[0]
            grad_Y[i, j] = grad_tmp[1]
    grad_Z = np.zeros_like(Z)

    # 创建 3D 图表
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 3D 曲面图
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # 绘制梯度箭头
    ax.quiver(X, Y, Z, grad_X, grad_Y, grad_Z, color='r', arrow_length_ratio=0.1)
    # 添加色标
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # 设置图表标题和坐标轴标签
    ax.set_title('3D plot of f(x, y) = x + y^2')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # 显示图表
    plt.show()


if __name__ == '__main__':
    sc = load_scenario('scenario_2_UEs.pickle')
    func_test_p(sc)