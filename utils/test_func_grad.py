from algorithms.main_algorithms.func_val_gradient import func_val_h, grad_h
from algorithms.main_algorithms.projection import proj_b
from scenarios.scenario_creators import load_scenario, create_scenario
import matplotlib.pyplot as plt
import numpy as np

# def test_func_h_plot(sc):
#
#     #sc = load_scenario('problem.pickle')
#     #sc = create_scenario(np.random.randint(2, 10), np.random.randint(1, 100))
#     # 创建一个包含0到100之间值的数组
#     x = np.linspace(25, 400, 1000)  # 生成400个点，以获得平滑的曲线
#     y = np.zeros(len(x))
#     # 计算这些点的函数值
#     for i in range(len(x)):
#         y[i], _, _ = func_val_h(sc, x[i])
#
#     # 绘制函数图像
#     plt.figure(figsize=(10, 5))  # 设置图像大小
#     plt.plot(x, y, label='obj')  # 绘制函数曲线
#
#     # 添加图例
#     plt.legend()
#
#     # 添加标题和轴标签
#     plt.title('Plot of f(x) = x^2 + 2x')
#     plt.xlabel('x')
#     plt.ylabel('f(x)')
#
#     # 显示网格
#     plt.grid(True)
#
#     # 显示图像
#     plt.show()
#
# def test_grad_h_plot():
#     sc = load_scenario('problem.pickle')
#     # sc = create_scenario(np.random.randint(2, 10), np.random.randint(1, 100))
#     # 创建一个包含0到100之间值的数组
#     x = np.linspace(25, 400, 1000)  # 生成400个点，以获得平滑的曲线
#     y = np.zeros(len(x))
#     # 计算这些点的函数值
#     for i in range(len(x)):
#         y[i], _, _ = grad_h(sc, x[i])
#
#     # 绘制函数图像
#     plt.figure(figsize=(10, 5))  # 设置图像大小
#     plt.plot(x, y, label='obj')  # 绘制函数曲线
#
#     # 添加图例
#     plt.legend()
#
#     # 添加标题和轴标签
#     plt.title('Plot of f(x) = x^2 + 2x')
#     plt.xlabel('x')
#     plt.ylabel('f(x)')
#
#     # 显示网格
#     plt.grid(True)
#
#     # 显示图像
#     plt.show()

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 定义二元函数
def f(p):
    z1 = np.log(1 + p[0])
    z2 = 2*np.log(1 + 4*p[1])
    z3 = 3*np.log(1+2*p[2])
    res = min(z1, z2, z3)
    return res

def grad_f(p):
    eps = 1e-8
    g_tmp = np.array([1/(1+p[0]), 8/(1+4*p[1]), 6/(1+2*p[2])])
    g = np.zeros(len(p))
    f_val = f(p)
    z = np.array([np.log(1 + p[0]), 2 * np.log(1 + 4 * p[1]), 3 * np.log(1 + 2 * p[2])])
    for i in range(len(p)):
        if np.abs(f_val - z[i]) < eps:
            g[i] = g_tmp[i]
    return g


def plot_f():
    # 创建 x 和 y 的网格
    x = np.linspace(0, 12, 10)
    y = np.linspace(0, 12, 10)
    X, Y = np.meshgrid(x, y)

    # 计算函数值 Z
    grad_X, grad_Y = np.zeros_like(X), np.zeros_like(Y)
    Z = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X[i])):
            Z[i, j] = f(X[i, j], Y[i, j])
            grad_tmp = grad_f(X[i, j], Y[i, j])
            grad_X[i, j] = grad_tmp[0]
            grad_Y[i, j] = grad_tmp[1]
    grad_Z = np.zeros_like(Z)
    # # 计算梯度
    # grad_X, grad_Y = gradient(X, Y)
    # grad_Z = np.zeros_like(X)  # 对于 f(x, y) = x + y^2，对 z 的偏导数是 0

    # 创建 3D 图表
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 3D 曲面图
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolor='none')

    # 添加色标
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # 绘制梯度箭头
    ax.quiver(X, Y, Z, grad_X, grad_Y, grad_Z, color='r', arrow_length_ratio=0.1)

    # 设置图表标题和坐标轴标签
    ax.set_title('3D plot of f(x, y) = x + y^2 with gradient arrows')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # 显示图表
    plt.show()

