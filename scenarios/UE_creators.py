import datetime
import math
import random
import matplotlib.pyplot as plt
import pickle

import numpy as np

from network_classes.ground_user import UE
from network_classes.network_slice import Slice


def generate_random_circular_coordinates(R, num_points):
    """
    生成随机分布在半径为R的圆形区域内的二维坐标。

    参数:
    R (float): 圆的半径。
    num_points (int): 要生成的坐标点的数量。

    返回:
    numpy.ndarray: 形状为(num_points, 2)的二维数组，包含生成的坐标点。
    """
    # 生成随机半径和角度
    r = R * np.random.rand(num_points)
    theta = 2 * math.pi * np.random.rand(num_points)

    # 将极坐标转换为直角坐标
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # 返回坐标点的二维数组
    return np.column_stack((x, y))


def create_UE_set(num, network_radius, tilde_R_l, tilde_R_u):
    # generate random positions of the UEs
    # tilde_R: the requested rate range of each UE.
    locations = generate_random_circular_coordinates(network_radius, num)
    UE_set = []
    for i in range(num):
        ue_tmp = UE(locations[i][0], locations[i][1], random.randint(int(tilde_R_l), int(tilde_R_u)))
        UE_set.append(ue_tmp)
    return UE_set


def save_UE(UE_set):
    # 获取当前日期和时间
    now = datetime.datetime.now()

    # 格式化日期和时间，例如：YYYY-MM-DD_HH-MM-SS
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # 定义文件的基本名称和扩展名
    base_filename = "UE_set"
    extension = ".pkl"

    # 组合完整的文件名
    filename = f"{base_filename}_{timestamp}{extension}"
    with open(filename, 'wb') as f:
        pickle.dump(UE_set, f)


def load_UE(filename):
    # 打开包含序列化对象的文件
    with open(filename, 'rb') as f:
        # 使用pickle.load()方法从文件中加载对象
        UE_set = pickle.load(f)
    return UE_set


def plot_UE(UE_set, network_radius):
    circle = plt.Circle((0, 0), network_radius, fill=False)
    fig, ax = plt.subplots()
    ax.add_patch(circle)
    plt.xlim(-network_radius - 1, network_radius + 1)
    plt.ylim(-network_radius - 1, network_radius + 1)
    plt.scatter([p.loc_x for p in UE_set], [p.loc_y for p in UE_set], color='red')

    plt.axis('equal')
    plt.xlabel('x (meter)')
    plt.ylabel('y (meter)')
    plt.title('User distribution')
    plt.show()


if __name__ == '__main__':
    network_radius = 4
    UE_set = create_UE_set(100, network_radius, 1, 2)
    save_UE(UE_set)
    plot_UE(UE_set, network_radius)