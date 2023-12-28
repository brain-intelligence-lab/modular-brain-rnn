import numpy as np
import torch

def sphere_distance(n, r):
    '''
    创建一个球体，将N个点平均分布到球面上，计算每个点之间的距离
    :param n: int, 点的数量
    :param r: float, 球体半径
    :return D: torch.tensor, N*N, 每个点之间的距离
    '''
    # 1. 创建一个球体
    center = np.array([0, 0, 0])  # 球体中心坐标

    # 2. 将N个点平均分布到球面上
    phi = np.arccos(1 - 2 * np.arange(n) / (n - 1))  # 仰角
    theta = np.pi * (1 + 5 ** 0.5) * np.arange(n)  # 方位角

    x = center[0] + r * np.sin(phi) * np.cos(theta)  # x坐标
    y = center[1] + r * np.sin(phi) * np.sin(theta)  # y坐标
    z = center[2] + r * np.cos(phi)  # z坐标

    points = np.column_stack((x, y, z))  # 合并坐标

    D = np.sqrt(np.sum((points[:, np.newaxis] - points) ** 2, axis=2))

    return torch.from_numpy(D).type(torch.float), points

def cube_distance(n):
    '''
    创建一个立方体，将N个点平均分布到立方体内，计算每个点之间的距离
    :param n:  int, 点的数量
    :return:  D: torch.tensor, N*N, 每个点之间的距离
    '''

    # 1. 创建一个立方体
    # 计算立方体的边长，a, b, c 满足条件 a*b*c = N 且 a,b,c 接近 N^(1/3)，a,b,c为整数
    a = int(np.ceil(n ** (1 / 3)))
    b = int(np.ceil(n ** (1 / 3)))
    c = int(np.ceil(n / (a * b)))
    print(a,b,c)

    # 2. 将N个点平均分布到立方体内
    x = np.linspace(0, a - 1, a)
    y = np.linspace(0, b - 1, b)
    z = np.linspace(0, c - 1, c)

    x, y, z = np.meshgrid(x, y, z)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    points = np.column_stack((x, y, z))  # 合并坐标

    points = points[0:n, :]

    D = np.sqrt(np.sum((points[:, np.newaxis] - points) ** 2, axis=2))


    return torch.from_numpy(D).type(torch.float), points

if __name__ == '__main__':
    N = 10
    print(N**(1/3))
    D, M = sphere_distance(N, 1)
