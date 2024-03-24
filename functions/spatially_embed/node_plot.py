import numpy as np
import matplotlib.pyplot as plt

def node_plot(node_coordinate, modularity_cluster, group_num=3, cmap='tab10'):
    '''
    绘制3D节点图
    :param node_coordinate: 节点坐标 (N, 3)
    :param modularity_cluster: 节点所属社团 (N, )
    :param group_num: 绘制前group_num个社团
    :return:
    '''

    modularity_cluster = modularity_cluster.astype(np.int_)

    if group_num != 0:
        # 找到点数最多的group_num个社团
        group_nums = {}
        max_group_number = np.max(modularity_cluster)
        for i in range(max_group_number+1):
            group_nums[i] = np.sum(modularity_cluster==i)
        group_nums = sorted(group_nums.items(), key=lambda x: x[1], reverse=True)
        group_nums = group_nums[0:group_num]
        group_nums = [i[0] for i in group_nums]

        # 提取这些社团的节点坐标
        new_node_coordinate = []
        new_modularity_cluster = []
        N = node_coordinate.shape[0]
        for i in range(N):
            if modularity_cluster[i] in group_nums:
                new_node_coordinate.append(node_coordinate[i, :])
                new_modularity_cluster.append(modularity_cluster[i])
        node_coordinate = np.array(new_node_coordinate)
        modularity_cluster = np.array(new_modularity_cluster)

    # 绘制图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(node_coordinate[:, 0], node_coordinate[:, 1], node_coordinate[:, 2],
                c=modularity_cluster, cmap=cmap, s=50)
    plt.show()


if __name__ == '__main__':
    coordinate = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    cluster = np.array([1., 2., 3., 2.])
    node_plot(coordinate, cluster, 2)
