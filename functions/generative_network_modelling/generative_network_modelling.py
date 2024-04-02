import numpy as np
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import bct
from functions.generative_network_modelling import bct_gpu
# import bct_gpu
import torch

def ks_statistic(x1, x2):
    combined = np.concatenate((x1, x2))
    bin_edges = np.concatenate(([-np.inf], np.sort(combined), [np.inf]))
    bin_counts1, _ = np.histogram(x1, bins=bin_edges)
    bin_counts2, _ = np.histogram(x2, bins=bin_edges)
    
    sum_counts1 = np.cumsum(bin_counts1) / np.sum(bin_counts1)
    sum_counts2 = np.cumsum(bin_counts2) / np.sum(bin_counts2)
    
    sample_cdf1 = sum_counts1[:-1]
    sample_cdf2 = sum_counts2[:-1]
    
    delta_cdf = np.abs(sample_cdf1 - sample_cdf2)
    
    k_statistic = np.max(delta_cdf)
    
    return k_statistic


def ks_statistic_gpu(x1, x2, device):
    # 将 x1 和 x2 转换为 PyTorch 张量并移至 GPU
    x1 = torch.tensor(x1, device=device)
    x2 = torch.tensor(x2, device=device)

    # 排序x1和x2
    sorted_x1, _ = torch.sort(x1)
    sorted_x2, _ = torch.sort(x2)
    
    # 获取所有唯一的值
    unique_values = torch.unique(torch.cat((sorted_x1, sorted_x2)))
    
    # 计算每个数组在这些点的CDF
    cdf_x1 = torch.searchsorted(sorted_x1, unique_values, right=True) / x1.size(0)
    cdf_x2 = torch.searchsorted(sorted_x2, unique_values, right=True) / x2.size(0)
    
    # 计算CDF之间的最大差异
    max_diff = torch.max(torch.abs(cdf_x1 - cdf_x2))
    
    return max_diff.item()



def Gen_one_connection(A, params, modelvar, device, D=None, use_matching=False, Fc=None, undirected=True):
    eta, gam, epsilon = params
    if use_matching:
        Kseed, _, _ = bct_gpu.matching_ind_gpu(A, device=device)
        # Kseed, _, _ = bct.matching_ind(A)
        Kseed = Kseed + epsilon  # add the epsilon

    n = len(A)  # take the nnode
    mv1 = modelvar[0]  # take if power law or exponential
    mv2 = modelvar[1]

    Fd = np.ones_like(A) / A.size
    Fk = np.ones_like(A) / A.size

    # compute the parameterized costs and values for wiring
    if D is not None:
        if mv1 == 'powerlaw':
            Fd = (D + epsilon )**eta
        elif mv1 == 'exponential':
            Fd = np.exp(eta * D)
    if use_matching:
        if mv2 == 'powerlaw':
            Fk = Kseed**gam
        elif mv2 == 'exponential':
            Fk = np.exp(gam * Kseed)
    
    # compute the initial wiring probability
    Ff = Fd * Fk * ~A.astype(bool)  # for non-extant edges
    if Fc is not None:
        Ff = Ff * (Fc + epsilon)
    
    if undirected:
        u_indx, v_indx = np.where(np.triu(np.ones((n, n)), k=1))  # compute indices
    else:
        u_indx, v_indx = np.where(np.ones((n, n)))  # compute indices
    
    indx = u_indx  * n + v_indx
    P = Ff.flatten()[indx]  # get the probability vector

    # add connection
    C = np.concatenate([np.array([0]), np.cumsum(P)])
    rand_value = np.random.rand()
    r = np.sum(rand_value * C[-1] >= C) - 1
    uu = u_indx[r]
    vv = v_indx[r]
    if undirected:
        A[uu, vv] = 1
        A[vv, uu] = 1
    else:
        A[uu, vv] = 1
    return A

# conn_matrices是connection_num个依次生成的conn_matrix
# y_target是根据person_num个brain_structure_matrix生成的graph statistic的list，形状为[4，person_num]
# (这些包括graph statistic包括了degrees_und、clustering_coef_bu、betweenness_bin、edge length)

def get_ks_list(conn_matrices, y_target, D, device, ifprint=False):
    person_num = len(y_target[0])
    connection_num = conn_matrices.shape[0]

    ks_list = []
    energy_list = []

    for i in range(connection_num):

        y_hat = [0 for _ in range(4)]
        y_hat[0] = bct.degrees_und(conn_matrices[i])

        y_hat[1] = bct_gpu.clustering_coef_bu_gpu(conn_matrices[i], device=device)
        y_hat[2] = bct_gpu.betweenness_bin_gpu(conn_matrices[i], device=device)

        y_hat[3] = D[np.triu(conn_matrices[i], k=1) > 0]

        ks = []

        for j in range(4):
            ks_j = [ks_statistic_gpu(x1=y_hat[j], x2=y_target[j][p], device=device) for p in range(person_num)]
            ks.append(np.array(ks_j).mean())
            # ks.append(np.array(ks_j))
        # ks = np.array(ks)
        
        energy_list.append(np.array(ks).max())
        # energy_list.append(ks.max(0).mean())
        ks_list.append(ks)
        if ifprint:
            print(f'{i}, energy:{energy_list[-1]:.4f}, ks_list:[{", ".join([f"{x:.4f}" for x in ks])}]')
            
    return ks_list, energy_list



# 计算节点大小的函数

def calculate_node_sizes(G, min_size=5, max_size=15):
    degrees = np.array([degree for _, degree in G.degree()])
    # 归一化度数到指定的大小范围
    return min_size + (max_size - min_size) * (degrees - degrees.min()) / (degrees.max() - degrees.min())



# 绘制brain的M条connection的一条一条生成的动画过程
# (conn_matrices是N*N*M的形状， coordinates是N*3的形状)

def plot_conn_generation(conn_matrices, coordinates, html_dir='result.html'):
    m = conn_matrices.shape[2]
    # 为了创建动画，我们需要先收集所有帧的数据
    frames = []

    for i in range(m):
        x = conn_matrices[:, :, i]
        G = nx.Graph(x)

        # 计算节点大小
        node_sizes = calculate_node_sizes(G)

        edge_x = []
        edge_y = []
        edge_z = []
        for edge in G.edges():
            x0, y0, z0 = coordinates[edge[0]]
            x1, y1, z1 = coordinates[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        node_x = []
        node_y = []
        node_z = []
        for node in G.nodes():
            x, y, z = coordinates[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

        trace_edges = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='grey', width=2))

        trace_nodes = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers', marker=dict(size=node_sizes, color='skyblue', 
                opacity=0.8,  # 增加透明度以增强立体感
                line=dict(color='black', width=0.5)  # 给节点添加黑色边框
            )
        )

        frames.append(go.Frame(data=[trace_edges, trace_nodes], name=str(i)))

    # 创建初始的图形布局
    fig = go.Figure(data=[frames[0]['data'][0], frames[0]['data'][1]],
                    layout=go.Layout(
                        scene=dict(xaxis=dict(showbackground=False),
                                    yaxis=dict(showbackground=False),
                                    zaxis=dict(showbackground=False)),
                        updatemenus=[dict(type='buttons', showactive=False,
                                        y=1,
                                        x=0.8,
                                        xanchor='left',
                                        yanchor='bottom',
                                        pad=dict(t=45, r=10),
                                        buttons=[dict(label='Play',
                                                        method='animate',
                                                        args=[None, dict(frame=dict(duration=200, redraw=True), 
                                                                        fromcurrent=True)])])],
                        sliders=[dict(steps=[dict(method='animate', args=[[f.name], 
                                                                        dict(mode='immediate',
                                                                                frame=dict(duration=200, redraw=True))],
                                                    label=f.name) for f in frames],
                                    transition=dict(duration=0),
                                    x=0, y=0, currentvalue=dict(visible=True, xanchor='right'))]))

    fig.frames = frames
    # fig.show()
    fig.write_html(html_dir)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # for _ in range(1000):
    #     x1 = np.random.randint(1000, size=(2000))
    #     x2 = np.random.randint(1000, size=(2000))
        
    #     value_0 = ks_statistic_gpu(x1, x2, device=device)
    #     value_1 = ks_statistic(x1, x2)
        

    #     if np.abs(value_0-value_1) > 1e-4:
    #         import pdb
    #         pdb.set_trace()
    #         break
    #     print(np.abs(value_0-value_1))
    # exit()
    
    
    import time

    # 记录函数开始时间
    start_time = time.time()

    for _ in range(10000):
        x1 = np.random.randint(1000, size=(2000))
        x2 = np.random.randint(1000, size=(2000))

        value_0 = ks_statistic_gpu(x1, x2, device=device)

        # value_1 = ks_statistic(x1, x2)


    # 记录函数开始时间
    end_time  = time.time()
    execution_time = end_time - start_time

    print(f"函数运行时间: {execution_time} 秒")


    # 记录函数开始时间
    start_time = time.time()

    for _ in range(10000):
        x1 = np.random.randint(1000, size=(2000))
        x2 = np.random.randint(1000, size=(2000))

        # value_0 = ks_statistic_gpu(x1, x2, device=device)

        value_1 = ks_statistic(x1, x2)


    # 记录函数开始时间
    end_time  = time.time()
    execution_time = end_time - start_time

    print(f"函数运行时间: {execution_time} 秒")