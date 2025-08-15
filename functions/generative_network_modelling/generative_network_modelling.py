import numpy as np
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