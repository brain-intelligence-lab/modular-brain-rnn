import enum
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
    # x1, x2均是一维向量
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



def ks_statistic_batch_gpu(x1, x2, device):
    """
    KS 统计量的 Batch 并行版本 (GPU)
    
    参数:
    x1: Tensor, 形状 (B, N) 或者list
    x2: Tensor, 形状 (B, M)
    
    返回:
    max_diff: Tensor, 形状 (B,) 每个样本的 KS 统计量
    """
    # 确保在同一设备上
    if isinstance(x1, list) or isinstance(x2, list):
        if isinstance(x2, list):
            x1, x2 = x2, x1

        batch_size = x2.shape[0]
        max_diff = torch.zeros((batch_size,), device=device)
        
        if isinstance(x1, list):
            for idx, x1_i in enumerate(x1):
                max_diff[idx] = ks_statistic_gpu(x1_i, x2[idx], device)
            
        return max_diff
    
    B, N = x1.shape
    _, M = x2.shape

    # 1. 对 x1 和 x2 分别进行行内排序
    sorted_x1, _ = torch.sort(x1, dim=-1)
    sorted_x2, _ = torch.sort(x2, dim=-1)

    # 2. 构造评估点 (Concatenate x1 and x2)
    # 在计算 KS 时，最大差异一定发生在 x1 或 x2 的样本点上
    # 形状为 (B, N + M)
    combined = torch.cat([sorted_x1, sorted_x2], dim=-1)
    # 这里的排序是为了配合 searchsorted 使用，形状 (B, N + M)
    evaluation_points, _ = torch.sort(combined, dim=-1)

    # 3. 计算 Batch CDF
    # searchsorted 支持 batch 模式: 
    # sorted_sequence (B, N), input (B, N+M) -> output (B, N+M)
    cdf_x1 = torch.searchsorted(sorted_x1, evaluation_points, right=True).float() / N
    cdf_x2 = torch.searchsorted(sorted_x2, evaluation_points, right=True).float() / M

    # 4. 计算最大绝对差异
    # 在 dim=-1 (N+M 维度) 上取最大值
    max_diff = torch.max(torch.abs(cdf_x1 - cdf_x2), dim=-1)[0]

    return max_diff


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



def Sample_conn_matrix(prob_matrix, tot_conn_num, directed=False):
    n = prob_matrix.shape[0]
    if directed:
        # 提取非对角线部分的索引
        u_indx, v_indx = np.where(~np.eye(n, dtype=bool))
    else:
        # 提取上三角部分的索引 (不包含自环 k=1，若包含自环则 k=0)
        u_indx, v_indx = np.triu_indices(n, k=1)
    
    # 获取索引对应的概率并归一化
    probs = prob_matrix[u_indx, v_indx]
    probs /= probs.sum() # 重新归一化，确保概率和为 1
    
    # 这里的 choice 选出的是 u_indx 数组中的下标
    if directed: 
        sampled_indices = np.random.choice(len(u_indx), \
            size=tot_conn_num, replace=False, p=probs)
    else:
        sampled_indices = np.random.choice(len(u_indx), \
            size=int(tot_conn_num//2), replace=False, p=probs)
    
    # 映射回原矩阵坐标
    final_u = u_indx[sampled_indices]
    final_v = v_indx[sampled_indices]
    
    # 构建结果矩阵
    result_matrix = np.zeros((n, n), dtype=int)
    result_matrix[final_u, final_v] = 1
    if not directed:
        result_matrix[final_v, final_u] = 1 # 无向图保证对称性
        
    assert result_matrix.sum() == tot_conn_num
    return result_matrix

def gnm(Distance, Kseed, eta, gamma, tot_conn_num, directed=False):
    Fd = ( Distance + 1e-5 ) ** eta
    Fk = ( Kseed + 1e-5 ) ** gamma
    prob_matrix = Fd * Fk
    prob_matrix = prob_matrix / prob_matrix.sum()
    return Sample_conn_matrix(prob_matrix, tot_conn_num, directed=directed)


def gnm_batched(prob_matrix_batch, tot_conn_num, directed=False, device='cuda:0'):
    """
    高度优化的 GNM 批量采样器：
    1. 全程 GPU 计算，无 CPU 传输。
    2. 支持超大规模 Batch 采样 (一次生成 batch_size 个图，batch_size = eta_vec * gamma_vec * n_sims)。
    3. 使用 torch.multinomial 替代极其缓慢的 np.random.choice。
    
    参数:
        prob_matrix_batch: [batch_size, n, n] 的概率矩阵 tensor，已经在 device 上
        tot_conn_num: 每个图的总连接数
        directed: 是否为有向图
        device: 设备 ('cuda:0' 等)
    
    返回:
        adj_matrices: [batch_size, n, n] 的邻接矩阵 tensor
    """
    
    # 1. 确保输入是 Tensor 且在正确的设备上
    prob_matrix_batch = torch.as_tensor(prob_matrix_batch, device=device, dtype=torch.float32)
    batch_size, n, _ = prob_matrix_batch.shape
    
    # 2. 提取有效连接的概率 (Masking)
    if directed:
        # 移除对角线 - 使用与重建索引相同的方法
        mask = ~torch.eye(n, dtype=torch.bool, device=device)  # [n, n]
        u_all, v_all = torch.where(mask)  # 获取非对角线元素的坐标
        # 对每个 batch 提取有效概率: [batch_size, n*(n-1)]
        valid_probs = prob_matrix_batch[:, u_all, v_all]  # [batch_size, n*(n-1)]
        num_edges_to_sample = tot_conn_num
    else:
        # 仅取上三角 (k=1)
        u_idx, v_idx = torch.triu_indices(n, n, offset=1, device=device)
        # 对每个 batch 提取上三角: [batch_size, n*(n-1)/2]
        valid_probs = prob_matrix_batch[:, u_idx, v_idx]  # [batch_size, n*(n-1)/2]
        num_edges_to_sample = int(tot_conn_num // 2)

    # 3. 归一化 (对每个 batch 分别归一化)
    # [batch_size, num_candidates] -> 对 dim=1 归一化
    valid_probs = valid_probs / valid_probs.sum(dim=1, keepdim=True)

    # 4. 批量采样 (核心加速点)
    # torch.multinomial 在 GPU 上对加权无放回采样极快
    # sampled_indices shape: [batch_size, num_edges_to_sample]
    sampled_flat_indices = torch.multinomial(valid_probs, num_edges_to_sample, replacement=False)

    # 5. 重建邻接矩阵
    # 创建全 0 容器: [batch_size, n, n]
    adj_matrices = torch.zeros((batch_size, n, n), dtype=torch.int8, device=device)

    if directed:
        # 映射回二维坐标
        # 使用与提取 valid_probs 时相同的索引方法
        mask = ~torch.eye(n, dtype=torch.bool, device=device)  # [n, n]
        u_all, v_all = torch.where(mask)  # 获取非对角线元素的坐标
        valid_linear_indices = (u_all * n + v_all).long()  # [n*(n-1)]
        
        # 获取采样到的真实线性索引
        # sampled_flat_indices 是 [batch_size, num_edges] 的 long tensor
        # valid_linear_indices[sampled_flat_indices] -> shape [batch_size, num_edges]
        real_flat_indices = valid_linear_indices[sampled_flat_indices]  # [batch_size, num_edges]
        
        # 填入 1，scatter_ 需要 long 类型的索引
        flat_adj = torch.zeros((batch_size, n * n), dtype=torch.int8, device=device)
        flat_adj.scatter_(1, real_flat_indices, 1)
        adj_matrices = flat_adj.view(batch_size, n, n)
        
    else:
        # 无向图处理
        # 1. 准备线性索引
        # u_idx, v_idx 已经是上三角坐标
        # 将二维坐标转为线性坐标: idx = u * n + v
        # 确保数据类型为 long
        flat_indices_map = (u_idx * n + v_idx).long()  # [num_candidates]
        
        # 2. 获取 batch 中选中的真实线性索引
        batch_real_indices = flat_indices_map[sampled_flat_indices]  # [batch_size, num_edges]

        # 3. 填入上三角
        flat_adj = torch.zeros((batch_size, n * n), dtype=torch.int8, device=device)
        flat_adj.scatter_(1, batch_real_indices.long(), 1)
        adj_upper = flat_adj.view(batch_size, n, n)
        
        # 4. 对称化: A + A.T
        adj_matrices = adj_upper + adj_upper.transpose(1, 2)

    return adj_matrices.float()  # 返回 float 以便后续计算统计量


def get_graph_distribution(conn_matrix, Distance, device, to_tensor=False):
    """
    计算单个图的分布统计量（保持向后兼容）
    """
    target = []

    target.append(bct.degrees_und(conn_matrix))
    target.append(bct_gpu.clustering_coef_bu_gpu(conn_matrix, device, to_tensor))
    target.append(bct_gpu.betweenness_bin_gpu(conn_matrix, device, to_tensor))
    target.append(Distance[conn_matrix > 0])
    
    if to_tensor:
        for i in range(4):
            target[i] = torch.tensor(target[i], dtype=torch.float32, device=device)  

    return target


def get_graph_distribution_batched(conn_matrix_batch, Distance, device, directed=False):
    """
    批量计算图的分布统计量
    
    参数:
        conn_matrix_batch: [batch_size, n, n] 的邻接矩阵 tensor，已经在 device 上
        Distance: [n, n] 的距离矩阵（numpy array 或 tensor）
        device: 设备
        to_tensor: 是否返回 tensor（默认 True）
        directed: 是否为有向图（默认 False）
    
    返回:
        target: list of tensors
            - degrees: [batch_size, n]
            - clustering: [batch_size, n]
            - betweenness: [batch_size, n]
            - edge_lengths: list of arrays/tensors，每个元素是 [num_edges_i] 的数组
    """
    conn_matrix_batch = torch.as_tensor(conn_matrix_batch, dtype=torch.float32, device=device)
    Distance = torch.as_tensor(Distance, dtype=torch.float32, device=device)
    
    # 所有函数都先返回 tensor，最后统一处理
    degrees = bct_gpu.degrees_und_batched(conn_matrix_batch, device)
    clustering = bct_gpu.clustering_coef_bu_gpu_batched(conn_matrix_batch, device)
    betweenness = bct_gpu.betweenness_bin_gpu_batched(conn_matrix_batch, device)
    edge_lengths_list = bct_gpu.edge_lengths_batched(conn_matrix_batch, Distance, device, directed=directed)
    
    target = [degrees, clustering, betweenness, edge_lengths_list]
    return target


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