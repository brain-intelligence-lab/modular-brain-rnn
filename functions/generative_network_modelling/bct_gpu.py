import numpy as np
import bct
import torch

def matching_ind_gpu(CIJ, device, only_min=False):
    CIJ = bct.utils.binarize(CIJ, copy=True)
    CIJ = torch.tensor(CIJ, dtype=torch.float, device=device)  # 转移到 GPU
    n = CIJ.shape[0]

    # 创建掩码以忽略自连接
    mask = 1 - torch.eye(n, device=device)
    mask = mask.bool()
    CIJ = CIJ * mask

    # 入度匹配
    in_deg = CIJ.sum(dim=0, keepdim=True)
    common_in = torch.mm(CIJ, CIJ.t())
    in_or_deg = (in_deg + in_deg.t())
    in_or_deg = (in_or_deg - CIJ * 2)
    Min = common_in * 2 / in_or_deg
    Min[in_or_deg==0] = 0
    Min = Min * mask
    if only_min:
        return Min

    # 出度匹配
    out_deg = CIJ.sum(dim=1, keepdim=True)
    common_out = torch.mm(CIJ.t(), CIJ) 
    out_or_deg = (out_deg + out_deg.t())
    out_or_deg = (out_or_deg - CIJ * 2)
    Mout = common_out * 2 / out_or_deg
    Mout[out_or_deg==0] = 0
    Mout = Mout * mask

    # 总体匹配
    Mall = (Min + Mout) / 2

    return Min.cpu().numpy(), Mout.cpu().numpy(), Mall.cpu().numpy()


def clustering_coef_bu_gpu(G, device, to_tensor=False):
    G = bct.utils.binarize(G, copy=True)
    G = torch.tensor(G, dtype=torch.float32, device=device)  # 将矩阵转移到 GPU
    n = G.shape[0]
    C = torch.zeros(n, device=device)  # 在 GPU 上初始化聚类系数向量

    # 计算所有可能的三角形组合
    G_square = torch.matmul(G, G)
    triangle_count = G_square * G  # 矩阵乘法后与原矩阵相乘得到三角形的两倍

    # 计算每个节点的度
    degrees = G.sum(dim=1)

    # 计算聚类系数
    possible_triangle_count = degrees * (degrees - 1)  # 每个节点可能的三角形数量
    C = triangle_count.sum(dim=1) / possible_triangle_count  # 聚类系数

    # 处理分母为0的情况
    C[possible_triangle_count == 0] = 0.0
    if to_tensor:
        return C
    return C.cpu().numpy()  # 将结果转回 CPU


def betweenness_bin_gpu(G, device, to_tensor=False):
    G = bct.utils.binarize(G, copy=True)
    G = torch.tensor(G, dtype=torch.float32, device=device)  # 将矩阵转移到 GPU
    n = len(G)  # number of nodes
    I = torch.eye(n, device=device)
    d = 1  # path length
    NPd = G.clone()  # number of paths of length |d|
    NSPd = G.clone()   # number of shortest paths of length |d|
    NSP = G.clone()  # number of shortest paths of any length
    L = G.clone()  # length of shortest paths

    NSP[I == 1] = 1
    L[I == 1] = 1

    # Calculate NSP and L
    while torch.any(NSPd):
        d += 1
        NPd = torch.mm(NPd, G)
        NSPd = NPd * (L == 0)
        NSP += NSPd
        L = L + d * (NSPd != 0).float()

    L[L == 0] = float('inf')  # L for disconnected vertices is inf
    L[I == 1] = 0
    NSP[NSP == 0] = 1  # NSP for disconnected vertices is 1

    DP = torch.zeros((n, n), device=device)  # vertex on vertex dependency
    diam = d - 1

    # Calculate DP
    for d in range(diam, 1, -1):
        DPd1 = torch.mm(((L == d) * (1 + DP) / NSP), G.t()) * \
            ((L == (d - 1)) * NSP)
        DP += DPd1
    result = torch.sum(DP, dim=0)
    if to_tensor:
        return result

    return result.cpu().numpy()


def degrees_und_batched(conn_matrix_batch, device):
    """
    批量计算无向图的度
    
    参数:
        conn_matrix_batch: [batch_size, n, n] 的邻接矩阵 tensor，已经在 device 上
        device: 设备
    
    返回:
        degrees_batch: [batch_size, n] 的度 tensor
    """
    conn_matrix_batch = torch.as_tensor(conn_matrix_batch, dtype=torch.float32, device=device)
    conn_matrix_batch = (conn_matrix_batch > 0).float()

    degrees_batch = conn_matrix_batch.sum(dim=2)  # [batch_size, n]
    return degrees_batch


def clustering_coef_bu_gpu_batched(G_batch, device):
    """
    批量计算聚类系数
    
    参数:
        G_batch: [batch_size, n, n] 的邻接矩阵 tensor，已经在 device 上
        device: 设备
    
    返回:
        C_batch: [batch_size, n] 的聚类系数 tensor
    """
    # 确保输入是 tensor 且在正确的设备上
    if isinstance(G_batch, np.ndarray):
        G_batch = torch.tensor(G_batch, dtype=torch.float32, device=device)
    else:
        G_batch = torch.as_tensor(G_batch, dtype=torch.float32, device=device)
    
    # 二值化
    G_batch = (G_batch > 0).float()
    
    batch_size, n, _ = G_batch.shape
    
    # 批量计算 G^2: [batch_size, n, n]
    G_square = torch.bmm(G_batch, G_batch)  # batch matrix multiplication
    triangle_count = G_square * G_batch  # [batch_size, n, n]
    
    # 计算每个节点的度: [batch_size, n]
    degrees = G_batch.sum(dim=2)  # [batch_size, n]
    
    # 计算聚类系数
    possible_triangle_count = degrees * (degrees - 1)  # [batch_size, n]
    C_batch = triangle_count.sum(dim=2) / possible_triangle_count  # [batch_size, n]
    
    # 处理分母为0的情况
    C_batch[possible_triangle_count == 0] = 0.0
    return C_batch

def betweenness_bin_gpu_batched(G_batch, device):
    """
    批量计算介数中心性
    
    参数:
        G_batch: [batch_size, n, n] 的邻接矩阵 tensor，已经在 device 上
        device: 设备
    
    返回:
        result_batch: [batch_size, n] 的介数中心性 tensor
    """
    # 确保输入是 tensor 且在正确的设备上
    if isinstance(G_batch, np.ndarray):
        G_batch = torch.tensor(G_batch, dtype=torch.float32, device=device)
    else:
        G_batch = torch.as_tensor(G_batch, dtype=torch.float32, device=device)
    
    # 二值化
    G_batch = (G_batch > 0).float()
    
    batch_size, n, _ = G_batch.shape
    I = torch.eye(n, device=device).unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n, n]
    
    d = 1  # path length
    NPd = G_batch.clone()  # [batch_size, n, n]
    NSPd = G_batch.clone()
    NSP = G_batch.clone()
    L = G_batch.clone()

    NSP[I == 1] = 1
    L[I == 1] = 1

    # Calculate NSP and L for all batches
    while torch.any(NSPd):
        d += 1
        NPd = torch.bmm(NPd, G_batch)  # batch matrix multiplication
        NSPd = NPd * (L == 0)
        NSP += NSPd
        L = L + d * (NSPd != 0).float()

    L[L == 0] = float('inf')
    L[I == 1] = 0
    NSP[NSP == 0] = 1

    DP = torch.zeros((batch_size, n, n), device=device)
    diam = d - 1

    # Calculate DP for all batches
    for d_val in range(diam, 1, -1):
        # [batch_size, n, n] operations
        L_eq_d = (L == d_val).float()
        L_eq_dm1 = (L == (d_val - 1)).float()
        DPd1 = torch.bmm((L_eq_d * (1 + DP) / NSP), G_batch.transpose(1, 2)) * (L_eq_dm1 * NSP)
        DP += DPd1
    
    result_batch = torch.sum(DP, dim=1)  # [batch_size, n]
    return result_batch


def edge_lengths_batched(conn_matrix_batch, Distance, device, directed=False):
    """
    批量提取边的长度
    
    参数:
        conn_matrix_batch: [batch_size, n, n] 的邻接矩阵 tensor，已经在 device 上
        Distance: [n, n] 的距离矩阵 tensor
        device: 设备
        directed: 是否为有向图
    
    返回:
        edge_lengths_list: list of tensors (长度为batch_size)，每个元素是 [num_edges_i] 的数组
    """
    conn_matrix_batch = torch.as_tensor(conn_matrix_batch, dtype=torch.float32, device=device)
    Distance = torch.as_tensor(Distance, dtype=torch.float32, device=device)
    
    batch_size, n, _ = conn_matrix_batch.shape
    conn_binary = (conn_matrix_batch > 0).float()
    edge_lengths_list = []
    
    if directed:
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        u_all, v_all = torch.where(mask)
        for i in range(batch_size):
            edges = conn_binary[i, u_all, v_all] > 0
            edge_lengths = Distance[u_all, v_all][edges]
            edge_lengths_list.append(edge_lengths)
    else:
        u_idx, v_idx = torch.triu_indices(n, n, offset=1, device=device)
        for i in range(batch_size):
            edges = conn_binary[i, u_idx, v_idx] > 0
            edge_lengths = Distance[u_idx, v_idx][edges]
            edge_lengths_list.append(edge_lengths)
    
    return edge_lengths_list


if __name__=='__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    import time

    # for _ in range(100):
    #     cij = np.random.randint(2, size=(50,50))
    #     tmp = betweenness_bin_gpu(cij, device)
    #     tmpp = bct.betweenness_bin(cij)
    #     if np.abs(tmp-tmpp).sum() > 1e-4:
    #         break
    #     print(cij)

    # pdb.set_trace()

    # 记录函数开始时间
    start_time = time.time()

    Dis = np.random.randint(5,size=(400,400))

    for _ in range(400):
        cij = np.random.randint(2, size=(400,400))
        cij = np.triu(cij, k=0) + np.triu(cij, k=1).T
        tmp0 = bct.degrees_und(cij)
        tmp0 = clustering_coef_bu_gpu(cij, device)
        tmp0 = betweenness_bin_gpu(cij, device)
        tmp0 = Dis[np.triu(cij, k=1) > 0]
        _, _, tmp2 = matching_ind_gpu(cij, device)

    # 记录函数开始时间
    end_time  = time.time()
    execution_time = end_time - start_time

    print(f"函数运行时间: {execution_time} 秒")


    # 记录函数开始时间
    start_time = time.time()

    for _ in range(400):
        cij = np.random.randint(2, size=(400,400))
        cij = np.triu(cij, k=0) + np.triu(cij, k=1).T
        tmp0 = bct.degrees_und(cij)
        tmp0 = bct.clustering_coef_bu(cij)
        tmp0 = bct.betweenness_bin(cij)
        tmp0 = Dis[np.triu(cij, k=1) > 0]
        _, _, tmp1 = bct.matching_ind(cij)

    # 记录函数开始时间
    end_time  = time.time()
    execution_time = end_time - start_time

    print(f"函数运行时间: {execution_time} 秒")