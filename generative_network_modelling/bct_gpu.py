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


def clustering_coef_bu_gpu(G, device):
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

    return C.cpu().numpy()  # 将结果转回 CPU

def betweenness_bin_gpu(G, device):
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

    return torch.sum(DP, dim=0).cpu().numpy()


# def matching_ind_fc(CIJ, fc):
#     n = len(CIJ)

#     Min = np.zeros((n, n))

#     # compare incoming connections
#     for i in range(n - 1):
#         for j in range(i + 1, n):
#             c1i = CIJ[:, i]
#             c2i = CIJ[:, j]

#             fc1i = fc[:, i]
#             fc2i = fc[:, j]
#             usei = np.logical_or(c1i, c2i)
#             usei[i] = 0
#             usei[j] = 0
#             nconi = np.sum(fc1i[usei]) + np.sum(fc2i[usei])
#             if not nconi:
#                 Min[i, j] = 0
#             else:
#                 and_mask = np.logical_and(c1i, c2i) 
#                 Min[i, j] = 2 * \
#                     (np.sum(fc1i[and_mask])+np.sum(fc2i[and_mask])) / nconi

#     Min = Min + Min.T

#     return Min

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