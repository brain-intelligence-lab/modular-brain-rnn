import random
import torch
import numpy as np
from typing import List
import os

def lock_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = "myseed"  # str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def find_connected_components(adj_matrix):     
    n = len(adj_matrix)  
      
    visited = {i: False for i in range(n)}  
    components = []  

    def dfs(node, component):  
        # 标记当前节点为已访问  
        visited[node] = True  
        component.append(node)        
        for neighbor in range(n):  
            if adj_matrix[node][neighbor] and not visited[neighbor]:  
                dfs(neighbor, component)  
    
    for i in range(n):  
        if not visited[i]:  
            component = []  
            dfs(i, component)  
            components.append(component)  

    return components


def get_induced_subgraphs(weight_matrix, components):  
    N = len(weight_matrix)  
    subgraphs = []  
      
    for component in components:  
        n = len(component)
        if n < 3:
            continue

        subgraph_matrix = np.zeros((n, n))

        for i in range(len(component)):
            for j in range(len(component)):
                u = component[i]
                v = component[j]
                subgraph_matrix[i,j] = weight_matrix[u,v]
          
        subgraphs.append(subgraph_matrix)  
      
    return subgraphs  



# --- Helper Function to Calculate ED ---
def calculate_effective_dimensionality(activations):
    """
    Calculates the Effective Dimensionality (Participation Ratio)
    from a set of activations.
    
    Args:
    activations (torch.Tensor): A 2D tensor of shape (Total_Samples, Num_Neurons)
    
    Returns:
    torch.Tensor: A single scalar value for the ED.
    """
    # 1. Center the data
    mean = torch.mean(activations, dim=0)
    centered_states = activations - mean
    
    # 2. Compute the covariance matrix
    # (Num_Neurons, Num_Neurons)
    cov_matrix = (centered_states.T @ centered_states) / (centered_states.shape[0] - 1)
    
    # 3. Get the eigenvalues
    # Using .eigh() for symmetric matrices
    eigenvalues = torch.linalg.eigh(cov_matrix)[0]
    
    # 4. Clamp eigenvalues (due to numerical precision)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    
    # 5. Calculate ED (Participation Ratio)
    # ED = (sum(lambda_i))^2 / sum(lambda_i^2)
    sum_eig = torch.sum(eigenvalues)
    sum_eig_sq = torch.sum(eigenvalues**2)
    
    # Avoid division by zero if all eigenvalues are tiny
    if sum_eig_sq < 1e-10:
        return torch.tensor(0.0)
        
    eff_dim = (sum_eig**2) / sum_eig_sq
    return eff_dim

def gen_ortho_matrix(dim, rng=None):
    """Generate random orthogonal matrix
    Taken from scipy.stats.ortho_group
    Copied here from compatibilty with older versions of scipy
    """
    H = np.eye(dim)
    for n in range(1, dim):
        if rng is None:
            x = np.random.normal(size=(dim-n+1,))
        else:
            x = rng.normal(size=(dim-n+1,))
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0])
        x[0] += D*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = -D*(np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    return H


# core-periphery organization
def generate_adj_matrix(
    core_sizes: List[int],
    periphery_size: int,
    total_connections: int,
    intra_core_weight: float = 0.8,
    inter_core_weight: float = 0.1,
    core_periphery_weight: float = 0.08,
    intra_periphery_weight: float = 0.02,
    seed: int = None
) -> np.ndarray:
    """
    根据给定的总连接数 total_connections 和各区域的连接“密度”权重，生成一个邻接矩阵。

    此函数确保生成的图中边的总数恰好为 total_connections。

    Args:
        core_sizes (List[int]): 每个核心的大小，例如 [30, 25]。
        periphery_size (int): 边缘区域的大小。
        total_connections (int): 图中需要生成的总连接数（边的数量）。
        intra_core_weight (float): 核心内部连接的相对权重。
        inter_core_weight (float): 不同核心之间连接的相对权重。
        core_periphery_weight (float): 核心与边缘之间连接的相对权重。
        intra_periphery_weight (float): 边缘内部连接的相对权重。
        seed (int, optional): 用于复现结果的随机种子。

    Returns:
        np.ndarray: 生成图的 NumPy 邻接矩阵。
    """
    # 0. 初始化
    rng = np.random.default_rng(seed)
    num_cores = len(core_sizes)
    total_nodes = sum(core_sizes) + periphery_size
    # 1. 建立一个从节点ID到其所属块(block)的映射
    # 例如，core_sizes=[10,5], periphery_size=8
    # 节点0-9属于块0(核心1), 10-14属于块1(核心2), 15-22属于块2(边缘)
    node_to_block = np.zeros(total_nodes, dtype=int)
    start_idx = 0
    for i, size in enumerate(core_sizes):
        node_to_block[start_idx : start_idx + size] = i
        start_idx += size
    if periphery_size > 0:
        node_to_block[start_idx:] = num_cores # 边缘区的块ID是最后一个

    # 2. 生成所有可能的连接及其权重
    possible_edges = []
    edge_weights = []
    
    # 遍历所有节点对 (i, j) 
    for i in range(total_nodes):
        for j in range(i, total_nodes):
            block_i = node_to_block[i]
            block_j = node_to_block[j]
            
            weight = 0.0
            # 情况1: 两个节点在同一个块中
            if block_i == block_j:
                if block_i < num_cores: # 核心内部
                    weight = intra_core_weight
                else: # 边缘内部
                    weight = intra_periphery_weight
            # 情况2: 两个节点在不同块中
            else:
                is_i_core = (block_i < num_cores)
                is_j_core = (block_j < num_cores)
                if is_i_core and is_j_core: # 两个不同核心之间
                    weight = inter_core_weight
                else: # 核心与边缘之间
                    weight = core_periphery_weight
            
            if weight > 0.0:
                possible_edges.append((i, j))
                edge_weights.append(weight)
                
                if j != i:
                    possible_edges.append((j, i))
                    edge_weights.append(weight)

    if total_connections > len(possible_edges):
        raise ValueError(
            f"请求的总连接数 {total_connections} 超过了基于当前权重所能创建的最大连接数 {len(possible_edges)}。"
        )

    # 3. 根据权重进行抽样
    # 将权重转换为概率
    total_weight = sum(edge_weights)
    if total_weight == 0:
        if total_connections > 0: raise ValueError("所有权重均为0，无法生成任何连接。")
        return np.zeros((total_nodes, total_nodes), dtype=int)
        
    probabilities = np.array(edge_weights) / total_weight
    
    # 从所有可能的连接中，根据概率不重复地抽取 total_connections 个
    chosen_indices = rng.choice(
        len(possible_edges), 
        size=total_connections, 
        replace=False, 
        p=probabilities
    )
    
    # 4. 构建邻接矩阵
    adj_matrix = np.zeros((total_nodes, total_nodes), dtype=int)
    for index in chosen_indices:
        u, v = possible_edges[index]
        adj_matrix[u, v] = 1
        
    return adj_matrix


def compute_syn_loglik(s_obs, s_sims, use_diag_cov=False, ridge=1e-6):
    """
    计算合成似然，输入全是 Tensor (GPU/CPU 均可，建议 GPU)
    s_obs: [d]
    s_sims: [n_sims, d]
    """
    n_sims, d = s_sims.shape
    mu = s_sims.mean(dim=0)
    
    if use_diag_cov or (n_sims <= d + 2):
        # Diagonal Covariance
        var = s_sims.var(dim=0, unbiased=True)
        var = torch.clamp(var, min=ridge)
        diff2 = (s_obs - mu) ** 2
        # log likelihood
        ll = -0.5 * (torch.sum(torch.log(2.0 * np.pi * var)) + torch.sum(diff2 / var))
    else:
        # Full Covariance
        X = s_sims - mu
        # Sample covariance: (X.T @ X) / (N-1)
        Sigma = (X.T @ X) / (n_sims - 1)
        Sigma = Sigma + ridge * torch.eye(d, device=s_sims.device, dtype=s_sims.dtype)
        
        try:
            # Cholesky decomposition
            L = torch.linalg.cholesky(Sigma)
            diff = (s_obs - mu)
            # Solve L y = diff -> y = L^{-1} diff
            y = torch.linalg.solve_triangular(L, diff.unsqueeze(1), upper=False)
            quad = torch.sum(y ** 2)
            logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
            ll = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)
        except RuntimeError:
            # Fallback if Cholesky fails (not PSD)
            return -1e9

    return ll.item()


def compute_syn_loglik_with_pca(s_obs, s_sims, n_components=10, use_diag_cov=True):
    """
    先对数据进行 PCA 降维，再计算合成似然。
    
    参数:
        s_obs: [d] 观测统计量
        s_sims: [n_sims, d] 模拟统计量
        n_components: 保留的主成分数量 (推荐 8-12)
        use_diag_cov: 降维后是否使用对角协方差 (PCA后特征理论上不相关，True 是合理的)
        
    注意:
        由于特征量纲不同，PCA 前会自动进行 Z-score 标准化。
    """
    # 1. 自动标准化 (Standardization)
    # 使用模拟数据的均值和标准差来标准化观测数据，防止数据泄露
    mu = s_sims.mean(dim=0)
    # 添加微小 epsilon 防止除以 0 (如果有特征在所有模拟中完全不变)
    std = s_sims.std(dim=0, unbiased=True) + 1e-6 
    
    s_sims_norm = (s_sims - mu) / std
    s_obs_norm = (s_obs - mu) / std
    
    # 2. 执行 PCA (使用 SVD 方法)
    # 对 s_sims_norm 进行 SVD 分解: X = U @ S @ Vh
    # Vh 的行就是主成分方向 (Eigenvectors)
    # full_matrices=False 确保我们得到紧凑形式
    try:
        U, S, Vh = torch.linalg.svd(s_sims_norm, full_matrices=False)
    except RuntimeError:
        # 如果 SVD 不收敛 (极少见)，返回极小似然值
        return -1e9

    # 3. 截断并投影
    # Vh: [min(n_sims, d), d]
    # 取前 n_components 个成分。注意 Vh 的形状，我们需要转置来进行投影
    # components: [d, n_components]
    components = Vh[:n_components, :].T 
    
    # 投影数据: [n_sims, d] @ [d, k] -> [n_sims, k]
    s_sims_pca = s_sims_norm @ components
    s_obs_pca = s_obs_norm @ components
    
    # 4. 在降维空间计算似然
    # PCA 的特性保证了 s_sims_pca 的各列之间是线性不相关的 (协方差矩阵是对角的)
    # 所以这里 use_diag_cov=True 不仅是近似，在理论上也是更准确的
    return compute_syn_loglik(s_obs_pca, s_sims_pca, use_diag_cov=use_diag_cov)


def check_pca_variance(s_sims):
    """
    打印 PCA 各主成分的解释方差比，帮助决定 n_components
    s_sims: [n_sims, d]
    """
    # 1. 标准化
    mu = s_sims.mean(dim=0)
    std = s_sims.std(dim=0, unbiased=True) + 1e-6
    s_sims_norm = (s_sims - mu) / std
    
    # 2. SVD 分解
    # U, S, Vh = torch.linalg.svd(s_sims_norm, full_matrices=False)
    # 奇异值 S 的平方与特征值（方差）成正比
    _, S, _ = torch.linalg.svd(s_sims_norm, full_matrices=False)
    
    # 3. 计算方差解释比
    eigvals = S ** 2
    total_variance = torch.sum(eigvals)
    explained_variance_ratio = eigvals / total_variance
    cumulative_variance_ratio = torch.cumsum(explained_variance_ratio, dim=0)
    
    # 4. 打印结果
    print(f"{'Component':<10} | {'Variance Explained':<20} | {'Cumulative':<10}")
    print("-" * 45)
    
    # 转为 numpy 方便打印
    evr = explained_variance_ratio.cpu().numpy()
    cvr = cumulative_variance_ratio.cpu().numpy()
    
    for i in range(len(evr)):
        print(f"{i+1:<10} | {evr[i]:.4f} ({evr[i]*100:.1f}%)   | {cvr[i]*100:.1f}%")
        
    return cvr
