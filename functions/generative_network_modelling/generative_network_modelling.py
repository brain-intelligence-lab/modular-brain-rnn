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
    # x1, x2 are both 1D vectors
    # Convert x1 and x2 to PyTorch tensors and move to GPU
    x1 = torch.tensor(x1, device=device)
    x2 = torch.tensor(x2, device=device)

    # Sort x1 and x2
    sorted_x1, _ = torch.sort(x1)
    sorted_x2, _ = torch.sort(x2)
    
    # Get all unique values
    unique_values = torch.unique(torch.cat((sorted_x1, sorted_x2)))
    
    # Calculate CDF of each array at these points
    cdf_x1 = torch.searchsorted(sorted_x1, unique_values, right=True) / x1.size(0)
    cdf_x2 = torch.searchsorted(sorted_x2, unique_values, right=True) / x2.size(0)
    
    # Calculate maximum difference between CDFs
    max_diff = torch.max(torch.abs(cdf_x1 - cdf_x2))
    
    return max_diff.item()



def ks_statistic_batch_gpu(x1, x2, device):
    """
    Batch parallel version of KS statistic (GPU)
    
    Args:
    x1: Tensor, shape (B, N) or list
    x2: Tensor, shape (B, M)
    
    Returns:
    max_diff: Tensor, shape (B,) KS statistic for each sample
    """
    # Ensure on the same device
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

    # 1. Sort x1 and x2 in each row
    sorted_x1, _ = torch.sort(x1, dim=-1)
    sorted_x2, _ = torch.sort(x2, dim=-1)

    # 2. Construct evaluation points (Concatenate x1 and x2)
    # When calculating KS, maximum difference must occur at sample points of x1 or x2
    # Shape is (B, N + M)
    combined = torch.cat([sorted_x1, sorted_x2], dim=-1)
    # Sorting here is to work with searchsorted, shape (B, N + M)
    evaluation_points, _ = torch.sort(combined, dim=-1)

    # 3. Calculate Batch CDF
    # searchsorted supports batch mode: 
    # sorted_sequence (B, N), input (B, N+M) -> output (B, N+M)
    cdf_x1 = torch.searchsorted(sorted_x1, evaluation_points, right=True).float() / N
    cdf_x2 = torch.searchsorted(sorted_x2, evaluation_points, right=True).float() / M

    # 4. Calculate maximum absolute difference
    # Take maximum on dim=-1 (N+M dimension)
    max_diff = torch.max(torch.abs(cdf_x1 - cdf_x2), dim=-1)[0]

    return max_diff


def Gen_one_connection(A, params, modelvar, device, D=None, use_matching=False, Fc=None, undirected=True):
    eta, gam, epsilon = params
    if use_matching:
        Kseed, _, _ = bct_gpu.matching_ind_gpu(A, device=device)
        # Kseed, _, _ = bct.matching_ind(A)
        Kseed = Kseed + epsilon  # Add the epsilon

    n = len(A)  # Take the number of nodes
    mv1 = modelvar[0]  # Take whether power law or exponential
    mv2 = modelvar[1]

    Fd = np.ones_like(A) / A.size
    Fk = np.ones_like(A) / A.size

    # Compute parameterized costs and values for wiring
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
    Ff = Fd * Fk * ~A.astype(bool)  # For non-existent edges
    if Fc is not None:
        Ff = Ff * (Fc + epsilon)
    
    if undirected:
        u_indx, v_indx = np.where(np.triu(np.ones((n, n)), k=1))  # Compute indices
    else:
        u_indx, v_indx = np.where(np.ones((n, n)))  # Compute indices
    
    indx = u_indx  * n + v_indx
    P = Ff.flatten()[indx]  # get the probability vector

    # Add connection
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
        # Extract off-diagonal indices
        u_indx, v_indx = np.where(~np.eye(n, dtype=bool))
    else:
        # Extract upper triangular indices (excluding self-loops k=1, use k=0 to include self-loops)
        u_indx, v_indx = np.triu_indices(n, k=1)
    
    # Get probabilities corresponding to indices and normalize
    probs = prob_matrix[u_indx, v_indx]
    probs /= probs.sum() # Renormalize to ensure probability sums to 1
    
    # The choice here selects indices from the u_indx array
    if directed: 
        sampled_indices = np.random.choice(len(u_indx), \
            size=tot_conn_num, replace=False, p=probs)
    else:
        sampled_indices = np.random.choice(len(u_indx), \
            size=int(tot_conn_num//2), replace=False, p=probs)
    
    # Map back to original matrix coordinates
    final_u = u_indx[sampled_indices]
    final_v = v_indx[sampled_indices]
    
    # Build result matrix
    result_matrix = np.zeros((n, n), dtype=int)
    result_matrix[final_u, final_v] = 1
    if not directed:
        result_matrix[final_v, final_u] = 1 # Ensure symmetry for undirected graphs
        
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
    Highly optimized GNM batch sampler:
    # 1. Full GPU computation, no CPU transfer.
    # 2. Support ultra-large batch sampling (generate batch_size graphs at once, batch_size = eta_vec * gamma_vec * n_sims)。
    # 3. Use torch.multinomial instead of extremely slow np.random.choice.
        
    Args:
        prob_matrix_batch: [batch_size, n, n] probability matrix tensor, already on device
        tot_conn_num: Total number of connections per graph
        directed: Whether graph is directed
        device: Device ('cuda:0' etc.)
    
    Returns:
        adj_matrices: [batch_size, n, n] adjacency matrix tensor
    """
    
    # 1. Ensure input is Tensor and on correct device
    prob_matrix_batch = torch.as_tensor(prob_matrix_batch, device=device, dtype=torch.float32)
    batch_size, n, _ = prob_matrix_batch.shape
    
    # 2. Extract probabilities of valid connections (Masking)
    if directed:
        # Remove diagonal - use same method as reconstruction index
        mask = ~torch.eye(n, dtype=torch.bool, device=device)  # [n, n]
        u_all, v_all = torch.where(mask)  # Get coordinates of off-diagonal elements
        # Extract valid probabilities for each batch: [batch_size, n*(n-1)]
        valid_probs = prob_matrix_batch[:, u_all, v_all]  # [batch_size, n*(n-1)]
        num_edges_to_sample = tot_conn_num
    else:
        # Only take upper triangular (k=1)
        u_idx, v_idx = torch.triu_indices(n, n, offset=1, device=device)
        # Extract upper triangular for each batch: [batch_size, n*(n-1)/2]
        valid_probs = prob_matrix_batch[:, u_idx, v_idx]  # [batch_size, n*(n-1)/2]
        num_edges_to_sample = int(tot_conn_num // 2)

    # 3. Normalize (normalize for each batch separately)
    # [batch_size, num_candidates] -> Normalize on dim=1
    valid_probs = valid_probs / valid_probs.sum(dim=1, keepdim=True)

    # 4. Batch sampling (core acceleration point)
    # torch.multinomial is extremely fast for weighted sampling without replacement on GPU
    # sampled_indices shape: [batch_size, num_edges_to_sample]
    sampled_flat_indices = torch.multinomial(valid_probs, num_edges_to_sample, replacement=False)

    # 5. Reconstruct adjacency matrix
    # Create all-zero container: [batch_size, n, n]
    adj_matrices = torch.zeros((batch_size, n, n), dtype=torch.int8, device=device)

    if directed:
        # Map back to 2D coordinates
        # Use same indexing method as when extracting valid_probs
        mask = ~torch.eye(n, dtype=torch.bool, device=device)  # [n, n]
        u_all, v_all = torch.where(mask)  # Get coordinates of off-diagonal elements
        valid_linear_indices = (u_all * n + v_all).long()  # [n*(n-1)]
        
        # Get actual linear indices sampled
        # sampled_flat_indices is [batch_size, num_edges] long tensor
        # valid_linear_indices[sampled_flat_indices] -> shape [batch_size, num_edges]
        real_flat_indices = valid_linear_indices[sampled_flat_indices]  # [batch_size, num_edges]
        
        # Fill with 1, scatter_ requires long type indices
        flat_adj = torch.zeros((batch_size, n * n), dtype=torch.int8, device=device)
        flat_adj.scatter_(1, real_flat_indices, 1)
        adj_matrices = flat_adj.view(batch_size, n, n)
        
    else:
        # Undirected graph processing
        # 1. Prepare linear indices
        # u_idx, v_idx are already upper triangular coordinates
        # Convert 2D coordinates to linear coordinates: idx = u * n + v
        # Ensure data type is long
        flat_indices_map = (u_idx * n + v_idx).long()  # [num_candidates]
        
        # 2. Get actual linear indices selected in batch
        batch_real_indices = flat_indices_map[sampled_flat_indices]  # [batch_size, num_edges]

        # 3. Fill upper triangular
        flat_adj = torch.zeros((batch_size, n * n), dtype=torch.int8, device=device)
        flat_adj.scatter_(1, batch_real_indices.long(), 1)
        adj_upper = flat_adj.view(batch_size, n, n)
        
        # 4. Symmetrize: A + A.T
        adj_matrices = adj_upper + adj_upper.transpose(1, 2)

    return adj_matrices.float()  # Return float for subsequent statistic computation


def get_graph_distribution(conn_matrix, Distance, device, to_tensor=False):
    """
    Calculate distribution statistics for a single graph (maintain backward compatibility)
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
    Batch compute distribution statistics of graphs
    
    Args:
        conn_matrix_batch: [batch_size, n, n] adjacency matrix tensor, already on device
        Distance: [n, n] distance matrix (numpy array or tensor)
        device: Device
        to_tensor: Whether to return tensor (default True)
        directed: Whether graph is directed (default False)
    
    Returns:
        target: list of tensors
            - degrees: [batch_size, n]
            - clustering: [batch_size, n]
            - betweenness: [batch_size, n]
            - edge_lengths: list of arrays/tensors, each element is [num_edges_i] array
    """
    conn_matrix_batch = torch.as_tensor(conn_matrix_batch, dtype=torch.float32, device=device)
    Distance = torch.as_tensor(Distance, dtype=torch.float32, device=device)
    
    # All functions return tensor first, then process uniformly
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

    # Record function start time
    start_time = time.time()

    for _ in range(10000):
        x1 = np.random.randint(1000, size=(2000))
        x2 = np.random.randint(1000, size=(2000))

        value_0 = ks_statistic_gpu(x1, x2, device=device)

        # value_1 = ks_statistic(x1, x2)


    # Record function start time
    end_time  = time.time()
    execution_time = end_time - start_time

    print(f"Function execution time: {execution_time} seconds")


    # Record function start time
    start_time = time.time()

    for _ in range(10000):
        x1 = np.random.randint(1000, size=(2000))
        x2 = np.random.randint(1000, size=(2000))

        # value_0 = ks_statistic_gpu(x1, x2, device=device)

        value_1 = ks_statistic(x1, x2)


    # Record function start time
    end_time  = time.time()
    execution_time = end_time - start_time

    print(f"Function execution time: {execution_time} seconds")