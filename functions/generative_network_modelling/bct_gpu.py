import numpy as np
import bct
import torch

def matching_ind_gpu(CIJ, device, only_min=False):
    CIJ = bct.utils.binarize(CIJ, copy=True)
    CIJ = torch.tensor(CIJ, dtype=torch.float, device=device)  # Transfer to GPU
    n = CIJ.shape[0]

    # Create mask to ignore self-connections
    mask = 1 - torch.eye(n, device=device)
    mask = mask.bool()
    CIJ = CIJ * mask

    # In-degree matching
    in_deg = CIJ.sum(dim=0, keepdim=True)
    common_in = torch.mm(CIJ, CIJ.t())
    in_or_deg = (in_deg + in_deg.t())
    in_or_deg = (in_or_deg - CIJ * 2)
    Min = common_in * 2 / in_or_deg
    Min[in_or_deg==0] = 0
    Min = Min * mask
    if only_min:
        return Min

    # Out-degree matching
    out_deg = CIJ.sum(dim=1, keepdim=True)
    common_out = torch.mm(CIJ.t(), CIJ) 
    out_or_deg = (out_deg + out_deg.t())
    out_or_deg = (out_or_deg - CIJ * 2)
    Mout = common_out * 2 / out_or_deg
    Mout[out_or_deg==0] = 0
    Mout = Mout * mask

    # Overall matching
    Mall = (Min + Mout) / 2

    return Min.cpu().numpy(), Mout.cpu().numpy(), Mall.cpu().numpy()


def clustering_coef_bu_gpu(G, device, to_tensor=False):
    G = bct.utils.binarize(G, copy=True)
    G = torch.tensor(G, dtype=torch.float32, device=device)  # Transfer matrix to GPU
    n = G.shape[0]
    C = torch.zeros(n, device=device)  # Initialize clustering coefficient vector on GPU

    # Calculate all possible triangle combinations
    G_square = torch.matmul(G, G)
    triangle_count = G_square * G  # Matrix multiplication then multiply with original matrix to get twice the triangles

    # Calculate degree of each node
    degrees = G.sum(dim=1)

    # Calculate clustering coefficient
    possible_triangle_count = degrees * (degrees - 1)  # Possible number of triangles for each node
    C = triangle_count.sum(dim=1) / possible_triangle_count  # Clustering coefficient

    # Handle division by zero
    C[possible_triangle_count == 0] = 0.0
    if to_tensor:
        return C
    return C.cpu().numpy()  # Transfer results back to CPU


def betweenness_bin_gpu(G, device, to_tensor=False):
    G = bct.utils.binarize(G, copy=True)
    G = torch.tensor(G, dtype=torch.float32, device=device)  # Transfer matrix to GPU
    n = len(G)  # Number of nodes
    I = torch.eye(n, device=device)
    d = 1  # Path length
    NPd = G.clone()  # Number of paths of length |d|
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

    DP = torch.zeros((n, n), device=device)  # Vertex on vertex dependency
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
    Batch compute degrees of undirected graphs
    
    Args:
        conn_matrix_batch: [batch_size, n, n] adjacency matrix tensor, already on device
        device: Device
    
    Returns:
        degrees_batch: [batch_size, n] degree tensor
    """
    conn_matrix_batch = torch.as_tensor(conn_matrix_batch, dtype=torch.float32, device=device)
    conn_matrix_batch = (conn_matrix_batch > 0).float()

    degrees_batch = conn_matrix_batch.sum(dim=2)  # [batch_size, n]
    return degrees_batch


def clustering_coef_bu_gpu_batched(G_batch, device):
    """
    Batch compute clustering coefficient
    
    Args:
        G_batch: [batch_size, n, n] adjacency matrix tensor, already on device
        device: Device
    
    Returns:
        C_batch: [batch_size, n] clustering coefficient tensor
    """
    # Ensure input is tensor and on correct device
    if isinstance(G_batch, np.ndarray):
        G_batch = torch.tensor(G_batch, dtype=torch.float32, device=device)
    else:
        G_batch = torch.as_tensor(G_batch, dtype=torch.float32, device=device)
    
    # Binarize
    G_batch = (G_batch > 0).float()
    
    batch_size, n, _ = G_batch.shape
    
    # Batch compute G^2: [batch_size, n, n]
    G_square = torch.bmm(G_batch, G_batch)  # Batch matrix multiplication
    triangle_count = G_square * G_batch  # [batch_size, n, n]
    
    # Calculate degree of each node: [batch_size, n]
    degrees = G_batch.sum(dim=2)  # [batch_size, n]
    
    # Calculate clustering coefficient
    possible_triangle_count = degrees * (degrees - 1)  # [batch_size, n]
    C_batch = triangle_count.sum(dim=2) / possible_triangle_count  # [batch_size, n]
    
    # Handle division by zero
    C_batch[possible_triangle_count == 0] = 0.0
    return C_batch

def betweenness_bin_gpu_batched(G_batch, device):
    """
    Batch compute betweenness centrality
    
    Args:
        G_batch: [batch_size, n, n] adjacency matrix tensor, already on device
        device: Device
    
    Returns:
        result_batch: [batch_size, n] betweenness centrality tensor
    """
    # Ensure input is tensor and on correct device
    if isinstance(G_batch, np.ndarray):
        G_batch = torch.tensor(G_batch, dtype=torch.float32, device=device)
    else:
        G_batch = torch.as_tensor(G_batch, dtype=torch.float32, device=device)
    
    # Binarize
    G_batch = (G_batch > 0).float()
    
    batch_size, n, _ = G_batch.shape
    I = torch.eye(n, device=device).unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n, n]
    
    d = 1  # Path length
    NPd = G_batch.clone()  # [batch_size, n, n]
    NSPd = G_batch.clone()
    NSP = G_batch.clone()
    L = G_batch.clone()

    NSP[I == 1] = 1
    L[I == 1] = 1

    # Calculate NSP and L for all batches
    while torch.any(NSPd):
        d += 1
        NPd = torch.bmm(NPd, G_batch)  # Batch matrix multiplication
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
    Batch extract edge lengths
    
    Args:
        conn_matrix_batch: [batch_size, n, n] adjacency matrix tensor, already on device
        Distance: [n, n] distance matrix tensor
        device: Device
        directed: Whether graph is directed
    
    Returns:
        edge_lengths_list: list of tensors (length batch_size), each element is [num_edges_i] array
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

    # Record function start time
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

    # Record function start time
    end_time  = time.time()
    execution_time = end_time - start_time

    print(f"Function execution time: {execution_time} seconds")


    # Record function start time
    start_time = time.time()

    for _ in range(400):
        cij = np.random.randint(2, size=(400,400))
        cij = np.triu(cij, k=0) + np.triu(cij, k=1).T
        tmp0 = bct.degrees_und(cij)
        tmp0 = bct.clustering_coef_bu(cij)
        tmp0 = bct.betweenness_bin(cij)
        tmp0 = Dis[np.triu(cij, k=1) > 0]
        _, _, tmp1 = bct.matching_ind(cij)

    # Record function start time
    end_time  = time.time()
    execution_time = end_time - start_time

    print(f"Function execution time: {execution_time} seconds")