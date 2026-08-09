import torch
import numpy as np
from typing import List


def find_connected_components(adj_matrix):     
    n = len(adj_matrix)  
      
    visited = {i: False for i in range(n)}  
    components = []  

    def dfs(node, component):
        # Mark current node as visited
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
    Generate an adjacency matrix based on total_connections and connection density weights.

    This function ensures the generated graph has exactly total_connections edges.

    Args:
        core_sizes (List[int]): Size of each core, e.g., [30, 25].
        periphery_size (int): Size of the periphery region.
        total_connections (int): Total number of connections (edges) to generate in the graph.
        intra_core_weight (float): Relative weight for connections within cores.
        inter_core_weight (float): Relative weight for connections between different cores.
        core_periphery_weight (float): Relative weight for connections between cores and periphery.
        intra_periphery_weight (float): Relative weight for connections within periphery.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: NumPy adjacency matrix of the generated graph.
    """
    # 0. Initialize
    rng = np.random.default_rng(seed)
    num_cores = len(core_sizes)
    total_nodes = sum(core_sizes) + periphery_size
    # 1. Create a mapping from node ID to its block
    # For example, core_sizes=[10,5], periphery_size=8
    # Nodes 0-9 belong to block 0 (core 1), 10-14 belong to block 1 (core 2), 15-22 belong to block 2 (periphery)
    node_to_block = np.zeros(total_nodes, dtype=int)
    start_idx = 0
    for i, size in enumerate(core_sizes):
        node_to_block[start_idx : start_idx + size] = i
        start_idx += size
    if periphery_size > 0:
        node_to_block[start_idx:] = num_cores # Periphery block ID is the last one

    # 2. Generate all possible connections and their weights
    possible_edges = []
    edge_weights = []

    # Iterate through all node pairs (i, j) 
    for i in range(total_nodes):
        for j in range(i, total_nodes):
            block_i = node_to_block[i]
            block_j = node_to_block[j]
            
            weight = 0.0
            # Case 1: Two nodes in the same block
            if block_i == block_j:
                if block_i < num_cores: # Within core
                    weight = intra_core_weight
                else: # Within periphery
                    weight = intra_periphery_weight
            # Case 2: Two nodes in different blocks
            else:
                is_i_core = (block_i < num_cores)
                is_j_core = (block_j < num_cores)
                if is_i_core and is_j_core: # Between different cores
                    weight = inter_core_weight
                else: # Between core and periphery
                    weight = core_periphery_weight
            
            if weight > 0.0:
                possible_edges.append((i, j))
                edge_weights.append(weight)
                
                if j != i:
                    possible_edges.append((j, i))
                    edge_weights.append(weight)

    if total_connections > len(possible_edges):
        raise ValueError(
            f"Requested total connections {total_connections} exceeds the maximum number of connections {len(possible_edges)} that can be created based on current weights."
        )

    # 3. Sample based on weights
    # Convert weights to probabilities
    total_weight = sum(edge_weights)
    if total_weight == 0:
        if total_connections > 0: raise ValueError("All weights are 0, cannot generate any connections.")
        return np.zeros((total_nodes, total_nodes), dtype=int)

    probabilities = np.array(edge_weights) / total_weight

    # Sample total_connections from all possible connections without replacement based on probabilities
    chosen_indices = rng.choice(
        len(possible_edges), 
        size=total_connections, 
        replace=False, 
        p=probabilities
    )

    # 4. Build adjacency matrix
    adj_matrix = np.zeros((total_nodes, total_nodes), dtype=int)
    for index in chosen_indices:
        u, v = possible_edges[index]
        adj_matrix[u, v] = 1
        
    return adj_matrix

