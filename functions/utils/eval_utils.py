import torch
import torch.nn.functional as F
import numpy as np
import datasets.multitask as task
from collections import defaultdict
from typing import List
import random
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

def calculate_modularity_in_r(weight_matrix: np.ndarray, r_script_path:str, verbose:bool=False):
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri, default_converter, conversion


    assert os.path.exists(r_script_path), f"R script '{r_script_path}' not found."

    np_cv_rules = default_converter + numpy2ri.converter
    with conversion.localconverter(np_cv_rules):
        r = robjects.r

        r.source(r_script_path) # 加载 R 脚本
        lpa_wb_plus_func = robjects.globalenv['LPA_wb_plus']
        dirt_lpa_func = robjects.globalenv['DIRT_LPA_wb_plus']
        r_matrix = robjects.conversion.py2rpy(weight_matrix) # 转换为 R 矩阵

        # 调用 R 函数
        if verbose:
            print(" 正在调用 LPA_wb_plus(MAT)...") 
        mod1_result = lpa_wb_plus_func(r_matrix)
        
        if verbose:
            print(" 正在调用 DIRT_LPA_wb_plus(MAT)...") 
        mod2_result = dirt_lpa_func(r_matrix)


    modularity1 = mod1_result['modularity'][0]
    modularity2 = mod2_result['modularity'][0]

    if verbose:
        print(f" 模块度结果: LPA_wb_plus = {modularity1}, DIRT_LPA_wb_plus = {modularity2}")
    
    return modularity1, modularity2


def calculate_modularity_for_fc_layer(weight_in_matrix: np.ndarray, weight_out_matrix:str, verbose:bool=False):
    import bct
    # weight_in_matrix: n * m, weight_out_matrix: m * q

    feature_matrix_1 = np.matmul(weight_in_matrix.T, weight_in_matrix)

    feature_matrix_2 = np.matmul(weight_out_matrix, weight_out_matrix.T)

    ci1, modularity1 = bct.modularity_dir(np.abs(feature_matrix_1))
    ci2, modularity2 = bct.modularity_dir(np.abs(feature_matrix_2))

    if verbose:
        print(f"mod1:{modularity1}, mod2:{modularity2}")
    
    return modularity1, modularity2


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



def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)


def popvec_torch(y):
    """
    Population vector read out (PyTorch version).
    在指定的设备上（CPU或GPU）运行。

    Args:
        y: Population output. PyTorch Tensor (Batch, Units)

    Returns:
        Readout locations: PyTorch Tensor (Batch,)
    """
    # 确保 pref 张量和输入 y 在同一个设备上
    device = y.device
    pref = torch.arange(0, 2 * torch.pi, 2 * torch.pi / y.shape[-1], device=device)  # preferences
    
    temp_sum = y.sum(axis=-1)
    # 使用 PyTorch 函数
    temp_cos = torch.sum(y * torch.cos(pref), axis=-1) / temp_sum
    temp_sin = torch.sum(y * torch.sin(pref), axis=-1) / temp_sum
    loc = torch.arctan2(temp_sin, temp_cos)
    
    return torch.remainder(loc, 2 * torch.pi)
    
def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.2*np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return perf


def get_perf_torch(y_hat, y_loc):
    """
    Get performance (PyTorch version).
    在指定的设备上（CPU或GPU）运行。

    Args:
      y_hat: Actual output. PyTorch Tensor (Time, Batch, Unit)
      y_loc: Target output location. PyTorch Tensor (Time, Batch)

    Returns:
      perf: PyTorch Tensor (Batch,)
    """
    if y_hat.ndim == 3:
        y_loc, y_hat = y_loc[-1], y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec_torch(y_hat[..., 1:]) # 调用 PyTorch 版本的 popvec

    # 在 PyTorch 中，布尔运算结果是 BoolTensor，需要转换为 float 进行数学运算
    fixating = (y_hat_fix > 0.5).float()

    original_dist = y_loc - y_hat_loc
    dist = torch.minimum(torch.abs(original_dist), 2 * torch.pi - torch.abs(original_dist))
    corr_loc = (dist < 0.2 * torch.pi).float()

    # Should fixate?
    should_fix = (y_loc < 0).float()

    # performance
    # (1-should_fix) 对应 (y_loc >= 0) 的情况
    perf = should_fix * fixating + (1 - should_fix) * corr_loc * (1 - fixating)
    return perf


def do_eval(model, rule_train, verbose=False):
    """Do evaluation.

    Args:
        model: Model class instance
        rule_train: string or list of strings, the rules being trained
    """
    log = defaultdict(list)
    hp = model.hp
    device = next(model.parameters()).device
    model.eval()
    for rule_test in hp['rules']:
        n_rep = 16
        batch_size_test_rep = int(hp['batch_size_test']/n_rep)
        perf_tmp = list()
        with torch.no_grad():
            for i_rep in range(n_rep):
                trial = task.generate_trials(
                    rule_test, hp, 'random', batch_size=batch_size_test_rep)
            
                input = torch.from_numpy(trial.x).to(device)
                y_hat_test = model(input)
                if hp['loss_type'] == 'lsq' and not hp['use_snn']:
                    y_hat_test = torch.sigmoid(y_hat_test).cpu().numpy()
                else:
                    y_hat_test = torch.nn.functional.softmax(y_hat_test, dim=-1)
                    y_hat_test = y_hat_test.cpu().numpy()
                
                # Cost is first summed over time,
                # and averaged across batch and units
                # We did the averaging over time through c_mask
                perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
                perf_tmp.append(perf_test)

        log['perf_'+rule_test].append(np.mean(perf_tmp, dtype=np.float64))
    model.train()

    if hasattr(rule_train, '__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]

    perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_avg'].append(perf_tests_mean)

    perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_min'].append(perf_tests_min)
    
    if verbose:
        print('avg'+'  | perf {:0.3f}'.format(perf_tests_mean))
        print('min'+'  | perf {:0.3f}'.format(perf_tests_min))

    return log


def do_eval_with_dataset(model, rule_train, dataset, verbose = False):
    """Do evaluation.

    Args:
        model: Model class instance
        rule_train: string or list of strings, the rules being trained
    """
    log = defaultdict(list)
    hp = model.hp
    model.eval()

    for rule_test in hp['rules']:
        perf_tmp = list()
        with torch.no_grad():
            n_rep = len(dataset[rule_test])
            for i_rep in range(n_rep):
                input, y_loc = dataset[rule_test][i_rep]    
                # import pdb
                # pdb.set_trace()    

                y_hat_test = model(input)
                if hp['loss_type'] == 'lsq' and not hp['use_snn']:
                    y_hat_test = torch.sigmoid(y_hat_test).cpu().numpy()
                else:
                    y_hat_test = F.softmax(y_hat_test, dim=-1)
                    y_hat_test = y_hat_test.cpu().numpy()
                
                perf_test = np.mean(get_perf(y_hat_test, y_loc))
                perf_tmp.append(perf_test)

        log['perf_'+rule_test].append(np.mean(perf_tmp, dtype=np.float64))
    model.train()

    if hasattr(rule_train, '__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]

    perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_avg'].append(perf_tests_mean)

    perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_min'].append(perf_tests_min)
    
    if verbose:
        print('avg'+'  | perf {:0.3f}'.format(perf_tests_mean))
        print('min'+'  | perf {:0.3f}'.format(perf_tests_min))

    return log


# ==============================================================================
# 终极方案 全局单次快速评估
# ==============================================================================
def do_eval_with_dataset_torch_fast(model, rule_train, global_data, verbose=False):
    """
    Performs evaluation in a single shot using globally preprocessed data.
    This is the fastest possible evaluation method, assuming sufficient VRAM.
    """
    log = defaultdict(list)
    hp = model.hp
    model.eval()

    with torch.no_grad():
        # 1. 一次前向传播，覆盖所有数据
        padded_y_hat = model(global_data['padded_inputs'])
        if hp['loss_type'] == 'lsq' and not hp['use_snn']:
            padded_y_hat = torch.sigmoid(padded_y_hat)
        else:
            padded_y_hat = F.softmax(padded_y_hat, dim=-1)

        # 2. 一次高级索引
        device = padded_y_hat.device
        total_trials = global_data['total_trials']
        y_hat_last_step = padded_y_hat[global_data['last_step_indices'], torch.arange(total_trials, device=device)]
        yloc_last_step = global_data['padded_ylocs'][global_data['last_step_indices'], torch.arange(total_trials, device=device)]
        
        # 3. 一次性能计算
        all_perf_values = get_perf_torch(y_hat_last_step, yloc_last_step.to(y_hat_last_step.dtype))
        
        # 4. 从总性能向量中切片，计算每个rule的均值
        for rule, (start, end) in global_data['rule_indices'].items():
            if rule in hp['rules']: # 确保只记录hp中指定的rules
                rule_perf = torch.mean(all_perf_values[start:end]).item()
                log['perf_' + rule].append(rule_perf)
    
    model.train()
    
    if hasattr(rule_train, '__iter__'): rule_tmp = rule_train
    else: rule_tmp = [rule_train]
    perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
    perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])

    log['perf_avg'].append(perf_tests_mean)
    log['perf_min'].append(perf_tests_min)
    if verbose: 
        print(f"avg  | perf {perf_tests_mean:0.3f}")
        print(f"min  | perf {perf_tests_min:0.3f}")
    return log


if __name__ == '__main__':
    import time
    lock_random_seed(2024)
    n_rnn = 32
    seed = 200 # 使用一个存在的种子以确保模型文件可被加载
    load_step = 12000
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = f'./runs/Fig2bcde_data/n_rnn_{n_rnn}_task_20_seed_{seed}/RNN_interleaved_learning_{load_step}.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the path is correct and the model file exists.")
        exit()

    trained_model = torch.load(model_path, map_location=device)

    hp = trained_model.hp
    hp['device'] = device
    
    test_set = task.Get_Testset(hp, data_dir='./datasets/multitask/test', n_rep=16)

    # --- 性能和结果比较 ---
    
    # NumPy 版本
    print("\nRunning original NumPy version...")
    start_time = time.time()
    log_np = do_eval_with_dataset(trained_model, \
        hp['rules'], dataset=test_set, verbose=True)
    end_time = time.time()
    time_np = end_time - start_time
    print(f"NumPy version took: {time_np:.4f} seconds")
    perf_avg_np = log_np['perf_avg'][0]

    # preprocess dataset (only do once) 
    global_data = task.preprocess_dataset_for_gpu_global(test_set,  hp['rules'], device)

    # Torch 版本 (使用填充)
    print("\nRunning optimized Torch version (with padding)...")
    start_time = time.time()
    log_torch_opt = do_eval_with_dataset_torch_fast(\
        trained_model, hp['rules'], global_data, verbose=True)
    end_time = time.time()
    time_torch_opt = end_time - start_time
    print(f"Optimized Torch version took: {time_torch_opt:.4f} seconds")
    perf_avg_torch_opt = log_torch_opt['perf_avg'][0]
    
    # --- 结果总结 ---
    print("\n" + "="*55)
    print(" " * 20 + "Summary")
    print("="*55)
    print(f"Performance (avg):")
    print(f"  - NumPy version          : {perf_avg_np:.6f}")
    print(f"  - Torch version  : {perf_avg_torch_opt:.6f}")
    print("\nExecution Time:")
    print(f"  - NumPy version          : {time_np:.4f} s")
    print(f"  - Torch version  : {time_torch_opt:.4f} s")
    print("\nSpeedup (Optimized vs NumPy)          : {:.2f}x".format(time_np / time_torch_opt))
    print("="*55)
    
    diff = abs(perf_avg_np - perf_avg_torch_opt)
    print(f"\nAbsolute difference between NumPy and Optimized Torch avg performance: {diff:.8f}")
    if diff < 1e-2:
        print("The outputs are consistent, the optimization is correct.")
    else:
        print("Warning: The outputs show a noticeable difference. Please check the logic.")