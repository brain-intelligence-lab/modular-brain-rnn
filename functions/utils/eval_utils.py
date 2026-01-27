import torch
import torch.nn.functional as F
import numpy as np
import datasets.multitask as task
from collections import defaultdict
import bct
import os


def calculate_modularity_in_r(weight_matrix: np.ndarray, r_script_path:str, verbose:bool=False):
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri, default_converter, conversion

    assert os.path.exists(r_script_path), f"R script '{r_script_path}' not found."

    np_cv_rules = default_converter + numpy2ri.converter
    with conversion.localconverter(np_cv_rules):
        r = robjects.r

        r.source(r_script_path) # Load R script
        lpa_wb_plus_func = robjects.globalenv['LPA_wb_plus']
        dirt_lpa_func = robjects.globalenv['DIRT_LPA_wb_plus']
        r_matrix = robjects.conversion.py2rpy(weight_matrix) # Convert to R matrix

        # Call R function
        if verbose:
            print(" Calling LPA_wb_plus(MAT)...") 
        mod1_result = lpa_wb_plus_func(r_matrix)
        
        if verbose:
            print(" Calling DIRT_LPA_wb_plus(MAT)...")
        mod2_result = dirt_lpa_func(r_matrix)


    modularity1 = mod1_result[0][0]
    modularity2 = mod2_result[0][0]

    if verbose:
        print(f" Modularity results: LPA_wb_plus = {modularity1}, DIRT_LPA_wb_plus = {modularity2}")
    
    return modularity1, modularity2


def calculate_modularity_for_fc_layer(weight_in_matrix: np.ndarray, weight_out_matrix:str, verbose:bool=False):
    # weight_in_matrix: n * m, weight_out_matrix: m * q

    feature_matrix_1 = np.matmul(weight_in_matrix.T, weight_in_matrix)

    feature_matrix_2 = np.matmul(weight_out_matrix, weight_out_matrix.T)

    ci1, modularity1 = bct.modularity_dir(np.abs(feature_matrix_1))
    ci2, modularity2 = bct.modularity_dir(np.abs(feature_matrix_2))

    if verbose:
        print(f"mod1:{modularity1}, mod2:{modularity2}")
    
    return modularity1, modularity2


class ActivationHook:
    def __init__(self):
        self.activations = []

    def __call__(self, module, input, output):
        """
        The hook function executed when `readout` is called.
        Input[0] is the (T, B, H) hidden_states tensor.
        """
        # We need to detach and clone to avoid issues with the computation graph
        # and to store the tensor outside of the hook's scope.
        self.activations.append(input[0].detach().clone())


def get_hidden_states(model, device):
    hp = model.hp
    hidden_states_list = []
    def hook(module, input, output):
        input, = input
        # input.shape: T * Batch_size * N
        hidden_states_list.append(input.detach().mean(dim=(1)))
        
    handle = model.readout.register_forward_hook(hook)
    for rule in hp['rule_trains']:
        trial = task.generate_trials(
            rule, hp, 'random',
            batch_size=512)

        input = torch.from_numpy(trial.x).to(device)
        output = model(input)

    handle.remove()
    return hidden_states_list


def prepare_data_for_gnm(load_step, num_of_seed:int, device, path='./datasets/brain_hcp_data/84/structureM_use.mat'):
    # Define cache filename including key parameters to prevent reading wrong data when parameters change
    cache_dir = './datasets/brain_hcp_data/84/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'gnm_prepared_data_step{load_step}_seed{num_of_seed}.npz')

    # 1. Check if cache file exists
    if os.path.exists(cache_path):
        print(f"Loading prepared GNM data from cache: {cache_path}")
        with np.load(cache_path, allow_pickle=True) as data:
            subjects_conn_data = data['GT']
            Distance = data['Distance']
            Fc = data['FC'] # Convert from ndarray back to list
        return subjects_conn_data, Distance, Fc

    print(f"No cache found. Processing raw data (load_step={load_step}, seeds={num_of_seed})...")

    import scipy.io
    from tqdm import tqdm
    subjects_conn_data = scipy.io.loadmat(path)['structureM_use']
    subjects_conn_data = np.transpose(subjects_conn_data, [2, 0, 1])
    subjects_conn_data = subjects_conn_data.astype(np.float32)
    Distance = np.load('./datasets/brain_hcp_data/84/Raw_dis.npy')

    Fc = []
    for seed in tqdm(range(1, num_of_seed+1)):

        file_name = f'./runs/Fig5_data/84/n_rnn_84_task_20_seed_{seed}/RNN_interleaved_learning_{load_step}.pth'
        model = torch.load(file_name, device)   

        hidden_states_list = get_hidden_states(model, device)
        for task_id in range(20):
            hidden_states = hidden_states_list[task_id]
            random_value = torch.rand_like(hidden_states) 
            hidden_states =  hidden_states + random_value * 1e-7

            hidden_states_mean = hidden_states.detach().cpu().numpy()
            fc = np.corrcoef(hidden_states_mean, rowvar=False)
            fc[fc < 0] = 1e-5
            np.fill_diagonal(fc, 0)
            Fc.append(fc)

    np.savez(cache_path, GT=subjects_conn_data, Distance=Distance, FC=np.array(Fc))
    print(f"Data saved to cache: {cache_path}")
        
    return subjects_conn_data, Distance, Fc


def diagnostic_mle_fit(s_sims, feature_names=None, save_path="diagnostic_mle_fit.png"):
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import pingouin as pg
    import seaborn as sns
    """
    Judge whether simulation features s_sims follow Gaussian distribution assumption.
    s_sims: np.array, shape [n_sims, d], d is 28
    """
    n_sims, d = s_sims.shape
    if feature_names is None:
        feature_names = [f"Feat_{i}" for i in range(d)]

    print(f"--- Diagnostic Report (Sample size: {n_sims}, Dimension: {d}) ---")

    # 1. Univariate skewness and kurtosis detection (ideal values should be close to 0)
    skewness = stats.skew(s_sims)
    kurtosis = stats.kurtosis(s_sims)

    # 2. Multivariate normality test (Henze-Zirkler Test)
    # Add timeout protection to prevent hanging
    try:
        import signal

        def timeout_handler(_signum=None, _frame=None):
            raise TimeoutError("HZ test timeout")

        # Set 30 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

        hz_result = pg.multivariate_normality(s_sims)
        signal.alarm(0)  # Cancel timeout

        print(f"\nMultivariate normality test (Henze-Zirkler):")
        print(hz_result)
    except (TimeoutError, RuntimeError, Exception) as e:
        print(f"\n⚠️  HZ test failed: {str(e)[:50]}... (possibly due to insufficient samples or singular covariance matrix)")
        print("   Suggestion: Increase n_sims or use diagonal covariance assumption")

    # 3. Calculate Mahalanobis Distance
    # Theoretically, for data following multivariate normal distribution, squared Mahalanobis distance should follow chi-square distribution chi2(df=d)
    mu = np.mean(s_sims, axis=0)
    cov = np.cov(s_sims, rowvar=False) + 1e-6 * np.eye(d)
    inv_cov = np.linalg.inv(cov)
    
    diff = s_sims - mu
    md_squared = np.sum(diff @ inv_cov * diff, axis=1)

    # 4. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Figure A: Skewness vs Kurtosis scatter plot (evaluate outlier dimensions)
    axes[0].scatter(skewness, kurtosis, alpha=0.6)
    axes[0].axhline(0, color='r', linestyle='--')
    axes[0].axvline(0, color='r', linestyle='--')
    axes[0].set_title("Skewness vs Kurtosis per Dimension")
    axes[0].set_xlabel("Skewness")
    axes[0].set_ylabel("Kurtosis")

    # Figure B: Mahalanobis distance vs chi-square distribution (Q-Q Plot concept)
    # Theoretical quantiles
    theoretical_qs = stats.chi2.ppf(np.linspace(0.01, 0.99, n_sims), df=d)
    sorted_md = np.sort(md_squared)
    axes[1].scatter(theoretical_qs, sorted_md, alpha=0.6)
    axes[1].plot([theoretical_qs.min(), theoretical_qs.max()], 
                 [theoretical_qs.min(), theoretical_qs.max()], 'r--')
    axes[1].set_title("MD^2 vs Chi-Square Quantiles")
    axes[1].set_xlabel("Theoretical Chi2 Quantiles")
    axes[1].set_ylabel("Empirical MD^2")

    # Figure C: Correlation heatmap (check feature redundancy)
    sns.heatmap(np.corrcoef(s_sims, rowvar=False), ax=axes[2], cmap='coolwarm', center=0)
    axes[2].set_title("Feature Correlation Matrix")

    plt.tight_layout()
    plt.savefig(save_path)

    plt.close(fig)
    plt.close('all') # Double insurance, clear all currently unclosed plotting objects
    del fig, axes

    return {"skew": skewness, "kurt": kurtosis, "hz": hz_result if 'hz_result' in locals() else None}


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
    Runs on specified device (CPU or GPU).

    Args:
        y: Population output. PyTorch Tensor (Batch, Units)

    Returns:
        Readout locations: PyTorch Tensor (Batch,)
    """
    # Ensure pref tensor and input y are on the same device
    device = y.device
    pref = torch.arange(0, 2 * torch.pi, 2 * torch.pi / y.shape[-1], device=device)  # preferences
    
    temp_sum = y.sum(axis=-1)
    # Use PyTorch functions
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
    Runs on specified device (CPU or GPU).

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
    y_hat_loc = popvec_torch(y_hat[..., 1:]) # Call PyTorch version of popvec

    # In PyTorch, boolean operation results are BoolTensor, need to convert to float for mathematical operations
    fixating = (y_hat_fix > 0.5).float()

    original_dist = y_loc - y_hat_loc
    dist = torch.minimum(torch.abs(original_dist), 2 * torch.pi - torch.abs(original_dist))
    corr_loc = (dist < 0.2 * torch.pi).float()

    # Should fixate?
    should_fix = (y_loc < 0).float()

    # Performance
    # (1-should_fix) corresponds to (y_loc >= 0) case
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
# Ultimate solution: Global single-shot fast evaluation
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
        # 1. Single forward pass covering all data
        padded_y_hat = model(global_data['padded_inputs'])
        if hp['loss_type'] == 'lsq' and not hp['use_snn']:
            padded_y_hat = torch.sigmoid(padded_y_hat)
        else:
            padded_y_hat = F.softmax(padded_y_hat, dim=-1)

        # 2. Single advanced indexing
        device = padded_y_hat.device
        total_trials = global_data['total_trials']
        y_hat_last_step = padded_y_hat[global_data['last_step_indices'], torch.arange(total_trials, device=device)]
        yloc_last_step = global_data['padded_ylocs'][global_data['last_step_indices'], torch.arange(total_trials, device=device)]

        # 3. Single performance calculation
        all_perf_values = get_perf_torch(y_hat_last_step, yloc_last_step.to(y_hat_last_step.dtype))

        # 4. Slice from total performance vector and calculate mean for each rule
        for rule, (start, end) in global_data['rule_indices'].items():
            if rule in hp['rules']: # Ensure only rules specified in hp are recorded
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
    from math_utils import lock_random_seed
    lock_random_seed(2024)
    n_rnn = 32
    seed = 200 # Use an existing seed to ensure model file can be loaded
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

    # --- Performance and result comparison ---

    # NumPy version
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

    # Torch version (with padding)
    print("\nRunning optimized Torch version (with padding)...")
    start_time = time.time()
    log_torch_opt = do_eval_with_dataset_torch_fast(\
        trained_model, hp['rules'], global_data, verbose=True)
    end_time = time.time()
    time_torch_opt = end_time - start_time
    print(f"Optimized Torch version took: {time_torch_opt:.4f} seconds")
    perf_avg_torch_opt = log_torch_opt['perf_avg'][0]

    # --- Result summary ---
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