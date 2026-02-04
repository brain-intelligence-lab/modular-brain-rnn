from functions.generative_network_modelling.generative_network_modelling import *
from functions.utils.eval_utils import prepare_data_for_gnm, diagnostic_mle_fit
from functions.utils.math_utils import lock_random_seed, \
      compute_syn_loglik, compute_syn_loglik_with_pca, check_pca_variance
from statannotations.Annotator import Annotator
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd 
import pdb

load_step = 40000
multi_task_num = 20
num_of_people = 200
lock_random_seed(2026)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


num_of_seed = (num_of_people + multi_task_num - 1) // multi_task_num
GT, Distance, FC = prepare_data_for_gnm(load_step, num_of_seed=num_of_seed, device=device)

# Convert quantiles to tensor for GPU usage
QUANTILES_TENSOR = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=device, dtype=torch.float32)
d_dimensions = 4 * (len(QUANTILES_TENSOR) + 2)  # 4 statistics, each with 5 quantiles + mean + standard deviation

def get_batch_dist_stats(data_tensor, log_transform):
    """
    Input [Batch, K, N] or [Batch, K, N*(N-1)/2],
    K is 3 or 1, 3: degree, clustering, betweenness; 1: edge length
    N is the number of nodes in the graph, N*(N-1)/2 is the number of edges
    Output [Batch, 7, N] or [Batch, 7, N*(N-1)/2] (5 quantiles + mean + standard deviation)
    """
    if log_transform:
        data_tensor = torch.log1p(torch.clamp(data_tensor, min=0.0))

    # Calculate quantiles (Batch dimension) torch.quantile expects dimension at the end, or specify dim
    qs = torch.quantile(data_tensor, QUANTILES_TENSOR, dim=2).permute(1, 2, 0) # [Batch, K, 5]

    mu = data_tensor.mean(dim=2, keepdim=True) # [Batch, K, 1]
    sd = data_tensor.std(dim=2, unbiased=True, keepdim=True) # [Batch, K, 1]

    return torch.cat([qs, mu, sd], dim=2) # [Batch, K, 7]


def get_summary_stats_torch_batched(conn_matrix_batch, Distance, device, log_transform=True, directed=False):
    """
    Batch compute statistics, fully utilizing GPU parallel processing
    
    Args:
        conn_matrix_batch: [batch_size, n, n] adjacency matrix tensor, already on device
        Distance: [n, n] distance matrix
        device: device
        log_transform: whether to perform log transform
        directed: whether graph is directed
    
    Returns:
        summary_stats: [batch_size, d_dimensions] statistics tensor
    """
    # Batch compute graph distribution
    tgt = get_graph_distribution_batched(conn_matrix_batch, Distance, device, directed=directed)
    # tgt = [degrees, clustering, betweenness, edge_lengths_list]
    # degrees: [batch_size, n]
    # clustering: [batch_size, n]
    # betweenness: [batch_size, n]
    # edge_lengths_list: list of tensors, each is [num_edges_i]
    
    batch_size = conn_matrix_batch.shape[0]
    
    # Stack first three statistics: [batch_size, 3, n]
    stacked_input = torch.stack([tgt[0], tgt[1], tgt[2]], dim=1)  # [batch_size, 3, n]
    
    # Compute distribution statistics for first three statistics: [batch_size, 3, 7]
    deg_clust_betw_concat = get_batch_dist_stats(stacked_input, log_transform=log_transform)
    # shape: [batch_size, 3, 7]
    
    # Handle edge lengths (each graph may have different number of edges)
    # We need to compute separately for each graph because edge counts differ
    edge_len_stats_list = []
    for i in range(batch_size):
        edge_len_i = tgt[3][i]  # [num_edges_i]
        edge_len_tensor = edge_len_i.unsqueeze(0).unsqueeze(0)  # [1, 1, num_edges_i]
        edge_len_stats = get_batch_dist_stats(edge_len_tensor, log_transform=log_transform)  # [1, 1, 7]
        edge_len_stats_list.append(edge_len_stats.squeeze(0))  # [1, 7]
    
    edge_len_stats_batch = torch.stack(edge_len_stats_list, dim=0)  # [batch_size, 1, 7]
    
    # Merge all statistics: [batch_size, 4, 7]
    concat_tensor = torch.cat([deg_clust_betw_concat, edge_len_stats_batch], dim=1)  # [batch_size, 4, 7]
    
    # Flatten: [batch_size, d_dimensions]
    flattened_tensor = concat_tensor.view(batch_size, -1)
    
    return flattened_tensor


def fit_mle_fast_batch(s_obs_tensor, prob_matrix_batch, Distance, tot_conn_num, 
                       n_sims, device, directed=False, max_batch_size=None,
                       use_pca=True, n_pca_components=8): # <--- Can add switch in parameters:
    """
    Batch compute likelihood values for multiple (eta, gamma) combinations
    
    Args:
        s_obs_tensor: [d] observed statistics
        prob_matrix_batch: [n_params, n, n] probability matrix for multiple parameter combinations, n_params = len(eta_vec) * len(gamma_vec)
        Distance: distance matrix
        tot_conn_num: total number of connections
        n_sims: number of simulations per parameter combination
        directed: whether graph is directed
        device: device ('cuda:0' 等)
        max_batch_size: maximum batch size, if None then process all data at once
        use_pca: whether to use PCA for likelihood estimation
        n_pca_components: number of PCA components (if using PCA)
    
    Returns:
        lls: [n_params] likelihood value for each parameter combination
        s_sims_all: [n_params * n_sims, d] statistics of all simulations
    """
    n_params = prob_matrix_batch.shape[0]
    total_batch_size = n_params * n_sims
    
    # Set default maximum batch size (if not specified, use a reasonable default value)
    if max_batch_size is None:
        # Default: if total batch size exceeds 40000, process in batches
        max_batch_size = min(40000, total_batch_size)
    
    # If total batch size is less than or equal to max_batch_size, process all at once
    if total_batch_size <= max_batch_size:
        # 1. Replicate each prob_matrix n_sims times to form [n_params * n_sims, n, n]
        prob_matrix_expanded = prob_matrix_batch.repeat_interleave(n_sims, dim=0)
        
        # 2. Batch generate all graphs: [n_params * n_sims, n, n]
        adj_batch = gnm_batched(prob_matrix_expanded, tot_conn_num, 
                                directed=directed, device=device)
        
        # 3. Batch compute statistics (fully utilizing GPU parallel processing)
        s_sims_all = get_summary_stats_torch_batched(adj_batch, Distance, device, directed=directed)  # [n_params * n_sims, d]
    else:
        # Process in batches to avoid GPU memory overflow
        s_list = []
        
        # Calculate how many parameter combinations each small batch should process
        # Each parameter combination needs n_sims graphs, so each small batch can process max_batch_size // n_sims parameter combinations
        params_per_batch = max(1, max_batch_size // n_sims)
        
        for start_idx in range(0, n_params, params_per_batch):
            end_idx = min(start_idx + params_per_batch, n_params)
            
            # Extract prob_matrix for current batch
            prob_matrix_batch_subset = prob_matrix_batch[start_idx:end_idx]  # [batch_n_params, n, n]
            
            # Replicate each prob_matrix n_sims times
            prob_matrix_expanded = prob_matrix_batch_subset.repeat_interleave(n_sims, dim=0)
            
            # Batch generate graphs
            adj_batch = gnm_batched(prob_matrix_expanded, tot_conn_num, 
                                    directed=directed, device=device)
            
            # Batch compute statistics (fully utilizing GPU parallel processing)
            s_stats_batch = get_summary_stats_torch_batched(adj_batch, Distance, device, directed=directed)  # [batch_n_params * n_sims, d]
            s_list.append(s_stats_batch)
            
            # Clear GPU memory
            del adj_batch, prob_matrix_expanded, s_stats_batch
            if isinstance(device, torch.device) and device.type == 'cuda':
                torch.cuda.empty_cache()
            elif isinstance(device, str) and device.startswith('cuda'):
                torch.cuda.empty_cache()
        
        s_sims_all = torch.cat(s_list, dim=0)  # [n_params * n_sims, d]
    
    # 4. Calculate likelihood value for each parameter combination
    lls = []
    for i in range(n_params):
        s_sims_i = s_sims_all[i * n_sims:(i + 1) * n_sims]  # [n_sims, d]
        # Key fix: force use of diagonal covariance to avoid full covariance matrix instability
        # Sample covariance matrix estimation is extremely unstable when n_sims=200, d=d_dimensions!
        if use_pca:
            ll = compute_syn_loglik_with_pca(s_obs_tensor, s_sims_i, use_diag_cov=True,
                                              n_components=n_pca_components)
        else:
            ll = compute_syn_loglik(s_obs_tensor, s_sims_i, use_diag_cov=True)
        lls.append(ll)
    
    return torch.tensor(lls, device=device), s_sims_all.detach()


def plot_model_comparison(results_df):
    names_order = ['spatial', 'multitask', 'spatial_multitask']
    modelname_map = {'spatial': 'spatial', 'multitask': 'task', 'spatial_multitask': 'spatial+task'}
    
    df_plot = results_df.copy()
    df_plot['Model_Display'] = df_plot['Model'].map(modelname_map)
    display_order = [modelname_map[m] for m in names_order]
    palette = ['#2ca02c', '#ff7f0e', '#1f77b4'] 
    metrics = ['AIC', 'BIC']
    
    fig, axes = plt.subplots(1, 2, figsize=(3.8, 2.0), dpi=300)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.boxplot(x='Model_Display', y=metric, data=df_plot, ax=ax, order=display_order,
                    palette=palette, width=0.3, linewidth=0.25, boxprops=dict(facecolor='none'),
                    flierprops={'markersize': 1.0,      
                    'markeredgewidth': 0.25,  
                    'alpha': 0.5             
                    })
        sns.stripplot(x='Model_Display', y=metric, data=df_plot, ax=ax, order=display_order,
                      palette=palette, jitter=0.1, size=0.6, alpha=0.4)

        ax.set_yscale('log')
        
        # 1. Clearly obtain median and establish precise mapping from value to color
        medians = df_plot.groupby('Model_Display')[metric].median()
        # Must map color according to the order of display_order
        val_to_color = {medians[m]: palette[idx] for idx, m in enumerate(display_order)}
        median_values = list(val_to_color.keys())

        # 2. Plot median horizontal dashed lines
        for idx, model_name in enumerate(display_order):
            m_val = medians[model_name]
            ax.axhline(y=m_val, xmin=0, xmax=(idx + 0.5) / len(display_order), 
                       color=palette[idx], linestyle='--', linewidth=0.25, alpha=0.5)

        # 3. Handle tick conflicts: remove default ticks close to median
        plt.draw() # First trigger default tick generation
        default_ticks = ax.get_yticks()
        final_ticks = []
        for d_tick in default_ticks:
            if all(abs(np.log10(d_tick) - np.log10(m)) > 0.12 for m in median_values):
                final_ticks.append(d_tick)
        
        all_ticks = sorted(list(set(final_ticks) | set(median_values)))
        ax.set_yticks(all_ticks)
        
        # 4. [Key fix]: Manually set labels, no longer rely on Formatter, ensure consistency
        # For non-median ticks, display if power of 10, otherwise hide or display simply
        tick_labels = []
        for v in all_ticks:
            is_median = False
            for m_val in median_values:
                if abs(v - m_val) < 1e-2:
                    tick_labels.append(f"{v:.1f}") # Median keeps one decimal place
                    is_median = True
                    break
            if not is_median:
                # Default ticks display integer values only
                tick_labels.append(f"{int(v)}" if v >= 1 else f"{v:.1g}")
        
        ax.set_yticklabels(tick_labels)

        # 5. Add significance annotation (Annotator)
        box_pairs = [(display_order[0], display_order[1]), (display_order[1], display_order[2]), (display_order[0], display_order[2])]
        annotator = Annotator(ax, box_pairs, data=df_plot, x='Model_Display', y=metric, order=display_order)
        annotator.configure(test='t-test_ind', text_format='star', loc='inside', fontsize=5, line_width=0.5)
        annotator.apply_test().annotate()

        # 6. [Color coloring fix]: Directly traverse the Labels we just set
        plt.draw()
        labels = ax.get_yticklabels()
        for lbl in labels:
            txt = lbl.get_text().replace('−', '-')
            if not txt: continue
            try:
                val = float(txt)
                for m_val, color in val_to_color.items():
                    if abs(val - m_val) < 1e-1: # Tolerance matching
                        lbl.set_color(color)
                        lbl.set_fontweight('bold')
                        lbl.set_fontsize(5)
            except: continue

        # Fine-tune details
        ax.set_xlabel('Generative Model', fontsize=6)
        ax.set_ylabel(metric, fontsize=6)
        ax.tick_params(axis='both', labelsize=5, width=0.5, length=2)

        # Fine-tune axis spine width
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    plt.subplots_adjust(wspace=0.01)  
    plt.tight_layout()
    plt.savefig("Fig5_Model_Comparison_Statistical.png", dpi=300)
    plt.savefig("Fig5_Model_Comparison_Statistical.svg", dpi=300)


nruns = 900
grid_n = int(np.sqrt(nruns))
if grid_n % 2 == 0:
    grid_n += 1  # ensure 0 included
eta_vec = np.linspace(-3.0, 3.0, grid_n)
gamma_vec = np.linspace(-3.0, 3.0, grid_n)

# ... previous code unchanged ...

n_sims = 200
directed = False
# Control batch size to avoid GPU memory overflow, can be adjusted according to GPU memory size
# For example: if n_sims=200, max_batch_size=4000 means processing at most 20 parameter combinations at a time
# If set to None, process all data at once (may cause GPU memory shortage)
max_batch_size = 8000  # Can be adjusted according to GPU memory, None means no limit
# Store model comparison results for all subjects
comparison_records = []
use_pca = True
n_pca_components = 10

for p_idx in tqdm(range(num_of_people)):

    save_filename = f'Fig5_mle_fit_results_{p_idx}.npz'
    
    # --- New feature: check if file exists, skip calculation if exists ---
    if os.path.exists(save_filename):
        print(f"Loading existing results for person {p_idx} from {save_filename}...")
        data = np.load(save_filename, allow_pickle=True)
        # Get the saved model dictionary
        model_name_lls = data['model_name_lls'].item()
        
        # Add existing results to plotting records
        for m_name, info in model_name_lls.items():
            # Compatible with old format (if tuple, manually convert to dict to ensure code robustness)
            if isinstance(info, (list, tuple)):
                ll_val, params_val = info[0], info[1]
                # Calculate missing AIC/BIC
                n_samples = d_dimensions # Default feature dimension
                k_params = 2 if m_name == 'spatial_multitask' else 1
                aic_val = 2 * k_params - 2 * ll_val
                bic_val = k_params * np.log(n_samples) - 2 * ll_val
            else:
                ll_val = info['ll']
                aic_val = info['aic']
                bic_val = info['bic']

            comparison_records.append({
                'Subject': p_idx,
                'Model': m_name,
                'LL': ll_val,
                'AIC': aic_val,
                'BIC': bic_val
            })
        continue # Skip subsequent calculations
    
    # --- If file does not exist, execute original calculation logic ---
    print(f"Processing person {p_idx} (No cache found)...")


    # --- Optimization: calculate s_obs once, reuse 1000 times ---
    # Prepare A_obs
    A_obs_np = (GT[p_idx] > 0).astype(int)
    np.fill_diagonal(A_obs_np, 0)
    A_obs_np = np.maximum(A_obs_np, A_obs_np.T)
    A_obs_tensor = torch.tensor(A_obs_np).unsqueeze(0)
    tot_conn_num = int(A_obs_tensor.sum())
    
    # Calculate Observed Summary Vector (GPU)
    s_obs_tensor = get_summary_stats_torch_batched(A_obs_tensor, Distance, device).squeeze()
    # ----------------------------------------
    # Sample size n needed for BIC calculation
    n_samples_for_bic = s_obs_tensor.shape[0]
    model_name_lls = {}
    
    # Prepare K matrix
    K_curr = FC[p_idx]

    # Pre-convert Distance and K to tensors
    D_tensor = torch.tensor(Distance, device=device, dtype=torch.float32)
    K_tensor = torch.tensor(K_curr, device=device, dtype=torch.float32)
    n = D_tensor.shape[0]
    
    for model_name in ['spatial', 'multitask', 'spatial_multitask']:
        best_ll = -np.inf
        best_params = (-10.0, -10.0)
        
        # Build parameter grid
        if model_name == 'spatial':
            loop_grid = [(e, 0.0) for e in eta_vec]
            k_params = 1
            desc = "Spatial"
        elif model_name == 'multitask':
            loop_grid = [(0.0, g) for g in gamma_vec]
            k_params = 1
            desc = "Multitask"
        else:
            import itertools
            loop_grid = list[tuple](itertools.product(eta_vec, gamma_vec))
            k_params = 2
            desc = "Spatial-Multitask"

        # Batch compute prob_matrix for all parameter combinations
        prob_matrix_list = []
        for e, g in loop_grid:
            # Handle parameter zeroing for corresponding model type
            curr_eta = float(e) if model_name != 'multitask' else 0.0
            curr_gamma = float(g) if model_name != 'spatial' else 0.0
            
            # Calculate probability matrix (gnm_batched will normalize internally)
            # Note: epsilon value is consistent with original gnm (1e-5)
            Fd = torch.pow(D_tensor + 1e-5, curr_eta)
            Fk = torch.pow(K_tensor + 1e-5, curr_gamma)
            prob_matrix = Fd * Fk
            prob_matrix_list.append(prob_matrix)
        
        # Stack into batch: [n_params, n, n]
        prob_matrix_batch = torch.stack(prob_matrix_list, dim=0)
        
        # Batch compute likelihood values for all parameter combinations
        lls, s_sims_all = fit_mle_fast_batch(
            s_obs_tensor=s_obs_tensor,
            prob_matrix_batch=prob_matrix_batch,
            Distance=Distance,
            tot_conn_num=tot_conn_num,
            n_sims=n_sims,
            device=device,
            directed=directed,
            max_batch_size=max_batch_size,
            use_pca=use_pca, 
            n_pca_components=n_pca_components,
        )
        
        # Find best parameters
        best_idx = torch.argmax(lls).item()
        best_ll = float(lls[best_idx])
        best_params = loop_grid[best_idx]
        
        # check_pca_variance(s_sims_all)
        # DEBUG: print top 5 and bottom 5 parameters and their LL values
        # top5_indices = torch.topk(lls, 5).indices
        # bottom5_indices = torch.topk(lls, 5, largest=False).indices
        # print(f"\n  {model_name} Top 5 params:")
        # for idx in top5_indices:
        #     print(f"    {loop_grid[idx.item()]}: LL={lls[idx]:.2f}")
        
        if use_pca:
            n_samples_for_bic = n_pca_components

        # Calculate AIC and BIC
        # AIC = 2k - 2ln(L)
        # BIC = k*ln(n) - 2ln(L)
        aic = 2 * k_params - 2 * best_ll
        bic = k_params * np.log(n_samples_for_bic) - 2 * best_ll

        model_name_lls[model_name] = {
            'll': best_ll, 
            'params': best_params, 
            'aic': aic, 
            'bic': bic
        }
        
        # Save to list for later plotting
        comparison_records.append({
            'Subject': p_idx,
            'Model': model_name,
            'LL': best_ll,
            'AIC': aic,
            'BIC': bic
        })



        # --- Run Diagnostic Check ---
        # Extract s_sims corresponding to best parameters (original space data)
        # start = best_idx * n_sims
        # end = (best_idx + 1) * n_sims
        # best_s_sims_raw = s_sims_all[start:end] 
        
        # if use_pca:
        #     # ---------------------------------------------------------
        #     # Perform PCA projection on best simulated data for diagnostics
        #     # We need to verify whether data follows normal distribution in PCA space
        #     # ---------------------------------------------------------
        #     # 1. Normailzation (consistent with fit_mle_fast_batch)
        #     mu = best_s_sims_raw.mean(dim=0)
        #     std = best_s_sims_raw.std(dim=0, unbiased=True) + 1e-6
        #     best_s_sims_norm = (best_s_sims_raw - mu) / std
            
        #     # 2. Calculate PCA Projection
        #     # Note: Although re-computing SVD here, for diagnostic purposes,
        #     # verifying "whether there exists a linear transformation that normalizes the data" is sufficient.
        #     U, S, Vh = torch.linalg.svd(best_s_sims_norm, full_matrices=False)
            
        #     # 3. Projection to the first n_features_effective-dimension
        #     # First few rows of Vh are principal component directions
        #     components = Vh[:n_pca_components, :].T
        #     best_s_sims_pca = best_s_sims_norm @ components # [n_sims, 8]
            
        #     # Convert back to numpy for plotting
        #     best_s_sims_for_plot = best_s_sims_pca.cpu().numpy()
            
        #     # Generate corresponding label names
        #     pca_feat_names = [f"PC_{i+1}" for i in range(n_pca_components)]
        # else:
        #     best_s_sims_for_plot = best_s_sims_raw.cpu().numpy()
        #     pca_feat_names = None
    
        # print(f"Running diagnostic for Subject {p_idx}, Model {model_name}...")
        # diagnostic_mle_fit(
        #     best_s_sims_for_plot, 
        #     feature_names=pca_feat_names,  # sendin label
        #     save_path=f"diagnostic_subject_{p_idx}_{model_name}_PCA.png"
        # )

        print(f"  > {model_name} Best LL: {best_ll:.2f} @ {best_params}")

        # best_sims_np = best_s_sims # [200, d_dimensions]
        # # Calculate correlation coefficient matrix
        # corr_matrix = np.corrcoef(best_sims_np, rowvar=False)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        # plt.title(f"Feature Correlation Matrix (Subject {p_idx})")
        # plt.savefig('Fig5_Feature_Corr_Subject_{}_{}.png'.format(p_idx, model_name), dpi=300)

    # save results
    np.savez(save_filename, model_name_lls=model_name_lls)

df_results = pd.DataFrame(comparison_records)
plot_model_comparison(df_results)