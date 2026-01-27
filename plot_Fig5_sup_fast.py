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

# 将 quantiles 转为 tensor 以便在 GPU 使用
QUANTILES_TENSOR = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=device, dtype=torch.float32)
d_dimensions = 4 * (len(QUANTILES_TENSOR) + 2)  # 4 个统计量，每个统计量 5 个分位数 + 均值 + 标准差

def get_batch_dist_stats(data_tensor, log_transform):
    """
    输入 [Batch, K, N]，或者[Batch, K, N*(N-1)/2]，
    K 是3或者1，3: degree, clustering, betweenness; 1: edge length
    N 是图的节点数，N*(N-1)/2 是图的边数
    输出 [Batch, 7, N] or [Batch, 7, N*(N-1)/2] (5个分位数 + 均值 + 标准差)
    """
    if log_transform:
        data_tensor = torch.log1p(torch.clamp(data_tensor, min=0.0))
    
    # 计算分位数 (Batch 维度) torch.quantile 期望维度在最后，或者指定 dim
    qs = torch.quantile(data_tensor, QUANTILES_TENSOR, dim=2).permute(1, 2, 0) # [Batch, K, 5]
    
    mu = data_tensor.mean(dim=2, keepdim=True) # [Batch, K, 1]
    sd = data_tensor.std(dim=2, unbiased=True, keepdim=True) # [Batch, K, 1]
    
    return torch.cat([qs, mu, sd], dim=2) # [Batch, K, 7]


def get_summary_stats_torch_batched(conn_matrix_batch, Distance, device, log_transform=True, directed=False):
    """
    批量计算统计量，充分利用 GPU 并行处理
    
    参数:
        conn_matrix_batch: [batch_size, n, n] 的邻接矩阵 tensor，已经在 device 上
        Distance: [n, n] 的距离矩阵
        device: 设备
        log_transform: 是否进行 log 变换
        directed: 是否为有向图
    
    返回:
        summary_stats: [batch_size, d_dimensions] 的统计量 tensor
    """
    # 批量计算图分布
    tgt = get_graph_distribution_batched(conn_matrix_batch, Distance, device, directed=directed)
    # tgt = [degrees, clustering, betweenness, edge_lengths_list]
    # degrees: [batch_size, n]
    # clustering: [batch_size, n]
    # betweenness: [batch_size, n]
    # edge_lengths_list: list of tensors，每个是 [num_edges_i]
    
    batch_size = conn_matrix_batch.shape[0]
    
    # 堆叠前三个统计量: [batch_size, 3, n]
    stacked_input = torch.stack([tgt[0], tgt[1], tgt[2]], dim=1)  # [batch_size, 3, n]
    
    # 计算前三个统计量的分布统计: [batch_size, 3, 7]
    deg_clust_betw_concat = get_batch_dist_stats(stacked_input, log_transform=log_transform)
    # shape: [batch_size, 3, 7]
    
    # 处理边长度（每个图的边数可能不同）
    # 我们需要对每个图分别计算，因为边数不同
    edge_len_stats_list = []
    for i in range(batch_size):
        edge_len_i = tgt[3][i]  # [num_edges_i]
        edge_len_tensor = edge_len_i.unsqueeze(0).unsqueeze(0)  # [1, 1, num_edges_i]
        edge_len_stats = get_batch_dist_stats(edge_len_tensor, log_transform=log_transform)  # [1, 1, 7]
        edge_len_stats_list.append(edge_len_stats.squeeze(0))  # [1, 7]
    
    edge_len_stats_batch = torch.stack(edge_len_stats_list, dim=0)  # [batch_size, 1, 7]
    
    # 合并所有统计量: [batch_size, 4, 7]
    concat_tensor = torch.cat([deg_clust_betw_concat, edge_len_stats_batch], dim=1)  # [batch_size, 4, 7]
    
    # 展平: [batch_size, d_dimensions]
    flattened_tensor = concat_tensor.view(batch_size, -1)
    
    return flattened_tensor


def fit_mle_fast_batch(s_obs_tensor, prob_matrix_batch, Distance, tot_conn_num, 
                       n_sims, device, directed=False, max_batch_size=None,
                       use_pca=True, n_pca_components=8): # <--- 可以在参数里加开关:
    """
    批量计算多个 (eta, gamma) 组合的似然值
    
    参数:
        s_obs_tensor: [d] 观测统计量
        prob_matrix_batch: [n_params, n, n] 多个参数组合的概率矩阵，n_params = len(eta_vec) * len(gamma_vec)
        Distance: 距离矩阵
        tot_conn_num: 总连接数
        n_sims: 每个参数组合的模拟次数
        directed: 是否为有向图
        device: 设备 ('cuda:0' 等)
        max_batch_size: 最大批次大小，如果 None 则一次性处理所有数据
        use_pca: 是否使用 PCA 进行似然估计
        n_pca_components: PCA 组件数量（如果使用 PCA）
    
    返回:
        lls: [n_params] 每个参数组合的似然值
        s_sims_all: [n_params * n_sims, d] 所有模拟的统计量
    """
    n_params = prob_matrix_batch.shape[0]
    total_batch_size = n_params * n_sims
    
    # 设置默认的最大批次大小（如果未指定，使用一个合理的默认值）
    if max_batch_size is None:
        # 默认值：如果总批次大小超过 40000，则分批处理
        max_batch_size = min(40000, total_batch_size)
    
    # 如果总批次大小小于等于 max_batch_size，一次性处理
    if total_batch_size <= max_batch_size:
        # 1. 将每个 prob_matrix 复制 n_sims 次，形成 [n_params * n_sims, n, n]
        prob_matrix_expanded = prob_matrix_batch.repeat_interleave(n_sims, dim=0)
        
        # 2. 批量生成所有图: [n_params * n_sims, n, n]
        adj_batch = gnm_batched(prob_matrix_expanded, tot_conn_num, 
                                directed=directed, device=device)
        
        # 3. 批量计算统计量（充分利用 GPU 并行处理）
        s_sims_all = get_summary_stats_torch_batched(adj_batch, Distance, device, directed=directed)  # [n_params * n_sims, d]
    else:
        # 分批处理以避免显存溢出
        s_list = []
        
        # 计算每个小批次应该处理多少个参数组合
        # 每个参数组合需要 n_sims 个图，所以每个小批次可以处理 max_batch_size // n_sims 个参数组合
        params_per_batch = max(1, max_batch_size // n_sims)
        
        for start_idx in range(0, n_params, params_per_batch):
            end_idx = min(start_idx + params_per_batch, n_params)
            
            # 提取当前批次的 prob_matrix
            prob_matrix_batch_subset = prob_matrix_batch[start_idx:end_idx]  # [batch_n_params, n, n]
            
            # 将每个 prob_matrix 复制 n_sims 次
            prob_matrix_expanded = prob_matrix_batch_subset.repeat_interleave(n_sims, dim=0)
            
            # 批量生成图
            adj_batch = gnm_batched(prob_matrix_expanded, tot_conn_num, 
                                    directed=directed, device=device)
            
            # 批量计算统计量（充分利用 GPU 并行处理）
            s_stats_batch = get_summary_stats_torch_batched(adj_batch, Distance, device, directed=directed)  # [batch_n_params * n_sims, d]
            s_list.append(s_stats_batch)
            
            # 清理显存
            del adj_batch, prob_matrix_expanded, s_stats_batch
            if isinstance(device, torch.device) and device.type == 'cuda':
                torch.cuda.empty_cache()
            elif isinstance(device, str) and device.startswith('cuda'):
                torch.cuda.empty_cache()
        
        s_sims_all = torch.cat(s_list, dim=0)  # [n_params * n_sims, d]
    
    # 4. 对每个参数组合计算似然值
    lls = []
    for i in range(n_params):
        s_sims_i = s_sims_all[i * n_sims:(i + 1) * n_sims]  # [n_sims, d]
        # 关键修复：强制使用对角协方差，避免全协方差矩阵不稳定
        # n_sims=200, d=d_dimensions 时样本协方差矩阵估计极其不稳定！
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
        
        # 1. 明确获取中位数并建立“数值->颜色”的精确映射
        medians = df_plot.groupby('Model_Display')[metric].median()
        # 必须按照 display_order 的顺序映射颜色
        val_to_color = {medians[m]: palette[idx] for idx, m in enumerate(display_order)}
        median_values = list(val_to_color.keys())

        # 2. 绘制中位数水平虚线
        for idx, model_name in enumerate(display_order):
            m_val = medians[model_name]
            ax.axhline(y=m_val, xmin=0, xmax=(idx + 0.5) / len(display_order), 
                       color=palette[idx], linestyle='--', linewidth=0.25, alpha=0.5)

        # 3. 处理刻度冲突：移除靠近中位数的默认刻度
        plt.draw() # 先触发一次默认刻度生成
        default_ticks = ax.get_yticks()
        final_ticks = []
        for d_tick in default_ticks:
            if all(abs(np.log10(d_tick) - np.log10(m)) > 0.12 for m in median_values):
                final_ticks.append(d_tick)
        
        all_ticks = sorted(list(set(final_ticks) | set(median_values)))
        ax.set_yticks(all_ticks)
        
        # 4. 【关键修复】：手动设置标签，不再依赖 Formatter，确保匹配一致性
        # 对非中位数的刻度，如果是10的整数幂则显示，否则隐藏或简单显示
        tick_labels = []
        for v in all_ticks:
            is_median = False
            for m_val in median_values:
                if abs(v - m_val) < 1e-2:
                    tick_labels.append(f"{v:.1f}") # 中位数保留一位小数
                    is_median = True
                    break
            if not is_median:
                # 默认刻度只显示整数
                tick_labels.append(f"{int(v)}" if v >= 1 else f"{v:.1g}")
        
        ax.set_yticklabels(tick_labels)

        # 5. 添加显著性标注 (Annotator)
        box_pairs = [(display_order[0], display_order[1]), (display_order[1], display_order[2]), (display_order[0], display_order[2])]
        annotator = Annotator(ax, box_pairs, data=df_plot, x='Model_Display', y=metric, order=display_order)
        annotator.configure(test='t-test_ind', text_format='star', loc='inside', fontsize=5, line_width=0.5)
        annotator.apply_test().annotate()

        # 6. 【颜色着色修复】：直接遍历刚才设置好的 Labels
        plt.draw()
        labels = ax.get_yticklabels()
        for lbl in labels:
            txt = lbl.get_text().replace('−', '-')
            if not txt: continue
            try:
                val = float(txt)
                for m_val, color in val_to_color.items():
                    if abs(val - m_val) < 1e-1: # 容差匹配
                        lbl.set_color(color)
                        lbl.set_fontweight('bold')
                        lbl.set_fontsize(5)
            except: continue

        # 细节微调
        ax.set_xlabel('Generative Model', fontsize=6)
        ax.set_ylabel(metric, fontsize=6)
        ax.tick_params(axis='both', labelsize=5, width=0.5, length=2)

        # 调细坐标轴边框
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

# ... 前面的代码不变 ...

n_sims = 200
directed = False
# 控制批次大小以避免显存溢出，可以根据 GPU 显存大小调整
# 例如：如果 n_sims=200，max_batch_size=4000 意味着每次最多处理 20 个参数组合
# 如果设置为 None，则一次性处理所有数据（可能显存不足）
max_batch_size = 8000  # 可以根据显存情况调整，None 表示不限制
# 存储所有人的模型比较结果
comparison_records = []
use_pca = True
n_pca_components = 10

for p_idx in tqdm(range(num_of_people)):

    save_filename = f'Fig5_mle_fit_results_{p_idx}.npz'
    
    # --- 新增功能：检查文件是否存在，如果存在则跳过计算 ---
    if os.path.exists(save_filename):
        print(f"Loading existing results for person {p_idx} from {save_filename}...")
        data = np.load(save_filename, allow_pickle=True)
        # 获取保存的模型字典
        model_name_lls = data['model_name_lls'].item()
        
        # 将已有的结果加入到绘图记录中
        for m_name, info in model_name_lls.items():
            # 兼容旧格式（如果是元组则手动转为字典，确保代码鲁棒性）
            if isinstance(info, (list, tuple)):
                ll_val, params_val = info[0], info[1]
                # 计算缺失的 AIC/BIC
                n_samples = d_dimensions # 默认特征维度
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
        continue # 跳过后续计算
    
    # --- 如果文件不存在，执行原来的计算逻辑 ---
    print(f"Processing person {p_idx} (No cache found)...")


    # --- 优化点：计算一次 s_obs，复用 1000 次 ---
    # 准备 A_obs
    A_obs_np = (GT[p_idx] > 0).astype(int)
    np.fill_diagonal(A_obs_np, 0)
    A_obs_np = np.maximum(A_obs_np, A_obs_np.T)
    A_obs_tensor = torch.tensor(A_obs_np).unsqueeze(0)
    tot_conn_num = int(A_obs_tensor.sum())
    
    # 计算 Observed Summary Vector (GPU)
    s_obs_tensor = get_summary_stats_torch_batched(A_obs_tensor, Distance, device).squeeze()
    # ----------------------------------------
    # BIC 计算需要的样本量 n
    n_samples_for_bic = s_obs_tensor.shape[0]
    model_name_lls = {}
    
    # 准备 K 矩阵
    K_curr = FC[p_idx]

    # 预先将 Distance 和 K 转为 tensor
    D_tensor = torch.tensor(Distance, device=device, dtype=torch.float32)
    K_tensor = torch.tensor(K_curr, device=device, dtype=torch.float32)
    n = D_tensor.shape[0]
    
    for model_name in ['spatial', 'multitask', 'spatial_multitask']:
        best_ll = -np.inf
        best_params = (-10.0, -10.0)
        
        # 构建参数网格
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

        # 批量计算所有参数组合的 prob_matrix
        prob_matrix_list = []
        for e, g in loop_grid:
            # 处理模型类型对应的参数归零
            curr_eta = float(e) if model_name != 'multitask' else 0.0
            curr_gamma = float(g) if model_name != 'spatial' else 0.0
            
            # 计算概率矩阵（gnm_batched 内部会归一化）
            # 注意：epsilon 值与原始 gnm 保持一致 (1e-5)
            Fd = torch.pow(D_tensor + 1e-5, curr_eta)
            Fk = torch.pow(K_tensor + 1e-5, curr_gamma)
            prob_matrix = Fd * Fk
            prob_matrix_list.append(prob_matrix)
        
        # 堆叠成 batch: [n_params, n, n]
        prob_matrix_batch = torch.stack(prob_matrix_list, dim=0)
        
        # 批量计算所有参数组合的似然值
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
        
        # 找到最佳参数
        best_idx = torch.argmax(lls).item()
        best_ll = float(lls[best_idx])
        best_params = loop_grid[best_idx]
        
        # check_pca_variance(s_sims_all)
        # DEBUG: 打印前5个最佳和最后5个参数及其LL值
        # top5_indices = torch.topk(lls, 5).indices
        # bottom5_indices = torch.topk(lls, 5, largest=False).indices
        # print(f"\n  {model_name} Top 5 params:")
        # for idx in top5_indices:
        #     print(f"    {loop_grid[idx.item()]}: LL={lls[idx]:.2f}")
        
        if use_pca:
            n_samples_for_bic = n_pca_components

        # 计算 AIC 和 BIC
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
        
        # 保存到列表以便后续绘图
        comparison_records.append({
            'Subject': p_idx,
            'Model': model_name,
            'LL': best_ll,
            'AIC': aic,
            'BIC': bic
        })



        # --- 运行 Diagnostic Check ---
        # 提取最佳参数对应的 s_sims (原始空间数据)
        # start = best_idx * n_sims
        # end = (best_idx + 1) * n_sims
        # best_s_sims_raw = s_sims_all[start:end] 
        
        # if use_pca:
        #     # ---------------------------------------------------------
        #     # 对最佳模拟数据执行 PCA 投影，以便进行诊断
        #     # 我们需要验证的是：在 PCA 空间中，数据是否服从正态分布
        #     # ---------------------------------------------------------
        #     # 1. 标准化 (与 fit_mle_fast_batch 逻辑一致)
        #     mu = best_s_sims_raw.mean(dim=0)
        #     std = best_s_sims_raw.std(dim=0, unbiased=True) + 1e-6
        #     best_s_sims_norm = (best_s_sims_raw - mu) / std
            
        #     # 2. 计算 PCA 投影
        #     # 注意：虽然这里是重新计算 SVD，但对于诊断目的，
        #     # 验证"是否存在一个线性变换使得数据正态化"是足够的。
        #     U, S, Vh = torch.linalg.svd(best_s_sims_norm, full_matrices=False)
            
        #     # 3. 投影到前 n_features_effective 个维度
        #     # Vh 的前几行是主成分方向
        #     components = Vh[:n_pca_components, :].T
        #     best_s_sims_pca = best_s_sims_norm @ components # [n_sims, 8]
            
        #     # 转回 numpy 用于绘图
        #     best_s_sims_for_plot = best_s_sims_pca.cpu().numpy()
            
        #     # 生成对应的标签名
        #     pca_feat_names = [f"PC_{i+1}" for i in range(n_pca_components)]
        # else:
        #     best_s_sims_for_plot = best_s_sims_raw.cpu().numpy()
        #     pca_feat_names = None
    
        # print(f"Running diagnostic for Subject {p_idx}, Model {model_name}...")
        # diagnostic_mle_fit(
        #     best_s_sims_for_plot, 
        #     feature_names=pca_feat_names,  # 传入标签
        #     save_path=f"diagnostic_subject_{p_idx}_{model_name}_PCA.png"
        # )

        print(f"  > {model_name} Best LL: {best_ll:.2f} @ {best_params}")

        # best_sims_np = best_s_sims # [200, d_dimensions]
        # # 计算相关系数矩阵
        # corr_matrix = np.corrcoef(best_sims_np, rowvar=False)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        # plt.title(f"Feature Correlation Matrix (Subject {p_idx})")
        # plt.savefig('Fig5_Feature_Corr_Subject_{}_{}.png'.format(p_idx, model_name), dpi=300)

    # save results
    np.savez(save_filename, model_name_lls=model_name_lls)

df_results = pd.DataFrame(comparison_records)
plot_model_comparison(df_results)