import datasets.multitask as task
from functions.utils.eval_utils import lock_random_seed
from multitask_train import do_eval
from collections import defaultdict
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from functions.generative_network_modelling.generative_network_modelling import Gen_one_connection
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage, distance, fcluster
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb
from tqdm import tqdm
import numpy as np
import bct
import torch

import matplotlib
from matplotlib import font_manager 

fonts_path = '/home/wyuhang/.conda/myfonts'
font_files = font_manager.findSystemFonts(fontpaths=fonts_path)

for file in font_files:
    font_manager.fontManager.addfont(file)

matplotlib.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams["font.sans-serif"] = 'Myriad Pro'
matplotlib.rcParams['pdf.fonttype'] = 42


lock_random_seed(2024)
task_num = 20
seed=100
device = torch.device('cuda:6')
# step_list = [ step for step in range(500, 40500, 6500)]
step_list = [ step for step in range(500, 40500, 3000)]

row_matrices = 2
col_matrices = int(len(step_list) / row_matrices)
# num_matrices = len(step_list)

# model_size_list = [128, 64, 30, 25, 20, 15, 10]
model_size = 128
# task_num_list = [3, 6, 11, 16, 20]
task_num = 20

fig, axs = plt.subplots(row_matrices, col_matrices, figsize=(1.35 * col_matrices, 1.35 * row_matrices)) 

q_value_list = []

for i, step in enumerate(tqdm(step_list)):

    model = torch.load(f'runs/Fig2bcde_data/n_rnn_{model_size}_task_{task_num}_seed_{seed}/RNN_interleaved_learning_{step}.pth', device) 
    model.hp['reg_term'] = False
    recurrent_conn = model.recurrent_conn.weight.data.detach().cpu().numpy()
    ci, sc_qvalue = bct.modularity_dir(np.abs(recurrent_conn))
    
    ruleset = 'all'

    hp = {'activation': 'softplus', 'use_snn':False}

    default_hp = task.get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)

    hp = default_hp
    hp['seed'] = 2024
    hp['rng'] = np.random.RandomState(hp['seed'])
    hp['rule_trains'] = task.rules_dict[ruleset]
    hp['rules'] = hp['rule_trains']

    N = len(recurrent_conn)

    hidden_states_list = []

    def hook(module, input, output):
        input, = input
        # input.shape: T * Batch_size * N
        hidden_states_list.append(input.detach().mean(dim=(1)))
        
    handle = model.readout.register_forward_hook(hook)

    for rule in hp['rule_trains'][:task_num]:

        trial = task.generate_trials(
            rule, hp, 'random',
            batch_size=512)

        input = torch.from_numpy(trial.x).to(device)
        target = torch.from_numpy(trial.y).to(device)

        output = model(input)

    # hidden_states_list:  [T1 * N, T2 * N, ..., Tm * N]
    min_value = 5000
    max_value = -5000
    for hidden_states in hidden_states_list:
        min_value = min(min_value, hidden_states.min().item())
        max_value = max(max_value, hidden_states.max().item())
    
    
    bins = np.linspace(min_value, max_value, 20)
    rs_list = []
    
    for neuron_id in range(model_size):
        corr = np.zeros(shape=(task_num, task_num))
        neuron_dist = []
        for task_i in range(task_num):
            neuron_activations = hidden_states_list[task_i][:,neuron_id].cpu()
            x, _ = np.histogram(neuron_activations, bins=bins)
            neuron_dist.append(x)
        data_matrix = np.array(neuron_dist) 
        correlation_matrix = np.corrcoef(data_matrix) # task_num * task_num 该神经元对不同任务参与度的相似性
        rs_list.append(correlation_matrix)
    
    
    corr_matrix = np.zeros(shape=(model_size, model_size))
    matrix_elements_list = []
    for neuron_i in range(model_size):
        matrix_i = rs_list[neuron_i]
        tri_upper_indices = np.triu_indices(n=matrix_i.shape[0], k=1)
        matrix_i_elements = matrix_i[tri_upper_indices] # 任务参与度矩阵的向量化  
        matrix_elements_list.append(matrix_i_elements) 
        
        #TODO: 这里需要尝试几种度量
        
        # for neuron_j in range(neuron_i, model_size):
        #     if neuron_i == neuron_j:
        #         corr_matrix[neuron_i, neuron_j] = 1.0
        #         continue

        #     matrix_j = rs_list[neuron_j]
        #     tri_upper_indices = np.triu_indices(n=matrix_j.shape[0], k=1)
        #     matrix_j_elements = matrix_j[tri_upper_indices]
            
        #     correlation, p_value = spearmanr(matrix_i_elements, matrix_j_elements)
            
            
        #     corr_matrix[neuron_i, neuron_j] = correlation
        #     corr_matrix[neuron_j, neuron_i] = correlation
        
    data_matrix = np.array(matrix_elements_list)
    corr_matrix = np.corrcoef(data_matrix) # 不同神经元基于任务参与度的相似性
        
    handle.remove()

    # dissimilarity = 1 - corr_matrix
    # ci, q_value = bct.modularity_dir(dissimilarity)
    ci, q_value = bct.community_louvain(W=corr_matrix, B='negative_asym', seed=2024)

    print(q_value)

    q_value_list.append(q_value)

    
    # distance_matrix = linkage(distance.pdist(dissimilarity), method='ward')
    
    # ci = fcluster(distance_matrix, 0.5, criterion='distance')
    
    # corr_matrix[corr_matrix < 0] =0
    # ci, q_value = bct.modularity_dir(dissimilarity)
    
    sorted_indices = np.argsort(ci)
    sorted_matrix = corr_matrix[sorted_indices][:, sorted_indices]
    sorted_cluster_labels = ci[sorted_indices]
    
    if len(axs.shape) > 1:
        ax = axs[(i//col_matrices)][(i%col_matrices)]
    else:
        ax = axs[(i//col_matrices) + (i%col_matrices)]
    im = ax.imshow(sorted_matrix, cmap='viridis', interpolation='nearest')

    ax.set_title(f'Iteration {step}', fontsize=6)
    interval_len = 16
    ticks_range = np.arange(interval_len, model_size+1, interval_len)
    ax.set_xticks(ticks_range)
    ax.set_xticklabels(ticks_range, rotation=45, fontsize=5)
    ax.set_yticks(ticks_range)
    ax.set_yticklabels(ticks_range, fontsize=5)
    if i >= col_matrices:
        ax.set_xlabel('Neurons', fontsize=6, labelpad=0)
    if i % col_matrices ==0 :
        ax.set_ylabel('Neurons', fontsize=6, labelpad=0)
    
    ax.tick_params(axis='both', width=0.25, length=1.0)
    ax.spines['top'].set_linewidth(0.25)    
    ax.spines['bottom'].set_linewidth(0.25) 
    ax.spines['left'].set_linewidth(0.25)  
    ax.spines['right'].set_linewidth(0.25) 
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=0)


plt.tight_layout()
# 调整不同row_matrices之间的间距
plt.subplots_adjust(hspace=0.00)  

# 添加 colorbar 并将 label 移动到左侧
cbar = plt.colorbar(im, ax=plt.gcf().get_axes(), orientation='vertical')

cbar.ax.yaxis.set_tick_params(labelsize=5)  # 控制 colorbar 刻度字体大小
cbar.outline.set_linewidth(0.25)  
cbar.ax.yaxis.set_tick_params(width=0.25, length=1.0)
cbar.ax.yaxis.set_tick_params(pad=0)
        
# 设置 colorbar 的标签并将其旋转以显示在左侧
cbar.set_label('Similarity', labelpad=2, fontsize=6)

# 调整 colorbar 标签的位置
cbar.ax.yaxis.set_label_position('left')

# plt.colorbar(im, ax=plt.gcf().get_axes(), orientation='vertical', label='Similarity')
plt.savefig(f'./figures/Fig3/Fig3b/Neurons_cluster{model_size}_{task_num}.svg', format='svg', dpi=300)
plt.savefig(f'./figures/Fig3/Fig3b/Neurons_cluster{model_size}_{task_num}.jpg', format='jpg', dpi=300)


# 检查 q_value_list 和 step_list 是否长度一致
assert len(q_value_list) == len(step_list), "q_value_list 和 step_list 长度必须一致"

# 绘制 q_value_list 的变化曲线
plt.figure(figsize=(5, 5))
plt.plot(step_list, q_value_list, marker='o', linestyle='-', label='Q Value')

# 添加图形的标题和坐标轴标签
# plt.title('Modularity based on Neural Similarity Over Iterations', fontsize=12)
plt.xlabel('Iterations', fontsize=7)
plt.ylabel('Modularity', fontsize=7)

# 添加网格
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 添加图例
plt.legend(fontsize=5)

# 调整 x 轴刻度，确保清晰显示
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

# 保存图形
# plt.savefig('./figures/Fig3/Fig3b/Q_value_change.svg', format='svg', dpi=300)
# plt.savefig('./figures/Fig3/Fig3b/Q_value_change.jpg', format='jpg', dpi=300)
plt.savefig('./figures/Fig3/Fig3b/suplementary_fig3c.pdf', format='pdf', dpi=300)