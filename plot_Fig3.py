import datasets.multitask as task
from functions.utils.common_utils import lock_random_seed
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import numpy as np
import bct
import torch
import os

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42


figures_path = './figures/Fig3/Fig3c/'
if not os.path.exists(figures_path):
    os.makedirs(figures_path)
    

lock_random_seed(2025)
task_num = 20
seed = 100
device = torch.device('cuda:0')
step_list = [ step for step in range(500, 40500, 3000)]

row_matrices = 2
col_matrices = int(len(step_list) / row_matrices)
# num_matrices = len(step_list)

model_size = 64
task_num = 20

fig, axs = plt.subplots(row_matrices, col_matrices, figsize=(1.27 * col_matrices, 1.27 * row_matrices)) 


for i, step in enumerate(tqdm(step_list)):

    model = torch.load(f'runs/Fig2b-h/n_rnn_{model_size}_task_{task_num}_seed_{seed}/RNN_interleaved_learning_{step}.pth', device) 
    hp = model.hp
    N = model_size
    
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
    neurons_corr_matrices = []
    
    for neuron_j in range(model_size):
        neuron_dist = []
        for task_i in range(task_num):
            neuron_activations = hidden_states_list[task_i][:,neuron_j].cpu()
            x, _ = np.histogram(neuron_activations, bins=bins) # x is a vector, representing the distribution of neuron_j activation during task_i
            neuron_dist.append(x) 
        neuron_dist = np.array(neuron_dist) # task_num * (len(bins)-1) matrix, representing the distribution of neuron_j activation across different tasks
        correlation_matrix = np.corrcoef(neuron_dist) # task_num * task_num similarity of this neuron's participation across different tasks
        neurons_corr_matrices.append(correlation_matrix) # neurons_corr_matrices is a list of length model_size, each element is a task_num * task_num matrix representing the similarity of this neuron's participation across different tasks
    

    neurons_vector_list = []
    for neuron_i in range(model_size):
        matrix_i = neurons_corr_matrices[neuron_i]
        tri_upper_indices = np.triu_indices(n=matrix_i.shape[0], k=1)
        vector_i = matrix_i[tri_upper_indices] # Vectorization of task participation matrix
        neurons_vector_list.append(vector_i)

    neurons_vector_list = np.array(neurons_vector_list)
    corr_matrix = np.corrcoef(neurons_vector_list) # Similarity between different neurons based on task participation
        
    handle.remove()

    ci, _ = bct.community_louvain(W=corr_matrix, B='negative_asym', seed=2024)

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
    if i % col_matrices == 0 :
        ax.set_ylabel('Neurons', fontsize=6, labelpad=0)
    
    ax.tick_params(axis='both', width=0.25, length=1.0)
    ax.spines['top'].set_linewidth(0.25)    
    ax.spines['bottom'].set_linewidth(0.25) 
    ax.spines['left'].set_linewidth(0.25)  
    ax.spines['right'].set_linewidth(0.25) 
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=0)


plt.tight_layout()
# Adjust spacing between different row_matrices
plt.subplots_adjust(hspace=0.2)

# Add colorbar and move label to left
cbar = plt.colorbar(im, ax=plt.gcf().get_axes(), orientation='vertical')

cbar.ax.yaxis.set_tick_params(labelsize=5)  # Control colorbar tick label font size
cbar.outline.set_linewidth(0.25)
cbar.ax.yaxis.set_tick_params(width=0.25, length=1.0)
cbar.ax.yaxis.set_tick_params(pad=0)

# Set colorbar label and rotate it to display on left
cbar.set_label('Similarity', labelpad=2, fontsize=6)

# Adjust colorbar label position
cbar.ax.yaxis.set_label_position('left')

# plt.colorbar(im, ax=plt.gcf().get_axes(), orientation='vertical', label='Similarity')
plt.savefig(f'./figures/Fig3/Fig3c/Neurons_cluster{model_size}_{task_num}.svg', format='svg', dpi=300)
plt.savefig(f'./figures/Fig3/Fig3c/Neurons_cluster{model_size}_{task_num}.jpg', format='jpg', dpi=300)
