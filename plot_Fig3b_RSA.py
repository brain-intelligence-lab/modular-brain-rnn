import datasets.multitask as task
from functions.utils.eval_utils import lock_random_seed
from multitask_train import do_eval
from collections import defaultdict
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from functions.generative_network_modelling.generative_network_modelling import Gen_one_connection
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage, distance, fcluster
import pdb
from tqdm import tqdm
import numpy as np
import bct
import torch


lock_random_seed(2024)
task_num = 20
num_matrices = 9
seed=100
device = torch.device('cuda:6')

# model_size_list = [10, 15, 20, 25, 30, 64]
model_size_list = [128, 64, 30, 25, 20, 15, 10]
task_num_list = [3, 6, 11, 16, 20]

for _, model_size in enumerate(model_size_list):


    for _, task_num in enumerate(task_num_list):
        
        plt.figure(figsize=(12 * num_matrices, 12))  # 根据矩阵数量调整整体图像大小

        for i in tqdm(range(num_matrices)):
            step = (i+1) * 5000
            
            model = torch.load(f'runs/Fig2bcde_data/n_rnn_{model_size}_task_{task_num}_seed_{seed}/RNN_interleaved_learning_{step}.pth', device)    
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
            
            # ci, q_value = bct.community_louvain(W=corr_matrix, B='negative_asym', seed=2024)
            

            dissimilarity = 1 - corr_matrix
            
            distance_matrix = linkage(distance.pdist(dissimilarity), method='ward')
            
            ci = fcluster(distance_matrix, 0.5, criterion='distance')
            
            # corr_matrix[corr_matrix < 0] =0
            # ci, q_value = bct.modularity_dir(dissimilarity)
            
            sorted_indices = np.argsort(ci)
            sorted_matrix = corr_matrix[sorted_indices][:, sorted_indices]
            sorted_cluster_labels = ci[sorted_indices]
            
            ax = plt.subplot(1, num_matrices, i + 1)  
            
            im = ax.imshow(sorted_matrix, cmap='viridis', interpolation='nearest')

            # ax.set_title(f'Step {step}, Q_{q_value:.4f}', fontsize=18)
            ax.set_title(f'Step {step}', fontsize=18)
            ticks_range = np.arange(0, model_size, 2)
            ax.set_xticks(ticks_range)
            ax.set_xticklabels(ticks_range, rotation=90, fontsize=10)
            ax.set_yticks(ticks_range)
            ax.set_yticklabels(ticks_range, fontsize=10)
            ax.set_xlabel('Neurons', fontsize=14)
            ax.set_ylabel('Neurons', fontsize=14)


        plt.tight_layout()
        plt.colorbar(im, ax=plt.gcf().get_axes(), orientation='vertical', label='Similarity')
        plt.savefig(f'./figures/Fig3/Fig3b/Neurons_cluster{model_size}_{task_num}.svg', format='svg')
        plt.savefig(f'./figures/Fig3/Fig3b/Neurons_cluster{model_size}_{task_num}.jpg', format='jpg')