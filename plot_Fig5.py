import numpy as np
import torch
from functions.generative_network_modelling.generative_network_modelling import *
from functions.utils.common_utils import lock_random_seed
from functions.utils.common_utils import prepare_data_for_gnm
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import matplotlib
from tqdm import tqdm
import re
import pandas as pd
from statannotations.Annotator import Annotator
import glob
import os
import pdb

matplotlib.rcParams['pdf.fonttype'] = 42
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def cal_energy_given_para(gt_dist, tot_conn_num, Distance, FC, eta_vec, gamma_vec, task_name, directed):

    params_energy = {}
    min_energy = 100000.0

    # Pre-convert Distance and K to tensor
    D_tensor = torch.tensor(Distance, device=device, dtype=torch.float32)
    K_tensor = torch.tensor(FC, device=device, dtype=torch.float32)


    prob_matrix_list = []
    for _, (eta, gamma) in enumerate(zip(eta_vec, gamma_vec)):
        # Calculate probability matrix
        Fd = torch.pow(D_tensor + 1e-9, eta)
        Fk = torch.pow(K_tensor + 1e-9, gamma)
        prob_matrix = Fd * Fk
        prob_matrix_list.append(prob_matrix)

    # Stack into batch: [n_params, n, n]
    prob_matrix_batch = torch.stack(prob_matrix_list, dim=0)
    
    conn_matrix_batch = gnm_batched(prob_matrix_batch, tot_conn_num, directed, device=device)
    
    graph_dists = get_graph_distribution_batched(conn_matrix_batch, Distance, device, directed=directed)
    
    batch_size = graph_dists[0].shape[0]

    KS_batch_list = []
    for i in range(4):
        # Fix: Convert gt to batch form
        gt_tensor = torch.tensor(gt_dist[i], device=device)
        # Expand to (B, N) to match graph_dists[i]
        gt_batch = gt_tensor.unsqueeze(0).expand(batch_size, -1)
        KS_batch = ks_statistic_batch_gpu(graph_dists[i], gt_batch, device)
        KS_batch_list.append(KS_batch) # Each KS_batch has shape (B,)

    KS_all_features = torch.stack(KS_batch_list, dim=1)

    for idx, (eta, gamma) in enumerate(zip(eta_vec, gamma_vec)):
        # Extract the 4 KS values corresponding to the idx-th parameter combination
        current_ks = KS_all_features[idx] # Shape (4,)
        params_energy[(eta, gamma)] = [current_ks.cpu().numpy()] # Maintain consistency with original code list structure

        e = current_ks.max().item() 
        if e < min_energy:
            min_energy = e
            
    
    file_path = './runs/Fig5_data/energy'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    with open(f'{file_path}/{task_name}.txt', 'w') as file:
        for key, ks_list in params_energy.items():
            for KS in ks_list:
                file.write('%s:%s\n' % (key, KS))

    return min_energy
    
def cal_energy(args):
    num_of_people = args.people_num
    load_step = args.load_step
    multi_task_num = 20
    num_of_seed = (num_of_people + multi_task_num - 1) // multi_task_num

    GT_subjects, Distance, Fc = prepare_data_for_gnm(load_step, num_of_seed=num_of_seed, device=device)

    num_of_people = min(GT_subjects.shape[0], num_of_people)
    fitness_list_dict = {}

    gt_dist_list = []
    for p in tqdm(range(num_of_people)):    
        cortex_conn = GT_subjects[p, ...]
        gt_dist = get_graph_distribution(cortex_conn, Distance, device=device)
        gt_dist_list.append(gt_dist)
    print("pre_process finished!")


    # ---- grid (ensure 0 is included) ----
    grid_n = int(np.sqrt(args.nruns))
    if grid_n % 2 == 0:
        grid_n += 1  # ensure 0 included
    eta_grid = np.linspace(-3.0, 3.0, grid_n)
    gamma_grid = np.linspace(-3.0, 3.0, grid_n)

    eta_vec, gamma_vec = np.meshgrid(eta_grid, gamma_grid)
    eta_vec, gamma_vec = eta_vec.ravel(), gamma_vec.ravel()

    eta_random, gamma_random = np.array([0.0]), np.array([0.0])
    
    fitness_list_random = []
    for p in tqdm(range(num_of_people)):
        total_conn_num = int((GT_subjects[p, ...] > 0).sum())
        gt_dist = gt_dist_list[p]
        fitness = cal_energy_given_para(gt_dist, total_conn_num, Distance, \
            Fc[p], eta_random, gamma_random, f'random_{p}', args.directed)
        fitness_list_random.append(fitness)
    fitness_list_dict['random'] = fitness_list_random

    eta_spatial, gamma_spatial = eta_grid, np.array([0.0]*len(eta_grid))

    fitness_list_spatial = []
    for p in tqdm(range(num_of_people)):
        total_conn_num = int((GT_subjects[p, ...] > 0).sum())
        gt_dist = gt_dist_list[p]
        fitness = cal_energy_given_para(gt_dist, total_conn_num, Distance, \
            Fc[p], eta_spatial, gamma_spatial, f'spatial_{p}', args.directed)
        fitness_list_spatial.append(fitness)
    fitness_list_dict['spatial'] = fitness_list_spatial

    eta_multitask, gamma_multitask = np.array([0.0]*len(gamma_grid)), gamma_grid
    fitness_list_multitask = []
    for p in tqdm(range(num_of_people)):
        total_conn_num = int((GT_subjects[p, ...] > 0).sum())
        gt_dist = gt_dist_list[p]
        fitness = cal_energy_given_para(gt_dist, total_conn_num, Distance, \
            Fc[p], eta_multitask, gamma_multitask, f'multitask_{p}', args.directed)
        fitness_list_multitask.append(fitness)
    fitness_list_dict['multitask'] = fitness_list_multitask

    eta_spatial_multitask, gamma_spatial_multitask = eta_vec, gamma_vec
    fitness_list_spatial_multitask = []
    for p in tqdm(range(num_of_people)):
        total_conn_num = int((GT_subjects[p, ...] > 0).sum())
        gt_dist = gt_dist_list[p]
        fitness = cal_energy_given_para(gt_dist, total_conn_num, Distance, \
            Fc[p],  eta_spatial_multitask, gamma_spatial_multitask, f'spatial_multitask_{p}', args.directed)
        fitness_list_spatial_multitask.append(fitness)
    fitness_list_dict['spatial_multitask'] = fitness_list_spatial_multitask
    
    return fitness_list_dict


def read_func(file_name, directory_path = './runs/Fig5_data/energy/', return_KS=False):
    file_pattern = os.path.join(directory_path, file_name)
    file_path_list = glob.glob(file_pattern)
    min_energy_list = []
    min_KS_list = []
    min_param_list = []
    
    for file_path in file_path_list:
        params_energy = {}
        min_energy = 100000.0
        min_KS = None
        min_param = None
        
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Regular expression pattern for data extraction
        pattern = re.compile(r'\((-?\d+\.\d+),\s*(-?\d+\.\d+)\):\[(.+?)\]')

        for line in lines:
            match = pattern.search(line)
            if match:
                eta, gamma = float(match.group(1)), float(match.group(2))
                energy_list = list(map(float, match.group(3).split()))
                energy_list = np.array(energy_list)
                if (eta, gamma) not in params_energy:
                    params_energy[(eta, gamma)] = []
                params_energy[(eta, gamma)].append(energy_list)
        
        for (eta, gamma) in params_energy:
            KS_array = np.array(params_energy[(eta, gamma)])
            energy_value = np.mean(KS_array.max(1))
            if energy_value < min_energy:
                min_energy = energy_value 
                min_KS = KS_array
                min_param = (eta, gamma)

        min_energy_list.append(min_energy)
        min_KS_list.append(min_KS)
        min_param_list.append(min_param)

    if return_KS:        
        return min_KS_list, min_param_list

    return min_energy_list


def plot_figure5b(fitness_list_dict):
    model_list = ['random', 'spatial', 'multitask', 'spatial_multitask',]
    modelname_map = {'random':'random', 'spatial':'spatial', 'multitask':'task', 'spatial_multitask':'spatial+task', 'matching':'matching*', 'spatial_matching':'matching'}
    data = []
    for _, model_name in enumerate(model_list):
        if fitness_list_dict is None:
            energy_dist = read_func(f'{model_name}_[0-9]*')
        else:
            energy_dist = fitness_list_dict[model_name]

        for energy in energy_dist:
            data.append({'Generative model': modelname_map[model_name], 'Energy': energy})

    df = pd.DataFrame(data)

    palette = ['#7f7f7f', '#2ca02c', '#ff7f0e', '#1f77b4']  # Beige, green, orange, blue
    names_order = ['random', 'spatial', 'task', 'spatial+task']
    
    group = 'Generative model'
    column = 'Energy'
    
    fig, ax = plt.subplots(figsize=(1.8, 1.5))
    ax = sns.boxplot(x=group, y=column, data=df, ax=ax, palette=palette,
                boxprops=dict(facecolor='none', linewidth=0.25), width=0.3,
                flierprops={
                             'markersize': 1,      # Size of outliers
                             'markeredgewidth': 0.25,  # Edge line width of outliers
                             }, 
                whiskerprops={'linewidth': 0.25}, medianprops={'linewidth': 0.25}, capprops={'linewidth': 0.25})
    ax = sns.stripplot(x=group, y=column, data=df, 
                dodge=False, ax=ax, palette=palette, jitter=0.1, size=0.6, color='black', alpha=0.4)
    
    box_pairs = []
    for i in range(len(names_order)):
        for j in range(i+1, len(names_order)):
            box_pairs.append((names_order[i], names_order[j]))

# 1. Initialize Annotator object
    annotator = Annotator(ax, box_pairs, data=df, x=group, y=column, order=names_order)

    # 2. Configure statistical test (t-test, significance stars, etc.)
    annotator.configure(
        test='t-test_ind', 
        text_format='star', 
        loc='inside', 
        verbose=2, 
        fontsize=5,
        line_width=0.5  # Can adjust line width as needed
    )
    
    # 3. Execute test and add to plot
    annotator.apply_test()
    annotator.annotate()

    ax.tick_params(axis='both', labelsize=5, pad=1.5)
    ax.tick_params(axis='both', width=0.25)
    ax.spines['top'].set_linewidth(0.25)    
    ax.spines['bottom'].set_linewidth(0.25) 
    ax.spines['left'].set_linewidth(0.25)  
    ax.spines['right'].set_linewidth(0.25)  

    ax.set_xlabel('Generative model', fontsize=6)
    ax.set_ylabel('Energy', fontsize=6)
    
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig('./figures/Fig5/Fig5b.jpg', format='jpg', dpi=300)
    plt.savefig('./figures/Fig5/Fig5b.svg', format='svg', dpi=300)

def plot_figure5c():
    model_list = ['random', 'spatial', 'multitask', 'spatial_multitask']
    modelname_map = {'random':'random', 'spatial':'spatial', 'multitask':'task', 'spatial_multitask':'spatial+task'}
    property_name = ['node\ndegree', 'clustering\ncoefficient', 'betweenness\ncentrality', 'edge\nlength']
    # property_name = ['degree', 'clustering', 'betweenness centrality', 'edge length']
    data_dict = {}

    for i, model_name in enumerate(model_list):
        KS_list, _ = read_func(f'{model_name}_[0-9]*', return_KS=True)
        KS_list = np.array(KS_list)
        KS_list = KS_list[:,0,:]
        KS_list = KS_list.reshape(-1, 4)
        
        data = []
        for _ in range(KS_list.shape[0]):
            for col in range(4):
                data.append({'Property': property_name[col], 'KS statistic': KS_list[_][col], 'Generative model': modelname_map[model_name]})
        
        data_dict[model_name] = data
    
    combined_data = []
    for name, data in data_dict.items():
        combined_data.extend(data)

    df = pd.DataFrame(combined_data)

    palette = ['#7f7f7f', '#2ca02c', '#ff7f0e', '#1f77b4']  # Beige, green, orange, blue
    fig, ax = plt.subplots(figsize=(2.8, 1.5))

    # Draw boxplot using hue to distinguish different models and keep colors consistent for the same model    
    ax = sns.boxplot(x='Property', y='KS statistic', hue='Generative model', data=df, ax=ax, palette=palette,
                boxprops=dict(facecolor='none', linewidth=0.25),
                flierprops={
                             'markersize': 1,      
                             'markeredgewidth': 0.25, 
                             }, 
                whiskerprops={'linewidth': 0.25}, medianprops={'linewidth': 0.25}, capprops={'linewidth': 0.25})
    

    ax = sns.stripplot(x='Property', y='KS statistic', hue='Generative model', data=df,
                dodge=True, ax=ax, palette=palette, jitter=0.1, size=0.8, color='black', alpha=0.4, legend=False)
    
    generative_model_name = ['random', 'spatial', 'task', 'spatial+task']

    box_pairs = []
    for property in property_name:
        for i in range(len(generative_model_name)):
            for j in range(i+1, len(generative_model_name)):
                box_pairs.append(((property, generative_model_name[i]), (property, generative_model_name[j])))

    annotator = Annotator(ax, box_pairs, data=df, x='Property', y='KS statistic', order=property_name, hue='Generative model')

    # 2. Configure statistical test (t-test, significance stars, etc.)
    annotator.configure(
        test='t-test_ind', 
        text_format='star', 
        loc='inside', 
        verbose=2, 
        fontsize=5,
        line_width=0.5
    )

    # 3. Execute test and add to plot
    annotator.apply_test()
    annotator.annotate()


    ax.tick_params(axis='both', labelsize=5, pad=1.5)
    ax.tick_params(axis='both', width=0.25)
    ax.spines['top'].set_linewidth(0.25)    
    ax.spines['bottom'].set_linewidth(0.25) 
    ax.spines['left'].set_linewidth(0.25)
    ax.spines['right'].set_linewidth(0.25)

    # Place legend outside the plot
    legend = ax.legend(title='Generative model', loc='lower right', fontsize=5, title_fontsize=6)
    frame = legend.get_frame()
    frame.set_linewidth(0.25) 

    ax.set_xlabel('Property', fontsize=6)
    ax.set_ylabel('KS statistic', fontsize=6)
    plt.savefig('./figures/Fig5/Fig5c.jpg', format='jpg', dpi=300)
    plt.savefig('./figures/Fig5/Fig5c.svg', format='svg', dpi=300)
    

def get_dist(cortex_conns, Distance, B, para_list, task_name, num_of_dist, property = 'edge length'):
    D = np.ones_like(Distance)
    K = np.ones_like(Distance)
    if 'spatial' in task_name:
        D = Distance

    dist_list = []
    for i in range(num_of_dist):
        tot_conn_num = int((cortex_conns[i] > 0).sum())
        (eta, gamma) = para_list[i]
        if 'multitask' in task_name:
            K = B[i]
        
        conn_matrix = gnm(D, K, eta, gamma, tot_conn_num)

        if property == 'degree':
            dist_list.append(bct.degrees_und(conn_matrix))
        elif property == 'clustering':
            dist_list.append(bct_gpu.clustering_coef_bu_gpu(conn_matrix, device=device))
        elif property == 'betweenness centrality':
            dist_list.append(bct_gpu.betweenness_bin_gpu(conn_matrix, device=device))
        elif property=='edge length':
            dist_list.append(Distance[ conn_matrix > 0 ])
        else:
            raise NotImplementedError

    return np.concatenate(dist_list)

def plot_figure5defg(args, property='edge length'):    
    multi_task_num = 20
    num_of_seed = (args.people_num + multi_task_num - 1) // multi_task_num

    subjects_conn_data, Distance, B = prepare_data_for_gnm(load_step=args.load_step, num_of_seed=num_of_seed, device=device)
    num_of_dist = min(subjects_conn_data.shape[0], args.people_num)

    model_dist = {}
    
    _, min_param = read_func(f'spatial_[0-9]*', return_KS=True)
    model_dist['spatial'] = get_dist(subjects_conn_data, Distance, B, min_param, 'spatial', num_of_dist, property)

    _, min_param = read_func(f'multitask_[0-9]*', return_KS=True)
    model_dist['task'] = get_dist(subjects_conn_data, Distance, B, min_param, 'multitask', num_of_dist, property)
    
    _, min_param = read_func(f'spatial_multitask_[0-9]*', return_KS=True)
    model_dist['spatial+task'] = get_dist(subjects_conn_data, Distance, B, min_param, 'spatial_multitask', num_of_dist, property)
    
    ground_truth_dist = []
    for i in range(num_of_dist):
        if property == 'degree':
            ground_truth_dist.append(bct.degrees_und(subjects_conn_data[i]))
        elif property == 'clustering':
            ground_truth_dist.append(bct_gpu.clustering_coef_bu_gpu(subjects_conn_data[i], device=device))
        elif property == 'betweenness centrality':
            ground_truth_dist.append(bct_gpu.betweenness_bin_gpu(subjects_conn_data[i], device=device))
        elif property=='edge length':
            ground_truth_dist.append(Distance[subjects_conn_data[i]> 0])
        else:
            raise NotImplementedError
    
    model_dist['Brain'] = np.concatenate(ground_truth_dist)
    
    fig, axes = plt.subplots(3, 1, figsize=(1.8, 3.0))
    # palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']  # Blue, orange, green,
    palette = ['#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd']  # Green, orange, blue, purple
    alpha = 0.4
    
    all_dist = [dist for name, dist in model_dist.items()]
    all_dist = np.concatenate(all_dist)
    min_val = np.percentile(all_dist, 1)  
    max_val = np.percentile(all_dist, 99)  
    xlim = (min_val, max_val)

    for dist_id, (name, dist) in enumerate(model_dist.items()):
        if name == 'Brain':
            for axd_id in range(len(axes)):
                # KDE curve + fill
                sns.kdeplot(dist, color=palette[dist_id], label=name, bw_adjust=0.5, fill=True, alpha=alpha-0.1, ax=axes[axd_id], linewidth=0.25)
                # Histogram
                axes[axd_id].hist(dist, bins=50, density=True, alpha=alpha, color=palette[dist_id], edgecolor='black', linewidth=0.25)
                axes[axd_id].set_xlim(xlim[0], xlim[1])
        else:
            sns.kdeplot(dist, color=palette[dist_id], label=name, bw_adjust=0.5, fill=True, alpha=alpha-0.1, ax=axes[dist_id], linewidth=0.25)
            axes[dist_id].hist(dist, bins=50, density=True, alpha=alpha, color=palette[dist_id], edgecolor='black', linewidth=0.25)
            axes[dist_id].set_xlim(xlim[0], xlim[1])


    for axs in axes:
        axs.legend(fontsize=5)
        axs.set_ylabel('Density', fontsize=5)
        axs.tick_params(axis='both', labelsize=4)
        axs.tick_params(axis='both', width=0.25)  
        axs.spines['top'].set_linewidth(0.25)    
        axs.spines['bottom'].set_linewidth(0.25) 
        axs.spines['left'].set_linewidth(0.25)  
        axs.spines['right'].set_linewidth(0.25)  

        
    plt.xlabel(f'{property}', fontsize=5)

    plt.savefig(f'./figures/Fig5/Fig5defg_{property}_dist.jpg', format='jpg', dpi=300)
    plt.savefig(f'./figures/Fig5/Fig5defg_{property}_dist.svg', format='svg', dpi=300)
    print(f'{property} finish!')




def start_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--people_num', default=200, type=int)
    parser.add_argument('--nruns', default=900, type=int)
    parser.add_argument('--load_step', default=40000, type=int)
    parser.add_argument('--directed', action='store_true')
    parser.add_argument('--read_from_file', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    figures_path = './figures/Fig5'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    lock_random_seed(2026)
    args = start_parse()
    if args.read_from_file:
        fitness_list_dict = None
    else:
        fitness_list_dict = cal_energy(args)
    
    plot_figure5b(fitness_list_dict)
    plot_figure5c()    
    property_list = ['degree', 'clustering', 'betweenness centrality', 'edge length']

    for i, property in enumerate(property_list):
        plot_figure5defg(args, property)
