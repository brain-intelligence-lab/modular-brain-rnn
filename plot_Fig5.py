import scipy.io
import numpy as np
import torch
from functions.generative_network_modelling.generative_network_modelling import *
from functions.utils.math_utils import lock_random_seed
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import matplotlib
import datasets.multitask as task
from tqdm import tqdm
import re
import pandas as pd
from statannot import add_stat_annotation
import glob
import os
import pdb

matplotlib.rcParams['pdf.fonttype'] = 42
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def Sample_conn_num(prob_matrix, tot_conn_num):
    n = prob_matrix.shape[0]
    u_indx, v_indx = np.where(np.ones((n, n))) 
    indx = u_indx  * n + v_indx
    choices_index = np.random.choice(indx, size=tot_conn_num, replace=False, p=prob_matrix.flatten())

    u_indices = choices_index // n  
    v_indices = choices_index % n  
    
    result_counter_matrix = np.zeros((n, n), dtype=int)
    np.add.at(result_counter_matrix, (u_indices, v_indices), 1)

    assert result_counter_matrix.sum() == tot_conn_num
    return result_counter_matrix

def gnm(Distance, Kseed, eta, gamma, tot_conn_num):
    Fd = ( Distance + 1e-5 ) ** eta
    Fk = ( Kseed + 1e-5 ) ** gamma
    prob_matrix = Fd * Fk
    prob_matrix = prob_matrix / prob_matrix.sum()
    return Sample_conn_num(prob_matrix, tot_conn_num)

def get_target(conn_matrix, Distance):
    target = []
    target.append(bct.degrees_und(conn_matrix))
    target.append(bct_gpu.clustering_coef_bu_gpu(conn_matrix, device=device))
    target.append(bct_gpu.betweenness_bin_gpu(conn_matrix, device=device))
    target.append(Distance[conn_matrix > 0])
    return target

def cal_energy_func(y_target, tot_conn_num, Distance, B, eta_vec, gamma_vec, task_name):

    params_energy = {}
    min_energy = 100000.0

    D = np.ones_like(Distance)
    K = np.ones_like(Distance)
    if 'spatial' in task_name:
        D = Distance
    
    for _, (eta, gamma) in enumerate(zip(eta_vec, gamma_vec)):
        params_energy[(eta, gamma)] = []

        if 'multitask' in task_name:
            K = B

        if 'matching' in task_name:
            conn_matrix = matching_gen(D, FC=K, eta=eta, gamma=gamma, tot_conn_num=tot_conn_num)
        else:
            conn_matrix = gnm(D, K, eta, gamma, tot_conn_num)

        x_target = get_target(conn_matrix, Distance)

        KS = np.zeros((4))
        for i in range(4):
            KS[i] = ks_statistic_gpu(x_target[i], y_target[i], device)

        params_energy[(eta, gamma)].append(KS)

        params_energy[(eta, gamma)] = np.array(params_energy[(eta, gamma)])
        min_energy = min(min_energy, np.mean(params_energy[(eta, gamma)].max(1)))

        if 'random' in task_name:
            break

    file_path = './runs/Fig5_data/energy'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    with open(f'{file_path}/{task_name}.txt', 'w') as file:
        for key, list in params_energy.items():
            for KS in list:
                file.write('%s:%s\n' % (key, KS))

    return min_energy

def get_hidden_states(model):
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

def pre_process(load_step, num_of_seed:int):
    subjects_conn_data = scipy.io.loadmat('./datasets/brain_hcp_data/84/structureM_use.mat')['structureM_use']
    subjects_conn_data = np.transpose(subjects_conn_data, [2, 0, 1])
    subjects_conn_data = subjects_conn_data.astype(np.float32)
    Distance = np.load('./datasets/brain_hcp_data/84/Raw_dis.npy')

    B = []
    for seed in tqdm(range(1, num_of_seed+1)):
        file_name = f'./runs/Fig5_data/84/n_rnn_84_task_20_seed_{seed}/RNN_interleaved_learning_{load_step}.pth'
        model = torch.load(file_name, device)   
        # weights = model.recurrent_conn.weight.data.detach().cpu().numpy()
        # weights = np.abs(weights)
        # np.fill_diagonal(weights, 0)
        # B.append(weights)

        hidden_states_list = get_hidden_states(model)
        for task_id in range(20):
            hidden_states = hidden_states_list[task_id]
            random_value = torch.rand_like(hidden_states) 
            hidden_states =  hidden_states + random_value * 1e-7

            hidden_states_mean = hidden_states.detach().cpu().numpy()
            fc = np.corrcoef(hidden_states_mean, rowvar=False)
            fc[fc < 0] = 1e-5
            B.append(fc)
        
    return subjects_conn_data, Distance, B
    
def cal_energy(num_of_people=20, load_step=10000):
    nruns = 900
    multi_task_num = 20
    num_of_seed = (num_of_people + multi_task_num - 1) // multi_task_num

    eta_vec = np.linspace(-3.0, 3.0, int(np.sqrt(nruns)))
    gamma_vec = np.linspace(-3.0, 3.0, int(np.sqrt(nruns)))

    eta_vec, gamma_vec = np.meshgrid(eta_vec, gamma_vec)
    eta_vec, gamma_vec = eta_vec.ravel(), gamma_vec.ravel()
    Atgt, Distance, B = pre_process(load_step, num_of_seed=num_of_seed)

    num_of_people = min(Atgt.shape[0], num_of_people)
    fitness_list_dict = {}

    y_target_list = []
    for p in tqdm(range(num_of_people)):
        cortex_conn = Atgt[p, ...]
        y_target = get_target(cortex_conn, Distance)
        y_target_list.append(y_target)
    print("pre_process finished!")

    fitness_list_random = []
    for p in tqdm(range(num_of_people)):
        total_conn_num = int((Atgt[p, ...] > 0).sum())
        y_target = y_target_list[p]
        fitness = cal_energy_func(y_target, total_conn_num, Distance, B[p], eta_vec, gamma_vec, task_name=f'random_{p}')
        fitness_list_random.append(fitness)
    fitness_list_dict['random'] = fitness_list_random
    
    fitness_list_spatial = []
    for p in tqdm(range(num_of_people)):
        total_conn_num = int((Atgt[p, ...] > 0).sum())
        y_target = y_target_list[p]
        fitness = cal_energy_func(y_target, total_conn_num, Distance, B[p], eta_vec, gamma_vec, f'spatial_{p}')
        fitness_list_spatial.append(fitness)
    fitness_list_dict['spatial'] = fitness_list_spatial

    fitness_list_multitask = []
    for p in tqdm(range(num_of_people)):
        total_conn_num = int((Atgt[p, ...] > 0).sum())
        y_target = y_target_list[p]
        fitness = cal_energy_func(y_target, total_conn_num, Distance, B[p], eta_vec, gamma_vec, f'multitask_{p}')
        fitness_list_multitask.append(fitness)
    fitness_list_dict['multitask'] = fitness_list_multitask

    fitness_list_spatial_multitask = []
    for p in tqdm(range(num_of_people)):
        total_conn_num = int((Atgt[p, ...] > 0).sum())
        y_target = y_target_list[p]
        fitness = cal_energy_func(y_target, total_conn_num, Distance, B[p], eta_vec, gamma_vec, f'spatial_multitask_{p}')
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

        # 正则表达式模式，用于提取数据
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

def plot_figure5c():
    model_list = ['random', 'spatial', 'multitask', 'spatial_multitask']
    modelname_map = {'random':'random', 'spatial':'spatial', 'multitask':'task', 'spatial_multitask':'spatial+task'}
    # property_name = ['degree', 'clustering', 'betweenness\ncentrality', 'edge\nlength']
    property_name = ['degree', 'clustering', 'betweenness centrality', 'edge length']
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

    palette = ['#7f7f7f', '#2ca02c', '#ff7f0e', '#1f77b4']  # 米色, 绿色, 橙色, 蓝色
    fig, ax = plt.subplots(figsize=(3.0, 1.5))
    
    # 绘制 boxplot，使用hue来区分不同模型，并保持同一模型的颜色一致    
    ax = sns.boxplot(x='Property', y='KS statistic', hue='Generative model', data=df, ax=ax, palette=palette,
                boxprops=dict(facecolor='none', linewidth=0.25),
                flierprops={
                             'markersize': 1,      
                             'markeredgewidth': 0.25, 
                             }, 
                whiskerprops={'linewidth': 0.25}, medianprops={'linewidth': 0.25}, capprops={'linewidth': 0.25})
    

    ax = sns.stripplot(x='Property', y='KS statistic', hue='Generative model', data=df, 
                dodge=True, ax=ax, palette=palette, jitter=0.1, size=0.8, color='black', alpha=0.4)
    
    generative_model_name = ['random', 'spatial', 'task', 'spatial+task']

    box_pairs = []
    for property in property_name:
        for i in range(len(generative_model_name)):
            for j in range(i+1, len(generative_model_name)):
                box_pairs.append(((property, generative_model_name[i]), (property, generative_model_name[j])))

    add_stat_annotation(ax, data=df, x='Property', y='KS statistic', hue='Generative model', box_pairs=box_pairs,
                        test='t-test_ind', loc='inside', verbose=2, fontsize=5)
    
    ax.tick_params(axis='both', labelsize=5)
    ax.tick_params(axis='both', width=0.25)
    ax.spines['top'].set_linewidth(0.25)    
    ax.spines['bottom'].set_linewidth(0.25) 
    ax.spines['left'].set_linewidth(0.25)  
    ax.spines['right'].set_linewidth(0.25)  

    # 将图例放在图形外部
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

    subjects_conn_data, Distance, B = pre_process(load_step=args.load_step, num_of_seed=num_of_seed)
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
    # palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']  # 蓝色, 橙色, 绿色, 
    palette = ['#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd']  # 绿色, 橙色, 蓝色, 紫色
    alpha = 0.4
    
    all_dist = [dist for name, dist in model_dist.items()]
    all_dist = np.concatenate(all_dist)
    min_val = np.percentile(all_dist, 1)  
    max_val = np.percentile(all_dist, 99)  
    xlim = (min_val, max_val)

    for dist_id, (name, dist) in enumerate(model_dist.items()):
        if name == 'Brain':
            for axd_id in range(len(axes)):
                # KDE曲线 + 填充
                sns.kdeplot(dist, color=palette[dist_id], label=name, bw_adjust=0.5, fill=True, alpha=alpha-0.1, ax=axes[axd_id], linewidth=0.25)
                # 直方图
                axes[axd_id].hist(dist, bins=50, density=True, alpha=alpha, color=palette[dist_id], edgecolor='black', linewidth=0.25)
                axes[axd_id].set_xlim(xlim[0], xlim[1])
        else:
            sns.kdeplot(dist, color=palette[dist_id], label=name, bw_adjust=0.5, fill=True, alpha=alpha-0.1, ax=axes[dist_id], linewidth=0.25)
            axes[dist_id].hist(dist, bins=50, density=True, alpha=alpha, color=palette[dist_id], edgecolor='black', linewidth=0.25)
            axes[dist_id].set_xlim(xlim[0], xlim[1])


    for axs in axes:
        axs.legend(fontsize=5)
        axs.set_ylabel('Density', fontsize=6)
        axs.tick_params(axis='both', labelsize=5)
        axs.tick_params(axis='both', width=0.25)  
        axs.spines['top'].set_linewidth(0.25)    
        axs.spines['bottom'].set_linewidth(0.25) 
        axs.spines['left'].set_linewidth(0.25)  
        axs.spines['right'].set_linewidth(0.25)  

        
    plt.xlabel(f'{property}', fontsize=6)

    plt.savefig(f'./figures/Fig5/Fig5defg_{property}_dist.jpg', format='jpg', dpi=300)
    plt.savefig(f'./figures/Fig5/Fig5defg_{property}_dist.svg', format='svg', dpi=300)
    print(f'{property} finish!')


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
    
    palette = ['#7f7f7f', '#2ca02c', '#ff7f0e', '#1f77b4']  # 米色, 绿色, 橙色, 蓝色
    names_order = ['random', 'spatial', 'task', 'spatial+task']
    
    group = 'Generative model'
    column = 'Energy'
    
    fig, ax = plt.subplots(figsize=(2.0, 1.5))
    ax = sns.boxplot(x=group, y=column, data=df, ax=ax, palette=palette,  
                boxprops=dict(facecolor='none', linewidth=0.25), width=0.3, 
                flierprops={
                             'markersize': 1,      # 异常值的大小
                             'markeredgewidth': 0.25,  # 异常值边框线宽
                             }, 
                whiskerprops={'linewidth': 0.25}, medianprops={'linewidth': 0.25}, capprops={'linewidth': 0.25})
    ax = sns.stripplot(x=group, y=column, data=df, 
                dodge=False, ax=ax, palette=palette, jitter=0.1, size=0.6, color='black', alpha=0.4)
    
    box_pairs = []
    for i in range(len(names_order)):
        for j in range(i+1, len(names_order)):
            box_pairs.append((names_order[i], names_order[j]))
    
    add_stat_annotation(ax, data=df, x=group, y=column, order=names_order,
                        box_pairs=box_pairs, test='t-test_ind', text_format='star', loc='inside', verbose=2, fontsize=5)

    ax.tick_params(axis='both', labelsize=5)
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

def start_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--people_num', default=200, type=int)
    parser.add_argument('--nruns', default=900, type=int)
    parser.add_argument('--load_step', default=40000, type=int)
    parser.add_argument('--read_from_file', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    figures_path = './figures/Fig5'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    lock_random_seed(2025)
    args = start_parse()
    if args.read_from_file:
        fitness_list_dict = None
    else:
        fitness_list_dict = cal_energy(num_of_people = args.people_num, load_step = args.load_step)
    
    plot_figure5b(fitness_list_dict)
    plot_figure5c()    
    property_list = ['degree', 'clustering', 'betweenness centrality', 'edge length']

    for i, property in enumerate(property_list):
        plot_figure5defg(args, property)
