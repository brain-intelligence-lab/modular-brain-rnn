import numpy as np 
import matplotlib.pyplot as plt
import argparse
import pdb
import tensorflow as tf
import matplotlib
import matplotlib.cm as cm
import scipy.stats as stats
from statannot import add_stat_annotation 
from functions.utils.plot_utils import list_files, plot_fig
from functions.utils.math_utils import find_connected_components, get_induced_subgraphs
import bct
import torch
import os
import seaborn as sns
import pandas as pd


matplotlib.rcParams['pdf.fonttype'] = 42

def start_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--incremental_path', default='./runs/Fig4bc/incremental', type=str)
    parser.add_argument('--normal_path', default='./runs/Fig4bc/all-at-once', type=str)
    parser.add_argument('--random_mask_path', default='./runs/Fig4efg/lottery_ticket_hypo_random', type=str)
    parser.add_argument('--prior_mask_path', default='./runs/Fig4efg/lottery_ticket_hypo_prior_modular', type=str)
    parser.add_argument('--posteriori_mask_path', default='./runs/Fig4efg/lottery_ticket_hypo_posteriori_modular', type=str)
    args = parser.parse_args()
    return args

def plot_figure4b(args):
    model_size = 84

    seed_list = [ i for i in range(100, 2100, 100)]

    task_list = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti', 
                 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo',
                'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm']

    task_perf_name_list = [f'perf_{task}' for task in task_list]
    
    
    fig, axs = plt.subplots(figsize=(2.0, 2.0))
    directory_name_list = [args.incremental_path, args.normal_path]
    
    for directory_name in directory_name_list:
        seed_paths_list = []
        for s_idx, seed_name in enumerate(seed_list):
            file_name = f"n_rnn_{model_size}_seed_{seed_name}"
            paths = list_files(directory_name, file_name)
            seed_paths_list.append(paths)

        modularity_seed_array = []
        task_perf_seed_array_dict = {task:[] for task in task_list}
        
        for ii, events_file in enumerate(seed_paths_list):            
            modularity_list = [0]
            task_perf_list = {task:[0] for task in task_list}
            
            for e in tf.compat.v1.train.summary_iterator(events_file):
                for v in e.summary.value:
                    if v.tag == 'SC_Qvalue':
                        modularity_list.append(v.simple_value)
                    if v.tag in task_perf_name_list:
                        task = v.tag.split('_')[1]
                        task_perf_list[task].append(v.simple_value)
            
            modularity_seed_array.append(np.array(modularity_list))
            for task in task_list:
                task_perf_seed_array_dict[task].append(task_perf_list[task])
        
        modularity_seed_array = np.array(modularity_seed_array)
    
        for task in task_list:
            task_perf_seed_array_dict[task] = np.array(task_perf_seed_array_dict[task])
        
        task_perf_mean_dict = {}
        task_perf_ste_dict = {}
        for task in task_list:
            task_perf_mean_dict[task] = np.mean(task_perf_seed_array_dict[task], axis=0)
            performance_std = np.std(task_perf_seed_array_dict[task], axis=0)
            task_perf_ste_dict[task] = performance_std / np.sqrt(task_perf_seed_array_dict[task].shape[0])
        
        
        modularity_mean = np.mean(modularity_seed_array, axis=0)
        modularity_std = np.std(modularity_seed_array, axis=0)
        modularity_ste = modularity_std / np.sqrt(modularity_seed_array.shape[0])
        
        print(f'n_rnn:{model_size}, avg_moduarlity:{modularity_mean.mean():.4f}')
        for task in task_list:
            print(f'perf_{task}: {task_perf_mean_dict[task].mean():.4f}')
        

        # 生成要显示的标签位置
        x_ticks = [i for i in range(20, modularity_seed_array.shape[1]+1, 20)]
        # x_ticks = [0] + x_ticks
        x_tick_labels = [500 * i for i in x_ticks]
                
        axs.set_xticks(x_ticks)
        axs.set_xticklabels(x_tick_labels, rotation=45, fontsize=6)
        axs.tick_params(axis='both', labelsize=5)
        axs.tick_params(axis='both', width=0.25)
        # axs.set_xlim(0, 90)
        # axs.axvline(x=80, color='green', linestyle='--', linewidth=0.75)
        
        color_list = ['#2171A8', '#DA762A']
        axs.set_ylim(0, 0.5)
        if 'incremental' in directory_name:
            color = color_list[1]
            axs.plot(modularity_mean, label='incremental', linewidth=0.5, color=color)
        else:
            color = color_list[0]
            axs.plot(modularity_mean, label='all-at-once', linewidth=0.5, color=color)
        
        axs.fill_between(range(modularity_seed_array.shape[1]), modularity_mean - modularity_ste, \
            modularity_mean + modularity_ste, alpha=0.2, color=color)
        
    axs.spines['left'].set_position('zero')
    axs.spines['bottom'].set_position('zero') 
    axs.spines['top'].set_linewidth(0.25)    
    axs.spines['bottom'].set_linewidth(0.25) 
    axs.spines['left'].set_linewidth(0.25)  
    axs.spines['right'].set_linewidth(0.25)  
    
    axs.set_xlabel('Iterations', fontsize=6, labelpad=2)

    plt.legend(loc='lower right', bbox_to_anchor=(1.00, 0.02), frameon=False, ncol=1, fontsize=5, title_fontsize=6)
    axs.set_ylabel('Modularity', fontsize=6, labelpad=2)
    
    plt.tight_layout()
    plt.savefig(f"./figures/Fig4/Fig4b.svg", format='svg', dpi=300)
    plt.savefig(f"./figures/Fig4/Fig4b.jpg", format='jpg', dpi=300)

def plot_figure4c(args):
    n_rnn = 84
    device = torch.device('cuda:0')

    conn_modes=["grow", "fix"]

    sc_qvalue_list_grow = []
    sc_qvalue_list_fixed = []

    for seed in range(100, 2100, 100):
        all_merge_into_one = False
        for step in range(3000, 40000, 500):
            if all_merge_into_one:
                break
        
            for _, conn_mode in enumerate(conn_modes):

                if conn_mode == 'grow':
                    path = args.incremental_path
                else:
                    path = args.normal_path
                model = torch.load(os.path.join(path, f'n_rnn_{n_rnn}_seed_{seed}', f'RNN_interleaved_learning_{step}.pth'), device)  
                    
                if conn_mode == 'grow':
                    components = find_connected_components(model.mask)

                weights = model.recurrent_conn.weight.data.detach().cpu().numpy()
                ci, sc_qvalue = bct.modularity_dir(np.abs(weights))

                weights = np.abs(weights)
                subgraphs_list = get_induced_subgraphs(weights, components)

                if len(subgraphs_list) < 2:
                    all_merge_into_one = True
                    break

                for subgraph in subgraphs_list:
                    ci, sc_qvalue = bct.modularity_dir(subgraph)
                    if conn_mode == 'grow':
                        sc_qvalue_list_grow.append(sc_qvalue)
                    else:
                        sc_qvalue_list_fixed.append(sc_qvalue)

                # print(len(subgraphs_list))

                    
    qvalue_dict = {'incremental': sc_qvalue_list_grow, 'all-at-once': sc_qvalue_list_fixed}
    model_list = ['incremental', 'all-at-once',]
    data = []
    for _, model_name in enumerate(model_list):
        Modularity_dist = qvalue_dict[model_name]
        for qvalue in Modularity_dist:
            data.append({'Training paradigm': model_name, 'Modularity': qvalue})

    df = pd.DataFrame(data)    

    palette = ['#2171A8', '#DA762A'] # all-at-once, incremental
    names_order = ['incremental', 'all-at-once']

    group = 'Training paradigm'
    column = 'Modularity'

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax = sns.boxplot(x=group, y=column, data=df, ax=ax, palette=palette,  
                boxprops=dict(facecolor='none', linewidth=0.25), width=0.3, 
                flierprops={
                                'markersize': 1,      # 异常值的大小
                                'markeredgewidth': 0.25,  # 异常值边框线宽
                                }, 
                whiskerprops={'linewidth': 0.25}, medianprops={'linewidth': 0.25}, capprops={'linewidth': 0.25})
    ax = sns.stripplot(x=group, y=column, data=df, 
                dodge=False, ax=ax, palette=palette, jitter=0.1, size=0.8, color='black', alpha=0.7)

    box_pairs = []
    for i in range(len(names_order)):
        for j in range(i+1, len(names_order)):
            box_pairs.append((names_order[i], names_order[j]))


    add_stat_annotation(ax, data=df, x=group, y=column, order=names_order, box_pairs=box_pairs, test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=5)
    # add_stat_annotation(ax, data=df, x=group, y=column, order=names_order, box_pairs=box_pairs, test='t-test_ind', text_format='star', loc='inside', verbose=2, fontsize=5)

    ax.tick_params(axis='both', labelsize=5)
    ax.tick_params(axis='both', width=0.25)
    ax.spines['top'].set_linewidth(0.25)    
    ax.spines['bottom'].set_linewidth(0.25) 
    ax.spines['left'].set_linewidth(0.25)  
    ax.spines['right'].set_linewidth(0.25)  

    ax.set_title('Induced Subgraphs Comparison', fontsize=6)
    ax.set_xlabel('Training paradigm', fontsize=6, labelpad=2)
    ax.set_ylabel('Modularity', fontsize=6, labelpad=2)

    plt.tight_layout()
    plt.savefig('./figures/Fig4/Fig4c.jpg', format='jpg', dpi=300)
    plt.savefig('./figures/Fig4/Fig4c.svg', format='svg', dpi=300)


def plot_figure4e(args):
    model_size_list = [8, 16, 32]

    for model_size in model_size_list:

        fig = plt.figure(figsize=(2.0, 2.0))
        task_num_list = [20]

        directory_name = args.random_mask_path
        seed_list = [ i for i in range(100, 2100, 100)]

        color_map = cm.get_cmap('Blues')
        color_indices = np.linspace(0.4, 0.9, len(model_size_list))  
        color_dict = {model_size: color_map(ci) for model_size, ci in zip(sorted(model_size_list), color_indices)}


        random_modularity_array, random_perf_array = \
            plot_fig(directory_name, seed_list, task_num_list, [model_size], ylabel='Performance', \
                plot_perf=True, linelabel=f'# random', color_dict=color_dict, y_lim_perf=0.8)
        

        color_map = cm.get_cmap('Reds')
        color_indices = np.linspace(0.4, 0.9, len(model_size_list)) 
        color_dict = {model_size: color_map(ci) for model_size, ci in zip(sorted(model_size_list), color_indices)}
        
        directory_name = args.posteriori_mask_path
        postriori_modularity_array, postriori_perf_array = \
            plot_fig(directory_name, seed_list, task_num_list, [model_size], ylabel='Avg performance', \
            plot_perf=True, linelabel=f'# posteriori_modular', color_dict=color_dict, y_lim_perf=0.8)


        color_map = cm.get_cmap('Greens')
        color_indices = np.linspace(0.4, 0.9, len(model_size_list)) 
        color_dict = {model_size: color_map(ci) for model_size, ci in zip(sorted(model_size_list), color_indices)}
        
        directory_name = args.prior_mask_path
        prior_modularity_array, prior_perf_array = \
            plot_fig(directory_name, seed_list, task_num_list, [model_size], ylabel='Avg performance', \
        plot_perf=True, linelabel=f'# prior_modular', color_dict=color_dict, y_lim_perf=0.8)

        iterations = np.arange(0, 40500, 500) 

        t_stat, p_value = stats.ttest_ind(postriori_modularity_array, random_modularity_array)

        first_line = "Iterations"
        second_line = "P value"

        for i in range(0, len(iterations), 6):
            if iterations[i] > 10000 and iterations[i] <= 42000:
                first_line += f" & {iterations[i]}"
                second_line += f" & {p_value[i]:.4f}"
    
        print(first_line)
        print(second_line)

        plt.title(f'# Hidden Neurons: {model_size}', fontsize=6)
        plt.tight_layout()
        
        fig.savefig(f'./figures/Fig4/Fig4e_{model_size}.jpg', format='jpg', dpi=300)
        fig.savefig(f'./figures/Fig4/Fig4e_{model_size}.svg', format='svg', dpi=300)


if __name__ == '__main__':
    figures_path = './figures/Fig4'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    args = start_parse()
    plot_figure4b(args)
    plot_figure4c(args)
    plot_figure4e(args)
    