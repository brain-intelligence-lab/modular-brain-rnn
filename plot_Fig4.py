import numpy as np 
import matplotlib.pyplot as plt
import argparse
import pdb
import tensorflow as tf
import matplotlib
from matplotlib import font_manager
from statannot import add_stat_annotation 
import bct
import torch
import os
import seaborn as sns
import pandas as pd

fonts_path = '/home/wyuhang/.conda/myfonts'
font_files = font_manager.findSystemFonts(fontpaths=fonts_path)

for file in font_files:
    font_manager.fontManager.addfont(file)

matplotlib.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams["font.sans-serif"] = 'Myriad Pro'
matplotlib.rcParams['pdf.fonttype'] = 42

_rule_color = {
    'reactgo': 'green',
            'delaygo': 'olive',
            'fdgo': 'forest green',
            'reactanti': 'mustard',
            'delayanti': 'tan',
            'fdanti': 'brown',
            'dm1': 'lavender',
            'dm2': 'aqua',
            'contextdm1': 'bright purple',
            'contextdm2': 'green blue',
            'multidm': 'blue',
            'delaydm1': 'indigo',
            'delaydm2': 'grey blue',
            'contextdelaydm1': 'royal purple',
            'contextdelaydm2': 'dark cyan',
            'multidelaydm': 'royal blue',
            'dmsgo': 'red',
            'dmsnogo': 'rose',
            'dmcgo': 'orange',
            'dmcnogo': 'peach'
            }

rule_color = {k: 'xkcd:'+v for k, v in _rule_color.items()}


def list_files(directory, name):
    path_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if '.pth' in file or '.txt' in file:
                continue
            if name == root.split('/')[-1]:
                path_list.append(os.path.join(root, file))
                
    if len(path_list)!=1:
        pdb.set_trace()

    return path_list[0]


def find_connected_components(adj_matrix):     
    n = len(adj_matrix)  
      
    visited = {i: False for i in range(n)}  
    components = []  

    def dfs(node, component):  
        # 标记当前节点为已访问  
        visited[node] = True  
        component.append(node)        
        for neighbor in range(n):  
            if adj_matrix[node][neighbor] and not visited[neighbor]:  
                dfs(neighbor, component)  
    
    for i in range(n):  
        if not visited[i]:  
            component = []  
            dfs(i, component)  
            components.append(component)  

    return components

def get_induced_subgraphs(weight_matrix, components):  
    N = len(weight_matrix)  
    subgraphs = []  
      
    for component in components:  
        n = len(component)
        if n < 3:
            continue

        subgraph_matrix = np.zeros((n, n))

        for i in range(len(component)):
            for j in range(len(component)):
                u = component[i]
                v = component[j]
                subgraph_matrix[i,j] = weight_matrix[u,v]
          
        subgraphs.append(subgraph_matrix)  
      
    return subgraphs  


def start_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg_factor', default=0.5, type=float)
    args = parser.parse_args()
    return args

def plot_figure4b(args):
    model_size = 84

    seed_list = [ i for i in range(100, 1100, 100)]

    task_list = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti', 
                 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo',
                'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm']
    
    task_name_abbreviation = {'fdgo': 'Go', 'reactgo': 'RT Go', 'delaygo': 'Dly Go', 'fdanti': 'Anti', 'reactanti': 'RT Anti', 'delayanti': 'Dly Anti',
                              'dmsgo': 'DMS', 'dmsnogo': 'DNMS', 'dmcgo': 'DMC', 'dmcnogo':'DNMC',
                            'dm1': 'DM 1', 'dm2': 'DM 2', 'contextdm1': 'Ctx DM1', 'contextdm2': 'Ctx DM2', 'multidm': 'MultSen DM',
                            'delaydm1': 'Dly DM 1', 'delaydm2': 'Dly DM 2',  'contextdelaydm1': 'Ctx Dly DM 1', 'contextdelaydm2': 'Ctx Dly DM 2',  'multidelaydm': 'MultSen Dly DM' }

    task_perf_name_list = [f'perf_{task}' for task in task_list]
    reg_factor = args.reg_factor
    
    
    for plot_perf in [True, False]:
        fig, axs = plt.subplots(figsize=(1.75, 1.75))

        if plot_perf:
            directory_name_list = ["./runs/Fig4_incremental_learning"]
        else:
            directory_name_list = ["./runs/Fig4_incremental_learning", "./runs/Fig4_interleaved_learning"]
        
        
        for directory_name in directory_name_list:
            seed_paths_list = []
            for s_idx, seed_name in enumerate(seed_list):
                if 'interleaved' in directory_name:
                    file_name = f"n_rnn_{model_size}_fixed_seed_{seed_name}"
                else:
                    file_name = f"n_rnn_{model_size}_regfactor_{reg_factor}_seed_{seed_name}"

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

            y_ticks, y_labels = plt.yticks()
            new_y_ticks = [tick for tick in y_ticks if tick != 0.0 ]
            
            
            axs.set_xticks(x_ticks)
            axs.set_xticklabels(x_tick_labels, rotation=45, fontsize=6)
            axs.set_xlim(0, 90)
            axs.tick_params(axis='both', labelsize=5)
            axs.tick_params(axis='both', width=0.25)
            
            if plot_perf:
                axs.set_ylim(0, 1)
                lines = list()
                labels = list()

                for task in task_list:
                    line = axs.plot(task_perf_mean_dict[task], label=task_name_abbreviation[task], linewidth=0.25, color=rule_color[task])
                    lines.append(line[0])
                    labels.append(task_name_abbreviation[task])

                    axs.fill_between(range(modularity_seed_array.shape[1]), task_perf_mean_dict[task] - task_perf_ste_dict[task], \
                        task_perf_mean_dict[task] + task_perf_ste_dict[task], alpha=0.2)
            else:

                color_list = ['#2171A8', '#DA762A']
                axs.set_ylim(0, 0.5)
                if 'interleaved' in directory_name:
                    color = color_list[0]
                    axs.plot(modularity_mean, label='normal', linewidth=0.25, color=color)
                else:
                    color = color_list[1]
                    axs.plot(modularity_mean, label='incremental', linewidth=0.25, color=color)
                
                axs.fill_between(range(modularity_seed_array.shape[1]), modularity_mean - modularity_ste, \
                    modularity_mean + modularity_ste, alpha=0.2, color=color)
            
        axs.spines['left'].set_position('zero')
        axs.spines['bottom'].set_position('zero') 
        axs.spines['top'].set_linewidth(0.25)    
        axs.spines['bottom'].set_linewidth(0.25) 
        axs.spines['left'].set_linewidth(0.25)  
        axs.spines['right'].set_linewidth(0.25)  
        
        axs.set_xlabel('Iterations', fontsize=6, labelpad=2)
        if plot_perf:
            axs.set_ylabel('Avg performance', fontsize=6, labelpad=2)
        else:
            plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0.02), frameon=False, ncol=1, fontsize=5, title_fontsize=6)
            axs.set_ylabel('Modularity', fontsize=6, labelpad=2)
            
        plt.tight_layout()
        filename = 'left' if plot_perf else 'right'
        plt.savefig(f"./figures/Fig4/fig4b_{filename}.svg", format='svg', dpi=300)
        plt.savefig(f"./figures/Fig4/fig4b_{filename}.jpg", format='jpg', dpi=300)

def plot_figure4c():
    n_rnn = 84
    device = torch.device('cuda:6')

    conn_modes=["grow", "fix"]

    sc_qvalue_list_grow = []
    sc_qvalue_list_fixed = []

    for seed in range(100, 1100, 100):
        all_merge_into_one = False
        for step in range(3000, 40000, 500):
            if all_merge_into_one:
                break
        
            for _, conn_mode in enumerate(conn_modes):

                if conn_mode == 'grow':
                    model = torch.load(f'runs/Fig4_incremental_learning/n_rnn_{n_rnn}_regfactor_0.5_seed_{seed}/RNN_continual_learning_{step}.pth', device)  
                else:
                    model = torch.load(f'runs/Fig4_interleaved_learning/n_rnn_{n_rnn}_fixed_seed_{seed}/RNN_interleaved_learning_{step}.pth', device)  
                            
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

                print(len(subgraphs_list))

                    
    qvalue_dict = {'incremental': sc_qvalue_list_grow, 'all-in-one': sc_qvalue_list_fixed}
    model_list = ['incremental', 'all-in-one',]
    data = []
    for _, model_name in enumerate(model_list):
        Modularity_dist = qvalue_dict[model_name]
        for qvalue in Modularity_dist:
            data.append({'Learning paradigm': model_name, 'Modularity': qvalue})

    df = pd.DataFrame(data)    

    palette = ['#2171A8', '#DA762A'] # all-in-one, incremental
    names_order = ['incremental', 'all-in-one']

    group = 'Learning paradigm'
    column = 'Modularity'

    fig, ax = plt.subplots(figsize=(1.4, 1.4))
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


    add_stat_annotation(ax, data=df, x=group, y=column, order=names_order, box_pairs=box_pairs, test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=5)
    # add_stat_annotation(ax, data=df, x=group, y=column, order=names_order, box_pairs=box_pairs, test='t-test_ind', text_format='star', loc='inside', verbose=2, fontsize=5)

    ax.tick_params(axis='both', labelsize=5)
    ax.tick_params(axis='both', width=0.25)
    ax.spines['top'].set_linewidth(0.25)    
    ax.spines['bottom'].set_linewidth(0.25) 
    ax.spines['left'].set_linewidth(0.25)  
    ax.spines['right'].set_linewidth(0.25)  

    ax.set_title('Induced Subgraphs Comparison', fontsize=6)
    ax.set_xlabel('Learning paradigm', fontsize=6, labelpad=2)
    ax.set_ylabel('Modularity', fontsize=6, labelpad=2)

    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig('./figures/Fig4/fig4c.jpg', format='jpg', dpi=300)
    plt.savefig('./figures/Fig4/fig4c.svg', format='svg', dpi=300)

def plot_figure4e():

    model_size_list = [64, 32, 16, 8]
    
    seed_list = [ i for i in range(1, 21)]
    seed_list += [100]

    directory_name = "./runs/Fig4_lottery_ticket_hypo"


    for m_idx, model_size in enumerate(model_size_list):
        fig, axs = plt.subplots(figsize=(1.6, 1.6))
        
        seed_paths_list = []
        for s_idx, seed_name in enumerate(seed_list):
            file_name = f"n_rnn_{model_size}_task_20_seed_{seed_name}"
            paths = list_files(directory_name, file_name)
            seed_paths_list.append(paths)

        modularity_seed_array = []
        perf_avg_seed_array = []
        
        lottery_modularity_array = []
        lottery_perf_array = []
        
        for ii, events_file in enumerate(seed_paths_list):            
            modularity_list = [0]
            perf_avg_list = [0]

            for e in tf.compat.v1.train.summary_iterator(events_file):

                for v in e.summary.value:
                    if v.tag == 'SC_Qvalue':
                        modularity_list.append(v.simple_value)
                    if v.tag == 'perf_avg':
                        perf_avg_list.append(v.simple_value)
            
            if ii < len(seed_paths_list) - 1:
                modularity_seed_array.append(modularity_list)
                perf_avg_seed_array.append(perf_avg_list)
            else:
                lottery_modularity_array.append(modularity_list)
                lottery_perf_array.append(perf_avg_list)
            
        modularity_seed_array = np.array(modularity_seed_array)
        perf_avg_seed_array = np.array(perf_avg_seed_array)
        
        lottery_modularity_array = np.array(lottery_modularity_array)
        lottery_modularity_mean = np.mean(lottery_modularity_array, axis=0)
        
        lottery_perf_array = np.array(lottery_perf_array)
        lottery_perf_mean = np.mean(lottery_perf_array, axis=0)

        perf_avg_mean = np.mean(perf_avg_seed_array, axis=0)
        perf_avg_median = np.median(perf_avg_seed_array, axis=0)
        perf_avg_std = np.std(perf_avg_seed_array, axis=0)
        perf_avg_ste = perf_avg_std / np.sqrt(perf_avg_seed_array.shape[0])
        
        modularity_mean = np.mean(modularity_seed_array, axis=0)
        modularity_std = np.std(modularity_seed_array, axis=0)
        modularity_ste = modularity_std / np.sqrt(modularity_seed_array.shape[0])
        
        print(f'n_rnn:{model_size}, avg_perf:{perf_avg_mean.mean():.4f}, avg_moduarlity:{modularity_mean.mean():.4f}')
        
        y_ticks, y_labels = plt.yticks()
        
        new_y_ticks = [tick for tick in y_ticks if tick != 0.0 ]
        axs.set_ylim(0, 1)

        # axes[m_idx].plot(modularity_mean, label='random_mask_avg')
        # axes[m_idx].fill_between(range(perf_avg_seed_array.shape[1]), modularity_mean - modularity_ste, modularity_mean + modularity_ste, alpha=0.2)

        # axes[m_idx].plot(lottery_modularity_mean, label='commnunity_mask')
        color_list = ['#2171A8', '#DA762A']

        # 生成要显示的标签位置
        x_ticks = [i for i in range(20, perf_avg_seed_array.shape[1]+1, 20)]
        # x_ticks = [0] + x_ticks
        x_tick_labels = [500*i for i in x_ticks]


        medians = np.median(perf_avg_seed_array, axis=0)
        axs.plot(medians, label='random_mask', linewidth=0.25, color=color_list[0])
        lower_quartile = np.percentile(perf_avg_seed_array, 25, axis=0)
        upper_quartile = np.percentile(perf_avg_seed_array, 75, axis=0)
        # lower_quartile = np.min(perf_avg_seed_array, axis=0)
        # upper_quartile = np.max(perf_avg_seed_array, axis=0)

        axs.fill_between(range(perf_avg_seed_array.shape[1]), lower_quartile, upper_quartile, alpha=0.2)

        axs.plot(lottery_perf_mean, label='commnunity_mask', linewidth=0.25, color=color_list[1])
        
        # 绘制perf的均值和标准误
        axs.tick_params(axis='both', labelsize=5)
        axs.tick_params(axis='both', width=0.25)
        axs.set_xticklabels(x_tick_labels, rotation=45, fontsize=6)
        axs.set_xticks(x_ticks)
        axs.set_xlim(0, 90)

        axs.spines['left'].set_position('zero')
        axs.spines['bottom'].set_position('zero') 
        axs.spines['top'].set_linewidth(0.25)    
        axs.spines['bottom'].set_linewidth(0.25) 
        axs.spines['left'].set_linewidth(0.25)  
        axs.spines['right'].set_linewidth(0.25)  

        axs.set_title(f'# Hidden Neurons: {model_size}', fontsize=6)
        axs.set_xlabel('Iterations', fontsize=6, labelpad=1)
        axs.set_ylabel('Performance', fontsize=6, labelpad=0)
        axs.legend(loc='lower right', bbox_to_anchor=(1.05, 0.05), frameon=False, fontsize=5)

        # 调整布局
        plt.tight_layout()
        plt.savefig(f"./figures/Fig4/fig4e_{model_size}.svg", format='svg')
        plt.savefig(f"./figures/Fig4/fig4e_{model_size}.jpg", format='jpg')



if __name__ == '__main__':
    args = start_parse()
    plot_figure4b(args)
    plot_figure4c()
    plot_figure4e()
    