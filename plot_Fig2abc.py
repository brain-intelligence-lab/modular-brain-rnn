import numpy as np 
import matplotlib
from matplotlib import font_manager 
import matplotlib.pyplot as plt
import pdb
import matplotlib.cm as cm
import tensorflow as tf
import scipy.stats as stats
import os


fonts_path = '~/.conda/myfonts'
font_files = font_manager.findSystemFonts(fontpaths=fonts_path)

for file in font_files:
    font_manager.fontManager.addfont(file)

matplotlib.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams["font.sans-serif"] = 'Myriad Pro'
matplotlib.rcParams['pdf.fonttype'] = 42


def list_files(directory, name):
    path_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if '.pth' in file or '.txt' in file:
                continue
            if name == root.split('/')[-1]:
                path_list.append(os.path.join(root, file))

    if len(path_list) == 1:
        return path_list[0]
    return None


def get_seed_avg(directory_name, model_size, task, seed_list):
    seed_paths_list = []
    for s_idx, seed_name in enumerate(seed_list):
        file_name = f"n_rnn_{model_size}_task_{task}_seed_{seed_name}"
        paths = list_files(directory_name, file_name)
        seed_paths_list.append(paths)

    modularity_seed_array = []
    perf_avg_seed_array = []
    for ii, events_file in enumerate(seed_paths_list):
        modularity_list = [0]
        perf_avg_list = [0]
        for e in tf.compat.v1.train.summary_iterator(events_file):
            for v in e.summary.value:
                if v.tag == 'SC_Qvalue':
                    modularity_list.append(v.simple_value)
                if v.tag == 'perf_avg':
                    perf_avg_list.append(v.simple_value)
        
        modularity_seed_array.append(modularity_list)
        perf_avg_seed_array.append(perf_avg_list)
        
    modularity_seed_array = np.array(modularity_seed_array)
    perf_avg_seed_array = np.array(perf_avg_seed_array)
    perf_avg_mean = np.mean(perf_avg_seed_array, axis=0)
    modularity_mean = np.mean(modularity_seed_array, axis=0)
    print(f'model_size:{model_size}, avg_perf:{perf_avg_mean.mean():.4f}, avg_moduarlity:{modularity_mean.mean():.4f}')    
    return modularity_seed_array, perf_avg_seed_array

def plot_fig(directory_name, seed_list, task_name_list, model_size_list, ylabel, plot_perf=True, linelabel=None, color_dict=None):

    for model_idx, model_size in enumerate(model_size_list):
        modularity_all_array = []
        perf_avg_all_array = []
        
        for task_idx, task_name in enumerate(task_name_list):
            modularity_seed_array, perf_avg_seed_array = get_seed_avg(directory_name, model_size, task=task_name, seed_list=seed_list)
            modularity_all_array.append(modularity_seed_array)
            perf_avg_all_array.append(perf_avg_seed_array)
        
        epochs_num = modularity_all_array[0].shape[-1]
        modularity_all_array = np.array(modularity_all_array).reshape(-1, epochs_num)
        perf_avg_all_array = np.array(perf_avg_all_array).reshape(-1, epochs_num)
        
        modularity_mean = np.mean(modularity_all_array, axis=0)
        modularity_std = np.std(modularity_all_array, axis=0)
        modularity_ste = modularity_std / np.sqrt(modularity_all_array.shape[0])
        
        perf_avg_mean = np.mean(perf_avg_all_array, axis=0)
        perf_avg_std = np.std(perf_avg_all_array, axis=0)
        perf_avg_ste = perf_avg_std / np.sqrt(perf_avg_all_array.shape[0])

        # 生成要显示的标签位置
        x_ticks = [i for i in range(20, perf_avg_seed_array.shape[1]+1, 20)]
        x_ticks = [0] + x_ticks
        x_tick_labels = [500*i for i in x_ticks]
                
        if plot_perf:
            # 从颜色映射中获取颜色
            color = color_dict[model_size]
            plt.plot(perf_avg_mean, label = f'# Hidden Neurons: {model_size}' if linelabel is None else linelabel, color=color, linewidth=0.25)
            plt.xticks(ticks=x_ticks, labels=x_tick_labels, fontsize=6)
            y_ticks = np.arange(0.0, 1.1, 0.2)  # 注意，终点设置为1.1以包括1.0
            plt.ylim([0.0, 1.0])  # 设置y轴的范围从0.0到1.0
            y_ticks_labels = [f"{tick:.1f}" for tick in y_ticks]  # 格式化标签为一位小数
            plt.yticks(ticks=y_ticks, labels=y_ticks_labels, fontsize=6)        
            plt.ylabel(f'{ylabel}', fontsize=7)
            plt.fill_between(range(perf_avg_seed_array.shape[1]), perf_avg_mean - perf_avg_ste, perf_avg_mean + perf_avg_ste, color=color, alpha=0.2)
        else:
            if 'Multi' in linelabel:
                last_key, color = next(reversed(color_dict.items()))
            else:
                first_key, color = next(iter(color_dict.items()))
            plt.plot(modularity_mean, label = f'# Hidden Neurons: {model_size}' if linelabel is None else linelabel, color=color, linewidth=0.25)
            plt.xticks(ticks=x_ticks, labels=x_tick_labels, fontsize=6)

            y_ticks = np.arange(0.0, 0.30, 0.05)  
            plt.ylim([0.0, 0.25])  
            # y_ticks = np.arange(0.0, 0.60, 0.05)  
            # plt.ylim([0.0, 0.55])  
            y_ticks_labels = [f"{tick:.2f}" for tick in y_ticks]  # 格式化标签为一位小数
            plt.yticks(ticks=y_ticks, labels=y_ticks_labels, fontsize=6) 
            plt.ylabel(f'{ylabel}', fontsize=7)
            plt.fill_between(range(modularity_seed_array.shape[1]), modularity_mean - modularity_ste, modularity_mean + modularity_ste, color=color, alpha=0.2)
            
        plt.xlabel('Iterations', fontsize=6)
        plt.legend(loc='lower right', bbox_to_anchor=(0.98, 0.05), frameon=False, fontsize=5)
        
    ax = plt.gca()
    ax.spines['left'].set_position('zero')  # 将左轴脊移动到0点
    ax.spines['bottom'].set_position('zero')  # 将底轴脊移动到0点
    
    ax.tick_params(axis='both', labelsize=5)
    ax.tick_params(axis='both', width=0.25)
    ax.spines['top'].set_linewidth(0.25)    
    ax.spines['bottom'].set_linewidth(0.25) 
    ax.spines['left'].set_linewidth(0.25)  
    ax.spines['right'].set_linewidth(0.25) 
    return modularity_all_array, perf_avg_all_array

    
def plot_fig2a(model_size_list, color_dict):
    fig = plt.figure(figsize=(2.0, 2.0))
    directory_name = "./runs/Fig2a_data"
    seed_list = [ i for i in range(100, 1100, 100)]
    task_name_list = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
    
    plot_fig(directory_name, seed_list, task_name_list, model_size_list, ylabel='Avg performance', color_dict=color_dict)
    plt.title('Single Task Learning', fontsize=7)
    plt.tight_layout()
    figures_path = './figures/Fig2'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    fig.savefig(f'{figures_path}/Fig2a.svg', format='svg', dpi=300)
    fig.savefig(f'{figures_path}/Fig2a.jpg', format='jpg', dpi=300)

def plot_fig2b(model_size_list, color_dict):
    fig = plt.figure(figsize=(2.0, 2.0))
    directory_name = "./runs/Fig2bcde_data"
    seed_list = [ i for i in range(100, 1100, 100)]
    task_num_list = [20]
    plot_fig(directory_name, seed_list, task_num_list, model_size_list, ylabel='Avg performance', color_dict=color_dict)
    plt.title('Multi-task Learning', fontsize=7)
    plt.tight_layout()
    figures_path = './figures/Fig2'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
        
    fig.savefig(f'{figures_path}/Fig2b.svg', format='svg', dpi=300)
    fig.savefig(f'{figures_path}/Fig2b.jpg', format='jpg', dpi=300)

def plot_fig2c(color_dict, N=15):
    fig = plt.figure(figsize=(2.0, 2.0))
    model_size_list = [N]
    task_num_list = [20]
    directory_name = "./runs/Fig2bcde_data"
    seed_list = [ i for i in range(100, 1100, 100)]
    multitask_modularity_array, multitask_perf_array = \
        plot_fig(directory_name, seed_list, task_num_list, model_size_list, ylabel='Modularity', \
             plot_perf=False, linelabel=f'# Multi-task', color_dict=color_dict)
    
    task_name_list = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

    directory_name = "./runs/Fig2a_data"

    singletask_modularity_array, singletask_perf_array = \
        plot_fig(directory_name, seed_list, task_name_list, model_size_list, ylabel='Modularity', \
        plot_perf=False, linelabel=f'# Single task', color_dict=color_dict)

    iterations = np.arange(0, 47000, 500) 

    t_stat, p_value = stats.ttest_ind(multitask_modularity_array, singletask_modularity_array)

    first_line = "Iterations"
    second_line = "P value"

    for i in range(0, len(iterations), 6):
        if iterations[i] > 10000 and iterations[i] <= 42000:
            first_line += f" & {iterations[i]}"
            second_line += f" & {p_value[i]:.4f}"

    print(first_line)
    print(second_line)

    # plt.title(f'Single task vs Multi-task\n(# Hidden Neurons: {N})', fontsize=6)
    plt.title(f'Single task vs Multi-task', fontsize=6)
    plt.tight_layout()
    
    figures_path = './figures/Fig2'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
        
    fig.savefig(f'{figures_path}/Fig2c_{N}.jpg', format='jpg', dpi=300)
    fig.savefig(f'{figures_path}/Fig2c_{N}.svg', format='svg', dpi=300)
    

model_size_list = [8, 16, 32, 64]
num_curves = len(model_size_list)

color_map = cm.get_cmap('Blues')
# color_map = cm.get_cmap('Reds')
color_indices = np.linspace(0.4, 0.9, len(model_size_list))  
color_dict = {model_size: color_map(ci) for model_size, ci in zip(sorted(model_size_list), color_indices)}

plot_fig2a(model_size_list, color_dict)
plot_fig2b(model_size_list, color_dict)
for N in model_size_list:
    plot_fig2c(color_dict, N)
