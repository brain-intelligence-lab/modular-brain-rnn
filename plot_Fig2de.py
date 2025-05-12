import numpy as np 
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
import matplotlib
from matplotlib import font_manager 
import matplotlib.cm as cm
import os

fonts_path = '/home/wyuhang/.conda/myfonts'
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

num_list = [3, 6, 11, 16, 20]
# num_list = [20, 16, 11, 6, 3]
task_num_list = [f"task_{i}" for i in num_list]
seed_name_list = [f"seed_{i}" for i in range(100, 1050, 100)]

num_curves = len(num_list)
color_map = cm.get_cmap('winter')
color_indices = np.linspace(0.00, 1.0, len(num_list))  
# color_map = cm.get_cmap('autumn')
# color_indices = np.linspace(0.25, 0.75, len(num_list))  
color_indices = color_indices[::-1]
color_dict = {idx: color_map(ci) for idx, ci in zip(range(num_curves), color_indices)}

# model_size_list = [64, 32, 16, 8, 4]
model_size_list = [64, 32, 16, 8]

for idx, n_rnn in enumerate(model_size_list):
    n_rnn_name = f"n_rnn_{n_rnn}"
    fig, axs = plt.subplots(figsize=(2.0, 2.0))
    
    for tid_, task_num in enumerate(task_num_list):
        exp_paths_list = []
        for __, seed_name in enumerate(seed_name_list):
            file_name = f"{n_rnn_name}_{task_num}_{seed_name}"
            paths = list_files("./runs/Fig2bcde_data/", file_name)
            if paths != None:
                exp_paths_list.append(paths)
    
        modularity_array = []
        perf_avg_array = []
        cluster_num_array = []
        for ii, events_file in enumerate(exp_paths_list):
            modularity_list = [0]
            perf_avg_list = [0]
            
            for e in tf.compat.v1.train.summary_iterator(events_file):
                for v in e.summary.value:
                    # if v.tag == 'SC_Qvalue':
                    if v.tag == 'FC_Qvalue':
                        modularity_list.append(v.simple_value)
                    if v.tag == 'perf_avg':
                        perf_avg_list.append(v.simple_value)
            
            if len(modularity_list) >= 93:
                modularity_array.append(np.array(modularity_list))
                perf_avg_array.append(np.array(perf_avg_list))
            else:
                print(f'{events_file} len(modularity_list):{len(modularity_list)}')                

        print(len(modularity_array))
        modularity_array = np.stack(modularity_array, axis=0)
        perf_avg_array = np.stack(perf_avg_array, axis=0)

        perf_avg_mean = np.mean(perf_avg_array, axis=0)
        perf_avg_std = np.std(perf_avg_array, axis=0)
        
        perf_avg_ste = perf_avg_std / np.sqrt(perf_avg_array.shape[0])
        

        modularity_mean = np.mean(modularity_array, axis=0)
        modularity_std = np.std(modularity_array, axis=0)
        modularity_ste = modularity_std / np.sqrt(modularity_array.shape[0])
        
        print(f'n_rnn:{n_rnn}, task_num:{task_num} avg_perf:{perf_avg_mean.mean():.4f}, avg_moduarlity:{modularity_mean.mean():.4f}')
        
        # 生成要显示的标签位置
        x_ticks = [ i for i in range(20, modularity_array.shape[1]+1, 20)]
        x_ticks = [0] + x_ticks
        x_tick_labels = [500*i for i in x_ticks]

        
        color = color_dict[tid_]
        
        # 绘制Modularity的均值和标准误
        axs.set_xticks(x_ticks)
        axs.set_xticklabels(x_tick_labels, rotation=45, fontsize=5)
        
        task_num_ = num_list[tid_]
        line_label = f'# Tasks: {task_num_:2}'
        axs.plot(modularity_mean, label=line_label, color=color, linewidth=0.25)
        axs.fill_between(range(modularity_array.shape[1]), modularity_mean - modularity_ste, modularity_mean + modularity_ste, color=color, alpha=0.2)
        

    axs.spines['left'].set_position('zero')
    axs.spines['bottom'].set_position('zero') 
    axs.spines['top'].set_linewidth(0.25)    
    axs.spines['bottom'].set_linewidth(0.25) 
    axs.spines['left'].set_linewidth(0.25)  
    axs.spines['right'].set_linewidth(0.25)  
    axs.tick_params(axis='both', labelsize=5)
    axs.tick_params(axis='both', width=0.25)
    
    # 设置子图的标题、轴标签和图例
    axs.set_title(f'# Hidden Neurons: {n_rnn}', fontsize=6)  
    axs.set_xlabel('Iterations', fontsize=6)    
    axs.set_ylabel('Modularity', fontsize=6)    
    axs.legend(loc='lower right', bbox_to_anchor=(0.98, 0.05), frameon=False, fontsize=6)
        
    plt.tight_layout()
    # 保存为SVG格式
    plt.savefig(f"./figures/Fig2/Fig2de_{n_rnn}.svg", format='svg', dpi=300)
    # 保存为JPG格式
    plt.savefig(f"./figures/Fig2/Fig2de_{n_rnn}.jpg", format='jpg', dpi=300)