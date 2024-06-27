import numpy as np 
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf

import os

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


seed_list = [ i for i in range(100, 1100, 100)]

# task_name_list = ['fdgo', 'fdanti', 'fdgo_fdanti']
# task_name_list = ['fdgo', 'delaygo', 'fdgo_delaygo']
# task_name_list = ['dmcgo', 'dmcnogo', 'dmcgo_dmcnogo']
task_name_list = ['contextdm1', 'contextdm2', 'contextdm1_contextdm2']

# model_size_list = [8, 9, 10, 11, 12, 13, 14]

model_size_list = [24, 26, 28, 44, 46, 48]


fig = plt.figure(figsize=(28, 16))

rows = 2
cols = len(model_size_list)
axes = []
for i in range(rows):
    for j in range(cols):
        axs = plt.subplot2grid((rows*2, cols*2), (i*2, j*2), rowspan=2, colspan=2)
        axes.append(axs)

# directory_name = "./runs/Fig3a_go_"
# directory_name = "./runs/Fig3a_dnmc_"
# directory_name = "./runs/Fig3a_dnmc"
directory_name = "./runs/Fig3a_contextdm"

for m_idx, model_size in enumerate(model_size_list):

    for t_idx, task_name in enumerate(task_name_list):
        seed_paths_list = []
        for s_idx, seed_name in enumerate(seed_list):
            file_name = f"n_rnn_{model_size}_task_{task_name}_seed_{seed_name}"
            paths = list_files(directory_name, file_name)
            seed_paths_list.append(paths)
    
        modularity_seed_array = []
        perf_avg_seed_array = []
        
        for ii, events_file in enumerate(seed_paths_list):            
            modularity_list = []
            perf_avg_list = []

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
        perf_avg_std = np.std(perf_avg_seed_array, axis=0)
        perf_avg_ste = perf_avg_std / np.sqrt(perf_avg_seed_array.shape[0])
        
        modularity_mean = np.mean(modularity_seed_array, axis=0)
        modularity_std = np.std(modularity_seed_array, axis=0)
        modularity_ste = modularity_std / np.sqrt(modularity_seed_array.shape[0])
        
        print(f'n_rnn:{model_size}, avg_perf:{perf_avg_mean.mean():.4f}, avg_moduarlity:{modularity_mean.mean():.4f}')
        

        # 生成要显示的标签位置
        x_ticks = [i for i in range(10, perf_avg_seed_array.shape[1]+1, 10)]
        x_ticks = [1] + x_ticks
        x_tick_labels = [500*64*i for i in x_ticks]

        if '_' in task_name or True:
        # if '_' in task_name :
            # 绘制Modularity的均值和标准误
            axes[m_idx].set_xticklabels(x_tick_labels, rotation=45)
            axes[m_idx].set_xticks(x_ticks)
            axes[m_idx].plot(modularity_mean)
            axes[m_idx].fill_between(range(modularity_seed_array.shape[1]), modularity_mean - modularity_ste, modularity_mean + modularity_ste, alpha=0.2)
            
            
            axes[cols + m_idx].axhline(y=0.95, color='green', linestyle='--', linewidth=1)  # 添加虚线
            axes[cols + m_idx].set_yticks(list(axes[m_idx+cols].get_yticks()) + [0.95])
            
            # 绘制perf的均值和标准误
            axes[cols + m_idx].set_xticklabels(x_tick_labels, rotation=45)
            axes[cols + m_idx].set_xticks(x_ticks)
            axes[cols + m_idx].plot(perf_avg_mean, label=task_name)
            axes[cols + m_idx].fill_between(range(perf_avg_seed_array.shape[1]), perf_avg_mean - perf_avg_ste, perf_avg_mean + perf_avg_ste, alpha=0.2)
    

for idx, n_rnn in enumerate(model_size_list):

    axes[idx].spines['top'].set_visible(False)
    axes[idx].spines['right'].set_visible(False)
    axes[idx].spines['left'].set_visible(False)
    axes[idx].spines['bottom'].set_visible(False)
    
    axes[cols + idx].spines['top'].set_visible(False)
    axes[cols + idx].spines['right'].set_visible(False)
    axes[cols + idx].spines['left'].set_visible(False)
    axes[cols + idx].spines['bottom'].set_visible(False)

    # 设置子图的标题、轴标签和图例
    axes[idx].set_title(f'Model size {n_rnn}')
    axes[idx].set_xlabel('Trials')
    axes[idx].set_ylabel('Modularity')
    axes[idx].legend(loc='lower right', frameon=False)
    
    # axes[cols + idx].set_title(f'Model size {n_rnn}')
    axes[cols + idx].set_xlabel('Trials')
    axes[cols + idx].set_ylabel('Avg_performance')
    axes[cols + idx].legend(loc='lower right', frameon=False)
    

# 调整布局
plt.tight_layout()

# 保存为SVG格式
plt.savefig("./figures/Fig3/Fig3a/Fig3a.svg", format='svg')
# 保存为JPG格式
plt.savefig("./figures/Fig3/Fig3a/Fig3a.jpg", format='jpg')
