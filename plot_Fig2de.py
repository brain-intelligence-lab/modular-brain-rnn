import numpy as np 
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
import matplotlib
from matplotlib import font_manager 
import os

fonts_path = '/home/wyuhang/.conda/myfonts'
font_files = font_manager.findSystemFonts(fontpaths=fonts_path)

for file in font_files:
    font_manager.fontManager.addfont(file)

matplotlib.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams["font.sans-serif"] = 'Myriad Pro'
plt.rcParams['font.size'] = 16


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


task_num_list = [f"task_{i}" for i in [3, 6, 11, 16, 20]]

# seed_name_list = [f"seed_{i}" for i in range(100, 1050, 100)]
seed_name_list = [f"seed_{i}" for i in range(100, 2050, 100)]

# model_size_list = [10, 15, 20, 30, 64, 128]
model_size_list = [10, 128]
# model_size_list = [10, 64]

fig = plt.figure(figsize=(28, 16))

# rows = 2
# cols = len(model_size_list)
# axes = []
# for i in range(rows):
#     for j in range(cols):
#         axs = plt.subplot2grid((rows*2, cols*2), (i*2, j*2), rowspan=2, colspan=2)
#         axes.append(axs)

rows = 1
cols = len(model_size_list)
axes = []
for i in range(rows):
    for j in range(cols):
        axs = plt.subplot2grid((rows, cols), (i, j), rowspan=1, colspan=1)
        axes.append(axs)


for idx, n_rnn in enumerate(model_size_list):
    n_rnn_name = f"n_rnn_{n_rnn}"
    
    for _, task_num in enumerate(task_num_list):
        exp_paths_list = []
        for __, seed_name in enumerate(seed_name_list):
            file_name = f"{n_rnn_name}_{task_num}_{seed_name}"
            paths = list_files("./runs/Fig2de_data/", file_name)
            if paths != None:
                exp_paths_list.append(paths)
        
        # print(f"{len(exp_paths_list)}")
    
        modularity_array = []
        perf_avg_array = []
        cluster_num_array = []
        for ii, events_file in enumerate(exp_paths_list):
            modularity_list = [0]
            perf_avg_list = [0]
            cluster_num_list = [0]
            
            for e in tf.compat.v1.train.summary_iterator(events_file):
                for v in e.summary.value:
                    if v.tag == 'SC_Qvalue':
                        modularity_list.append(v.simple_value)
                    # if v.tag == 'FC_Qvalue':
                    #     modularity_list.append(v.simple_value)
                    if v.tag == 'perf_avg':
                        perf_avg_list.append(v.simple_value)
                    if v.tag == 'Cluster_Num':
                        cluster_num_list.append(v.simple_value)
            
            if len(modularity_list) >= 93:
                modularity_array.append(np.array(modularity_list))
                perf_avg_array.append(np.array(perf_avg_list))
                cluster_num_array.append(np.array(cluster_num_list))
            else:
                print(f'{events_file} len(modularity_list):{len(modularity_list)}')
                # pdb.set_trace()
                

        print(len(modularity_array))
        modularity_array = np.stack(modularity_array, axis=0)
        perf_avg_array = np.stack(perf_avg_array, axis=0)
        cluster_num_array = np.stack(cluster_num_array, axis=0) 
        

        perf_avg_mean = np.mean(perf_avg_array, axis=0)
        perf_avg_std = np.std(perf_avg_array, axis=0)
        
        perf_avg_ste = perf_avg_std / np.sqrt(perf_avg_array.shape[0])
        

        modularity_mean = np.mean(modularity_array, axis=0)
        modularity_std = np.std(modularity_array, axis=0)
        modularity_ste = modularity_std / np.sqrt(modularity_array.shape[0])
        
        cluster_num_mean = np.mean(cluster_num_array, axis=0)
        cluster_num_std = np.std(cluster_num_array, axis=0)
        cluster_num_ste = cluster_num_std / np.sqrt(cluster_num_array.shape[0])
        
        print(f'n_rnn:{n_rnn}, task_num:{task_num} avg_perf:{perf_avg_mean.mean():.4f}, avg_moduarlity:{modularity_mean.mean():.4f}')
        
        # 生成要显示的标签位置
        x_ticks = [ i for i in range(20, modularity_array.shape[1]+1, 20)]
        x_ticks = [0] + x_ticks
        x_tick_labels = [500*i for i in x_ticks]
        

        # 绘制Modularity的均值和标准误
        axes[idx].set_xticklabels(x_tick_labels, rotation=45)
        axes[idx].set_xticks(x_ticks)
        axes[idx].plot(modularity_mean, label=task_num)
        axes[idx].fill_between(range(modularity_array.shape[1]), modularity_mean - modularity_ste, modularity_mean + modularity_ste, alpha=0.2)
        
        # # 绘制perf的均值和标准误
        # axes[cols + idx].set_xticklabels(x_tick_labels, rotation=45)
        # axes[cols + idx].set_xticks(x_ticks)
        # axes[cols + idx].plot(perf_avg_mean, label=task_num)
        # axes[cols + idx].fill_between(range(perf_avg_array.shape[1]), perf_avg_mean - perf_avg_ste, perf_avg_mean + perf_avg_ste, alpha=0.2)
        

for idx, n_rnn in enumerate(model_size_list):
    
    ax = plt.gca()
    ax.spines['left'].set_position('zero')  # 将左轴脊移动到0点
    ax.spines['bottom'].set_position('zero')  # 将底轴脊移动到0点


    axes[idx].spines['top'].set_visible(False)
    axes[idx].spines['right'].set_visible(False)
    
    axes[idx].spines['left'].set_position('zero')
    axes[idx].spines['bottom'].set_position('zero') 
    
    # axes[cols + idx].spines['top'].set_visible(False)
    # axes[cols + idx].spines['right'].set_visible(False)
    
    # axes[cols + idx].spines['left'].set_position('zero') 
    # axes[cols + idx].spines['bottom'].set_position('zero') 

    # text_list = ['A' ,'B', 'C', 'D']
    
    # axes[idx].text(0.02, 0.98, text_list[i], transform=axes[i].transAxes, fontsize=16, fontweight='bold', va='top')

    # 设置子图的标题、轴标签和图例
    axes[idx].set_title(f'Model size {n_rnn}')
    axes[idx].set_xlabel('Trials')
    axes[idx].set_ylabel('Modularity')
    axes[idx].legend(loc='lower right', frameon=False)
        
    # axes[cols + idx].set_title(f'Model size {n_rnn}')
    # axes[cols + idx].set_xlabel('Trials')
    # axes[cols + idx].set_ylabel('Avg_performance')
    # axes[cols + idx].legend(loc='lower right', frameon=False)
    

# 调整布局
plt.tight_layout()

# 保存为SVG格式
plt.savefig("./figures/Fig2/Fig2de.svg", format='svg')
# 保存为JPG格式
plt.savefig("./figures/Fig2/Fig2de.jpg", format='jpg')