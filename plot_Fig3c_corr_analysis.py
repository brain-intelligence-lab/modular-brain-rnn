import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import time
import pdb
import tensorflow as tf
import os


def list_files(directory):
    path_list = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()  # 对目录列表进行就地排序
        files.sort()  # 对文件列表进行就地排序
        for file in files:
            if '.pth' in file or '.txt' in file:
                continue
            path_list.append(os.path.join(root, file))

    return path_list


model_size_list = [10, 15, 20, 25, 30, 64]
# model_size_list = [30]
# m= 160
# task_num = 20

fig = plt.figure(figsize=(28, 16))

rows = 2
cols = len(model_size_list)

axes = []
for i in range(rows):
    for j in range(cols):
        axs = plt.subplot2grid((rows*2, cols*2), (i*2, j*2), rowspan=2, colspan=2)
        axes.append(axs)

for m_idx, n_rnn in enumerate(model_size_list):
    paths = list_files(f"./runs/seed_search_0.1_relu_{n_rnn}")
    # paths = list_files(f"./runs/seed_search_0.1_softplus_{n_rnn}")
    # paths = list_files(f"./runs/Fig4_topology_task/{n_rnn}_{m}_{task_num}")
    # paths = list_files(f"./runs/Fig4_topology")
    
    r_list = []
    p_list = []

    for step in range(500, 18000, 500):
    # for step in range(500, 47000, 500):
        perf_mean_array = []
        modularity_mean_array = []
        
        modularity_array = []
        perf_avg_array = []
        perf_var_array = []
        
        for idx, events_file in enumerate(paths):
            if idx == 100:
                break
            
            modularity = -1
            perf_avg = -1
            
            perf_list = []
            
            perf_mean_list = []
            modularity_mean_list = []
            
            for e in tf.compat.v1.train.summary_iterator(events_file):
                if e.step > 0:
                    for v in e.summary.value:
                        if v.tag == 'SC_Qvalue':
                            modularity_mean_list.append(v.simple_value)
                        if v.tag == 'perf_avg':
                            perf_mean_list.append(v.simple_value)
                
                if e.step != step:
                    continue
                
                for v in e.summary.value:
                    if v.tag == 'SC_Qvalue':
                        modularity = v.simple_value
                    if v.tag == 'perf_avg':
                        perf_avg = v.simple_value
                    elif v.tag!= 'perf_min' and 'perf' in v.tag:
                        perf_list.append(v.simple_value)
                        
            if modularity != -1:
                modularity_array.append(modularity)
                perf_avg_array.append(perf_avg)
                perf_var = np.var(perf_list)
                perf_var_array.append(perf_var)
                
                perf_mean_array.append(np.mean(perf_mean_list))
                modularity_mean_array.append(np.mean(modularity_mean_list))
                

        modularity_array = np.array(modularity_array)
        perf_avg_array = np.array(perf_avg_array)
        perf_var_array = np.array(perf_var_array)
        
        perf_mean_array = np.array(perf_mean_array)
        modularity_mean_array = np.array(modularity_mean_array)
        
        # r, p = stats.pearsonr(modularity_array, perf_mean_array)
        # print(f'step:{step}, r:{r:.4f}, p:{p:.4f}, len:{len(modularity_array)}')

        print(perf_avg_array.mean())

        r, p = stats.pearsonr(modularity_array, perf_avg_array)
        print(f'step:{step}, r:{r:.4f}, p:{p:.4f}, len:{len(modularity_array)}')


        # 绘制散点图，选择更美观的配色
        # plt.scatter(modularity_array, perf_avg_array, color='deepskyblue', edgecolors='black', s=50)

        # 计算拟合线
        # slope, intercept = np.polyfit(modularity_array, perf_avg_array, 1)
        # line = slope * modularity_array + intercept

        # 添加拟合线，选择红色并设置线条样式
        # plt.plot(modularity_array, line, color='darkred', linestyle='--', label=f'Fit Line: r={r:.2f}, p={p:.3f}')

        # 添加图例
        # plt.legend()

        # 添加标签和标题
        # plt.xlabel('Modularity')
        # plt.ylabel('Performance Average')
        # plt.title('Modularity vs Performance')

        # 设置背景色为浅灰色，提高视觉效果
        # plt.gca().set_facecolor('lightgrey')

        # 显示图形
        # plt.savefig(f'corr.svg',format='svg')


        r_list.append(r)
        p_list.append(p)

    # 生成要显示的标签位置
    x_ticks = [ i for i in range(10, len(r_list)+1, 10)]
    x_ticks = [1] + x_ticks
    x_tick_labels = [500*64*i for i in x_ticks]

    axes[m_idx].set_xticklabels(x_tick_labels, rotation=45)
    axes[m_idx].set_xticks(x_ticks)


    axes[m_idx].plot(r_list, label='Correlation')

    axes[m_idx+cols].set_xticklabels(x_tick_labels, rotation=45)
    axes[m_idx+cols].set_xticks(x_ticks)
    

    axes[m_idx+cols].axhline(y=0.05, color='green', linestyle='--', linewidth=1)  # 添加虚线
    axes[m_idx+cols].set_yticks(list(axes[m_idx+cols].get_yticks()) + [0.05])
    axes[m_idx+cols].set_ylim([0, 1])  # 设置 y 轴范围以清晰显示虚线
    axes[m_idx+cols].plot(p_list, label='P value', color='black')


    axes[m_idx].spines['top'].set_visible(False)
    axes[m_idx].spines['right'].set_visible(False)
    axes[m_idx].spines['left'].set_visible(False)
    axes[m_idx].spines['bottom'].set_visible(False)
    # axes[m_idx].text(0.02, 0.98, 'A', transform=axes[m_idx].transAxes, fontsize=16, fontweight='bold', va='top')

    # 设置子图的标题、轴标签和图例
    axes[m_idx].set_title(f'Correlation N={n_rnn}')
    axes[m_idx].set_xlabel('Trials')
    axes[m_idx].set_ylabel('corr')
    # axes[m_idx].legend(loc='upper right', frameon=False)


    axes[m_idx+cols].spines['top'].set_visible(False)
    axes[m_idx+cols].spines['right'].set_visible(False)
    axes[m_idx+cols].spines['left'].set_visible(False)
    axes[m_idx+cols].spines['bottom'].set_visible(False)

    # axes[m_idx+cols].text(0.02, 0.98, 'B', transform=axes[m_idx+cols].transAxes, fontsize=16, fontweight='bold', va='top')
    # axes[m_idx+cols].set_title('p_value')
    axes[m_idx+cols].set_xlabel('Trials')
    axes[m_idx+cols].set_ylabel('p_value')
    # axes[m_idx+cols].legend(loc='upper right', frameon=False)

# 调整布局
plt.tight_layout()


plt.savefig("./figures/Fig3/Fig3c/Correlation.svg", format='svg')
plt.savefig("./figures/Fig3/Fig3c/Correlation.jpg", format='jpg')

# plt.savefig(f"Correlation_{n_rnn}_{m}_{task_num}.svg", format='svg')
# plt.savefig(f"Correlation_{n_rnn}_{m}_{task_num}.jpg", format='jpg')