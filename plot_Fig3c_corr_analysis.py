import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import time
import pdb
import tensorflow as tf
import statsmodels.stats.multitest as smm

import matplotlib
from matplotlib import font_manager 
import os

fonts_path = '/home/wyuhang/.conda/myfonts'
font_files = font_manager.findSystemFonts(fontpaths=fonts_path)

for file in font_files:
    font_manager.fontManager.addfont(file)

matplotlib.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams["font.sans-serif"] = 'Myriad Pro'
matplotlib.rcParams['pdf.fonttype'] = 42


def list_files(directory):
    path_list = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()  # 对目录列表进行就地排序
        files.sort()  # 对文件列表进行就地排序
        files.reverse()
        for file in files:
            if '.pth' in file or '.txt' in file:
                continue
            path_list.append(os.path.join(root, file))
            break

    return path_list


task_num = 20

# model_size_list = [8, 10, 15, 16, 30, 32, 64]
model_size_list = [8, 16, 32, 64]
for m_idx, n_rnn in enumerate(model_size_list):
    # paths = list_files(f"./runs/seed_search_0.1_relu_{n_rnn}")
    paths = list_files(f"./runs/Fig3c_seed_search_0.1_relu_{n_rnn}")
    
    r_list = []
    p_list = []
    step_range = range(500, 20500, 500)

    for step in step_range:
        
        modularity_array = []
        perf_avg_array = []
        perf_var_array = []
        
        for idx, events_file in enumerate(paths):
            # if idx == 100:
            #     break
            
            modularity = -1
            perf_avg = -1
            
            perf_list = []
            
            for e in tf.compat.v1.train.summary_iterator(events_file):

                if e.step != step:
                    continue
                
                for v in e.summary.value:
                    if v.tag == 'SC_Qvalue':
                        modularity = v.simple_value
                    if v.tag == 'perf_avg':
                        perf_avg = v.simple_value
                    # elif v.tag != 'perf_min' and 'perf' in v.tag:
                    #     perf_list.append(v.simple_value)
                        
            if modularity != -1:
                modularity_array.append(modularity)
                perf_avg_array.append(perf_avg)
                # perf_var = np.var(perf_list)
                # perf_var_array.append(perf_var)
                

        modularity_array = np.array(modularity_array)
        perf_avg_array = np.array(perf_avg_array)
        # perf_var_array = np.array(perf_var_array)
        
        print(perf_avg_array.mean())

        r, p = stats.pearsonr(modularity_array, perf_avg_array)
        print(f'step:{step}, r:{r:.4f}, p:{p:.4f}, len:{len(modularity_array)}')
        
        r_list.append(r)
        p_list.append(p)

    sort_p_list = sorted(p_list)

    m = len(p_list)

    # 计算 FDR threshold
    thresholds = np.arange(1, m+1) / m * 0.05

    sort_p_list = np.array(sort_p_list)
    # 找出最大满足 p(i) <= threshold 的 p-value
    significant = sort_p_list <= thresholds

    if np.any(significant):
        fdr_threshold = sort_p_list[significant][-1]
    else:
        fdr_threshold = 0.05
    
    # for i in range(m):
    #     q_i = ( (i+1) / m ) * p_value_threshold
    #     if sort_p_list[i] <= q_i:
    #         threshold_after_correction = sort_p_list[i]

    print(fdr_threshold)

    # 生成要显示的标签位置
    x_ticks = [ i for i in range(19, len(r_list)+1, 20)]

    x_tick_labels = [500*(i+1) for i in x_ticks]
    
    fig = plt.figure(figsize=(1.5, 3.0))
    
    rows = 2

    axes = []
    for i in range(rows):
        axs = plt.subplot2grid((rows, 1), (i, 0), rowspan=1, colspan=1)
        axes.append(axs)

    axes[0].bar(range(len(r_list)), r_list, color='lightsteelblue')
    axes[0].set_title(f'# Hidden Neurons: {n_rnn}', fontsize=7)
    axes[0].set_ylabel('Correlation', fontsize=6, labelpad=1)
    
    logp_value = -np.log10(fdr_threshold)
    axes[1].axhline(y=logp_value, color='green', linestyle='--', linewidth=0.25)
    axes[1].set_ylim([0, 5])  
    p_list = [-np.log10(value) for value in p_list]
    axes[1].bar(range(len(p_list)), p_list, color='lightgray')    
    axes[1].set_ylabel('-log(p)', fontsize=6, labelpad=1)

    for j in range(2):
        axes[j].set_xticklabels(x_tick_labels)
        axes[j].set_xticks(x_ticks)
        axes[j].spines['top'].set_linewidth(0.25)    
        axes[j].spines['bottom'].set_linewidth(0.25) 
        axes[j].spines['left'].set_linewidth(0.25)  
        axes[j].spines['right'].set_linewidth(0.25)  
        axes[j].tick_params(axis='both', width=0.25, length=1.0, labelsize=5, pad=1)
        axes[j].set_xlabel('Iterations', fontsize=6, labelpad=1)

    # 调整布局
    plt.tight_layout()
    plt.savefig(f"./figures/Fig3/Fig3c/Correlation_bar_{n_rnn}.svg", format='svg', dpi=300)
    plt.savefig(f"./figures/Fig3/Fig3c/Correlation_bar_{n_rnn}.jpg", format='jpg', dpi=300)
