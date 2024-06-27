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
    assert len(path_list)==1, "path_list_error!"
    return path_list[0]


wiring_rules=["distance", "random", "dis_rand"]
conn_modes=["grow", "fix"]

# seed_list = [ i for i in range(100, 1100, 100)]
seed_list = [ i for i in range(100, 600, 100)]

task_num = 20
n_rnn = 84

fig = plt.figure(figsize=(14, 8))

directory_name = "./runs/gen_conn_compare"

fig = plt.figure(figsize=(12, 8))

ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)

ax2 = plt.subplot2grid((2, 4), (0, 2), rowspan=2, colspan=2)
# ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
# ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
# axes = [ax1, ax2, ax3]

axes = [ax1, ax2]

for _, wiring_rule in enumerate(wiring_rules):
    for __, conn_mode in enumerate(conn_modes):
        exp_paths_list = []
        for s_idx, seed in enumerate(seed_list):
            file_name = f"n_rnn_{n_rnn}_task_{task_num}_seed_{seed}_rule_{wiring_rule}_mode_{conn_mode}"
            paths = list_files(directory_name, file_name)
            exp_paths_list.append(paths)
    
        print(f"{len(exp_paths_list)}")
    
        modularity_array = []
        perf_avg_array = []
        for ii, events_file in enumerate(exp_paths_list):
            modularity_list = []
            perf_avg_list = []
            
            
            for e in tf.compat.v1.train.summary_iterator(events_file):
                for v in e.summary.value:
                    if v.tag == 'SC_Qvalue':
                        modularity_list.append(v.simple_value)
                    if v.tag == 'perf_avg':
                        perf_avg_list.append(v.simple_value)
            
            if len(modularity_list) == 93:
                modularity_array.append(modularity_list)
                perf_avg_array.append(perf_avg_list)
            else:
                pdb.set_trace()
                
        
        modularity_array = np.array(modularity_array)
        perf_avg_array = np.array(perf_avg_array)
        
        modularity_mean = np.mean(modularity_array, axis=0)
        modularity_std = np.std(modularity_array, axis=0)
        modularity_ste = modularity_std / np.sqrt(modularity_array.shape[0])
        
        perf_avg_mean = np.mean(perf_avg_array, axis=0)
        perf_avg_std = np.std(perf_avg_array, axis=0)
        perf_avg_ste = perf_avg_std / np.sqrt(perf_avg_array.shape[0])
        
        # 生成要显示的标签位置
        x_ticks = [ i for i in range(10, modularity_array.shape[1]+1, 10)]
        x_ticks = [1] + x_ticks
        x_tick_labels = [500*64*i for i in x_ticks]
        

        # 绘制Modularity的均值和标准误
        axes[0].set_xticklabels(x_tick_labels, rotation=45)
        axes[0].set_xticks(x_ticks)
        axes[0].plot(modularity_mean, label=f'{wiring_rule}_{conn_mode}')
        axes[0].fill_between(range(modularity_array.shape[1]), modularity_mean - modularity_ste, modularity_mean + modularity_ste, alpha=0.2)
        
        # if wiring_rule != 'random':
        #     continue
        
        # 绘制Perf的均值和标准误
        axes[1].set_xticklabels(x_tick_labels, rotation=45)
        axes[1].set_xticks(x_ticks)
        axes[1].plot(perf_avg_mean, label=f'{wiring_rule}_{conn_mode}')
        axes[1].fill_between(range(modularity_array.shape[1]), perf_avg_mean - perf_avg_ste, perf_avg_mean + perf_avg_ste, alpha=0.2)
        
        
        
        # # 绘制Perf_avg的均值和标准误
        # if conn_mode == 'fix':    
        #     axes[1].set_xticklabels(x_tick_labels, fontsize=8)
        #     axes[1].set_xticks(x_ticks)
        #     axes[1].plot(perf_avg_mean, label=f'{wiring_rule}_{conn_mode}')
        #     axes[1].fill_between(range(perf_avg_array.shape[1]), perf_avg_mean - perf_avg_ste, perf_avg_mean + perf_avg_ste, alpha=0.2)
        # else:
        #     axes[2].set_xticklabels(x_tick_labels, fontsize=8)
        #     axes[2].set_xticks(x_ticks)
        #     axes[2].plot(perf_avg_mean, label=f'{wiring_rule}_{conn_mode}')
        #     axes[2].fill_between(range(perf_avg_array.shape[1]), perf_avg_mean - perf_avg_ste, perf_avg_mean + perf_avg_ste, alpha=0.2)
            
        

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)


axes[0].text(0.02, 0.98, 'A', transform=axes[0].transAxes, fontsize=16, fontweight='bold', va='top')

# 设置子图的标题、轴标签和图例
axes[0].set_title('Modularity')
axes[0].set_xlabel('Trials')
axes[0].set_ylabel('Modularity')
axes[0].legend(loc='lower right', frameon=False)


axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

axes[1].text(0.02, 0.98, 'B', transform=axes[1].transAxes, fontsize=16, fontweight='bold', va='top')
axes[1].set_title('Avg performance')
axes[1].set_xlabel('Trials')
axes[1].set_ylabel('Performance')
axes[1].legend(loc='lower right', frameon=False)


# axes[2].spines['top'].set_visible(False)
# axes[2].spines['right'].set_visible(False)
# axes[2].spines['left'].set_visible(False)
# axes[2].spines['bottom'].set_visible(False)

# axes[2].text(0.02, 0.98, 'C', transform=axes[2].transAxes, fontsize=16, fontweight='bold', va='top')
# axes[2].set_title('Avg performance')
# axes[2].set_xlabel('Trials')
# axes[2].set_ylabel('Performance')
# axes[2].legend(loc='lower right', frameon=False)


# 调整布局
plt.tight_layout()

# 保存为SVG格式
plt.savefig("growth.svg", format='svg')
plt.savefig("growth.jpg", format='jpg')



