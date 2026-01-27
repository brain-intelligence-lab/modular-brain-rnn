import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
from functions.utils.plot_utils import get_seed_avg
from matplotlib.patches import Patch
import os

matplotlib.rcParams['pdf.fonttype'] = 42



def plot_fig2a(model_size_list, color_dict=None):
    directory_name = "./runs/Fig2a_data_RNN_relu"
    seed_list = [i for i in range(100, 2100, 100)]
    # 确保任务名称与您的文件系统/数据加载器中的名称完全匹配
    task_name_list = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                      'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                      'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                      'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

    # 1. 根据 Yang et al. (2019) Nature Neuroscience 的定义对任务进行分类
    task_groups = {
        'Go Family': ['fdgo', 'reactgo', 'delaygo'],
        'Anti Family': ['fdanti', 'reactanti', 'delayanti'],
        'DM Family': ['dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm'],
        'Dly DM Family': ['delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm'],
        'Matching Family': ['dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
    }
    
    # 创建一个从任务名到组名的反向映射
    task_to_group = {task: group for group, tasks in task_groups.items() for task in tasks}

    # 2. 如果未提供颜色字典，则创建一个默认的
    if color_dict is None:
        # 使用一个视觉上区分度高的色板
        group_colors = plt.cm.get_cmap('tab10', len(task_groups))
        color_dict = {group: group_colors(i) for i, group in enumerate(task_groups.keys())}

    save_dir = "./figures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for model_idx, model_size in enumerate(model_size_list):
        modularity_mp = {}

        for task_idx, task_name in enumerate(task_name_list):
            modularity_seed_array, _ = get_seed_avg(directory_name,
                model_size, task=task_name, seed_list=seed_list, chance_flag=False)
            
            # **修改点 1：为箱线图收集每个种子（seed）的最大模块化值**
            # modularity_mp[task_name] = modularity_seed_array.mean(axis=1) # 原来的代码
            modularity_mp[task_name] = modularity_seed_array.max(axis=1) # 形状: (num_seeds,)

        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 准备箱线图数据和颜色
        data_for_boxplot = [modularity_mp[task] for task in task_name_list]
        box_colors = [color_dict[task_to_group[task]] for task in task_name_list]

        # **修改点 2：使用 ax.boxplot 替换 ax.bar**
        bp = ax.boxplot(data_for_boxplot, 
                        patch_artist=True, # 允许填充颜色
                        vert=True,       # 垂直箱线图
                        showfliers=False) # 不显示离群值（通常为了图表更简洁）

        # 设置箱体颜色
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7) # 添加一些透明度
        
        # 可选：设置中位数线的样式
        for median in bp['medians']:
            median.set(color='black', linewidth=1.5)

        # 设置轴标签和标题
        ax.set_xticks(range(1, len(task_name_list) + 1)) # 箱线图的x轴从1开始
        ax.set_xticklabels(task_name_list, rotation=90, fontsize=13)
        
        # **更新 Y 轴标签**
        # ax.set_ylabel('Max Modularity (Q)', fontsize=12) 
        ax.set_ylabel('Mean Modularity (Q)', fontsize=15) 
        ax.set_xlabel('Task', fontsize=15)
        
        # **更新标题**
        # ax.set_title(f'Task-elicited Max Modularity Distribution (Model Size: {model_size})', fontsize=14)
        ax.set_title(f'Task-elicited Mean Modularity Distribution (Model Size: {model_size})', fontsize=17)


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(x=0.01)
        # ax.set_ylim(bottom=0.15)
        ax.set_ylim(bottom=0.05)

        # 图例保持不变
        legend_elements = [Patch(facecolor=color_dict[group], label=group)
                           for group in task_groups.keys()]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), fontsize=15, title_fontsize=16,
            loc='upper left', title="Task Families\n(Yang et al., 2019)")

        plt.tight_layout()
        fig_path = os.path.join(save_dir, f'Fig2a_modularity_boxplot_by_task_size_{model_size}.svg')
        plt.savefig(fig_path, dpi=300)

        fig_path = os.path.join(save_dir, f'Fig2a_modularity_boxplot_by_task_size_{model_size}.png')
        plt.savefig(fig_path, dpi=300)
        print(f"Figure saved to {fig_path}")
        plt.show()


# --- 使用示例 ---
if __name__ == '__main__':
    model_sizes_to_test = [64, 32, 16, 8] 
    
    # (可选) 自定义颜色
    custom_colors = {
        'Go Family': 'C0', # Blue
        'Anti Family': 'C1', # Orange
        'DM Family': 'C2', # Green
        'Dly DM Family': 'C3', # Red
        'Matching Family': 'C4' # Purple
    }
    
    plot_fig2a(model_sizes_to_test, color_dict=custom_colors)