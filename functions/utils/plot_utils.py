import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

def get_seed_avg(directory_name, model_size, task, seed_list, chance_flag=False):
    seed_paths_list = []
    for s_idx, seed_name in enumerate(seed_list):
        if chance_flag:
            file_name = f"chance_n_rnn_{model_size}_task_{task}_seed_{seed_name}"
        else:
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

def plot_fig(directory_name, seed_list, task_name_list, model_size_list, ylabel, **kwargs):
    plot_perf = kwargs.get('plot_perf', True)
    linelabel = kwargs.get('linelabel', None)
    linewidth = kwargs.get('linewidth', 1.0)
    color_dict = kwargs.get('color_dict', None)
    chance_flag = kwargs.get('chance_flag', False)
    y_lim_perf = kwargs.get('y_lim_perf', 1.0)
    y_lim_mod = kwargs.get('y_lim_mod', 0.25)


    for model_idx, model_size in enumerate(model_size_list):
        modularity_all_array = []
        perf_avg_all_array = []
        
        for task_idx, task_name in enumerate(task_name_list):
            modularity_seed_array, perf_avg_seed_array = get_seed_avg(directory_name, \
                model_size, task=task_name, seed_list=seed_list, chance_flag=chance_flag)

            modularity_all_array.append(modularity_seed_array)
            perf_avg_all_array.append(perf_avg_seed_array)
        

        # epochs_num = modularity_all_array[0].shape[-1]
        modularity_all_array = np.array(modularity_all_array).reshape(-1, modularity_all_array[0].shape[-1])
        perf_avg_all_array = np.array(perf_avg_all_array).reshape(-1, perf_avg_all_array[0].shape[-1])
        
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
            label = f'# Hidden Neurons: {model_size}' if linelabel is None else linelabel
            label = None if chance_flag else label
            plt.plot(perf_avg_mean, label = label, \
                color=color, linewidth=linewidth, linestyle=(0, (2, 5)) if chance_flag else '-')
            
            plt.xticks(ticks=x_ticks, labels=x_tick_labels, fontsize=6)
            y_ticks = np.arange(0.0, y_lim_perf+0.1, 0.2)  # 注意，终点设置为1.1以包括1.0
            plt.ylim([0.0, y_lim_perf])  # 设置y轴的范围
            y_ticks_labels = [f"{tick:.1f}" for tick in y_ticks]  # 格式化标签为一位小数
            plt.yticks(ticks=y_ticks, labels=y_ticks_labels, fontsize=6)        
            plt.ylabel(f'{ylabel}', fontsize=7)
            if not chance_flag:
                plt.fill_between(range(perf_avg_seed_array.shape[1]), perf_avg_mean - perf_avg_ste,\
                    perf_avg_mean + perf_avg_ste, color=color, alpha=0.2)
        else:
            if 'Multi' in linelabel:
                last_key, color = next(reversed(color_dict.items()))
            else:
                first_key, color = next(iter(color_dict.items()))

            label = f'# Hidden Neurons: {model_size}' if linelabel is None else linelabel
            label = None if chance_flag else label    
            plt.plot(modularity_mean, label = label, \
                color=color, linewidth=linewidth, linestyle=':' if chance_flag else '-')
            plt.xticks(ticks=x_ticks, labels=x_tick_labels, fontsize=6)

            y_ticks = np.arange(0.0, y_lim_mod+0.05, 0.05)  
            plt.ylim([0.0, y_lim_mod])  
            # y_ticks = np.arange(0.0, 0.60, 0.05)  
            # plt.ylim([0.0, 0.55])  
            y_ticks_labels = [f"{tick:.2f}" for tick in y_ticks]  # 格式化标签为一位小数
            plt.yticks(ticks=y_ticks, labels=y_ticks_labels, fontsize=6) 
            plt.ylabel(f'{ylabel}', fontsize=7)
            if not chance_flag:
                plt.fill_between(range(modularity_seed_array.shape[1]), modularity_mean - modularity_ste, \
                                modularity_mean + modularity_ste, color=color, alpha=0.2)
            
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