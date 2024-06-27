import numpy as np 
import matplotlib
from matplotlib import font_manager 
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
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

def plot_fig(directory_name, seed_list, task_name_list, model_size_list, ylabel, plot_perf=True, plot_mod=False, linelabel=None):
    for _, model_size in enumerate(model_size_list):
        modularity_all_array = []
        perf_avg_all_array = []
        
        for _, task_name in enumerate(task_name_list):
        
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
        # x_tick_labels = [500*64*i for i in x_ticks]
        x_tick_labels = [500*i for i in x_ticks]
        
                
        if plot_perf:
            plt.plot(perf_avg_mean, label = f'N={model_size}' if linelabel is None else linelabel)
            plt.xticks(ticks=x_ticks, labels=x_tick_labels)
            y_ticks = np.arange(0.0, 1.1, 0.2)  # 注意，终点设置为1.1以包括1.0
            plt.ylim([0.0, 1.0])  # 设置y轴的范围从0.0到1.0
            y_ticks_labels = [f"{tick:.1f}" for tick in y_ticks]  # 格式化标签为一位小数
            plt.yticks(ticks=y_ticks, labels=y_ticks_labels)        
            plt.ylabel(f'{ylabel}')
            plt.fill_between(range(perf_avg_seed_array.shape[1]), perf_avg_mean - perf_avg_ste, perf_avg_mean + perf_avg_ste, alpha=0.2)
        else:
            plt.plot(modularity_mean, label = f'N={model_size}' if linelabel is None else linelabel )
            plt.xticks(ticks=x_ticks, labels=x_tick_labels)
            plt.ylabel(f'{ylabel}')
            plt.fill_between(range(modularity_seed_array.shape[1]), modularity_mean - modularity_ste, modularity_mean + modularity_ste, alpha=0.2)
            
        plt.legend(loc='lower right')
        
    ax = plt.gca()
    ax.spines['left'].set_position('zero')  # 将左轴脊移动到0点
    ax.spines['bottom'].set_position('zero')  # 将底轴脊移动到0点

    # 隐藏上方和右方的轴脊
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    
def plot_fig2a():
    fig = plt.figure(figsize=(28, 16))
    directory_name = "./runs/Fig2a_data"
    seed_list = [ i for i in range(100, 1100, 100)]
    model_size_list = [10, 15, 20, 25, 30, 64]
    task_name_list = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
    
    plot_fig(directory_name, seed_list, task_name_list, model_size_list, ylabel='Avg performance')
    plt.tight_layout()
    fig.savefig('./figures/Fig2/Fig2a.svg', format='svg')
    fig.savefig('./figures/Fig2/Fig2a.jpg', format='jpg')

def plot_fig2b():
    fig = plt.figure(figsize=(28, 16))
    directory_name = "./runs/Fig2de_data"
    seed_list = [ i for i in range(100, 1100, 100)]
    model_size_list = [10, 15, 20, 25, 30, 64]
    task_num_list = [20]
    plot_fig(directory_name, seed_list, task_num_list, model_size_list, ylabel='Avg performance')
    plt.tight_layout()
    fig.savefig('./figures/Fig2/Fig2b.svg', format='svg')
    fig.savefig('./figures/Fig2/Fig2b.jpg', format='jpg')

def plot_fig2c(N=15):
    fig = plt.figure(figsize=(28, 16))
    model_size_list = [N]
    task_num_list = [20]
    directory_name = "./runs/Fig2de_data"
    seed_list = [ i for i in range(100, 1100, 100)]
    plot_fig(directory_name, seed_list, task_num_list, model_size_list, ylabel='Modularity', \
             plot_perf=False, plot_mod=True, linelabel=f'multitask_{N}')
    
    task_name_list = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

    directory_name = "./runs/Fig2a_data"
    
    plot_fig(directory_name, seed_list, task_name_list, model_size_list, ylabel='Modularity', \
        plot_perf=False, plot_mod=True, linelabel=f'singletask_{N}')
    plt.tight_layout()
    fig.savefig(f'./figures/Fig2/Fig2c_{N}.jpg', format='jpg')
    fig.savefig(f'./figures/Fig2/Fig2c_{N}.svg', format='svg')
    

plot_fig2a()
plot_fig2b()
for N in [10, 15, 20, 25, 30, 64]:
# for N in [15]:
    plot_fig2c(N)
