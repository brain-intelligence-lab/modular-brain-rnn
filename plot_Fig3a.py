import numpy as np 
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
import argparse
import matplotlib
from matplotlib import font_manager 

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
                
    if len(path_list)!=1:
        pdb.set_trace()

    return path_list[0]

def start_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default=1, type=int, choices=[1,2,3])
    parser.add_argument('--gpu', default=1, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = start_parse()
    
    seed_list = [ i for i in range(100, 1100, 100)]

    if args.case == 1:
        directory_name = "./runs/Fig3a_go"
        task_name_list = ['fdgo', 'fdanti', 'fdgo_fdanti']
        model_size_list = [4, 10]
    elif args.case == 2:
        directory_name = "./runs/Fig3a_contextdm"
        task_name_list = ['contextdm1', 'contextdm2', 'contextdm1_contextdm2']
        model_size_list = [26, 44]
    elif args.case == 3:
        directory_name = "./runs/Fig3a_go"
        task_name_list = ['fdgo', 'delaygo', 'fdgo_delaygo']
        model_size_list = [4, 6]
    
    task_name_abbreviation = {'fdgo': 'Go', 'reactgo': 'RT Go', 'delaygo': 'Dly Go', 'fdanti': 'Anti', 'reactanti': 'RT Anti', 'delayanti': 'Dly Anti',
                              'dmsgo': 'DMS', 'dmsnogo': 'DNMS', 'dmcgo': 'DMC', 'dmcnogo':'DNMC',
                            'dm1': 'DM 1', 'dm2': 'DM 2', 'contextdm1': 'Ctx DM1', 'contextdm2': 'Ctx DM2', 'multidm': 'MultSen DM',
                            'delaydm1': 'Dly DM 1', 'delaydm2': 'Dly DM 2',  'contextdelaydm1': 'Ctx Dly DM 1', 'contextdelaydm2': 'Ctx Dly DM 2',  'multidelaydm': 'MultSen Dly DM' }

    for m_idx, model_size in enumerate(model_size_list):
        fig, axs = plt.subplots(figsize=(1.4, 1.4))

        for t_idx, task_name in enumerate(task_name_list):
            seed_paths_list = []
            for s_idx, seed_name in enumerate(seed_list):
                file_name = f"n_rnn_{model_size}_task_{task_name}_seed_{seed_name}"
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
            perf_avg_std = np.std(perf_avg_seed_array, axis=0)
            perf_avg_ste = perf_avg_std / np.sqrt(perf_avg_seed_array.shape[0])
            
            modularity_mean = np.mean(modularity_seed_array, axis=0)
            modularity_std = np.std(modularity_seed_array, axis=0)
            modularity_ste = modularity_std / np.sqrt(modularity_seed_array.shape[0])
            
            print(f'n_rnn:{model_size}, avg_perf:{perf_avg_mean.mean():.4f}, avg_moduarlity:{modularity_mean.mean():.4f}')
            

            # 生成要显示的标签位置
            x_ticks = [i for i in range(20, perf_avg_seed_array.shape[1]+1, 20)]
            # x_ticks = [0] + x_ticks
            x_tick_labels = [500*i for i in x_ticks]

    
            axs.axhline(y=0.95, color='green', linestyle='--', linewidth=0.25)  # 添加虚线

            y_ticks, y_labels = plt.yticks()

            new_y_ticks = [tick for tick in y_ticks if tick != 1.0 and tick !=0.95]
            axs.set_yticks(new_y_ticks + [0.95])
            axs.set_ylim(0, 1.0)
            # axes[m_idx].set_yticks(list(axes[m_idx].get_yticks()) + [0.95])
            
            # 绘制perf的均值和标准误
            
            axs.set_xticklabels(x_tick_labels, rotation=45)
            axs.set_xticks(x_ticks)
            axs.set_xlim(0, 90)
            if '_' in task_name:
                task_name = task_name_abbreviation[task_name.split('_')[0]] + '_' + task_name_abbreviation[task_name.split('_')[1]]
                task_name = task_name.replace('_', '&\n')
            else:
                task_name = task_name_abbreviation[task_name]

            axs.plot(perf_avg_mean, label=task_name, linewidth=0.25)
            axs.fill_between(range(perf_avg_seed_array.shape[1]), perf_avg_mean - perf_avg_ste, perf_avg_mean + perf_avg_ste, alpha=0.2)
        

        axs.spines['left'].set_position('zero')
        axs.spines['bottom'].set_position('zero') 
        axs.spines['top'].set_linewidth(0.25)    
        axs.spines['bottom'].set_linewidth(0.25) 
        axs.spines['left'].set_linewidth(0.25)  
        axs.spines['right'].set_linewidth(0.25)  
        # axs.tick_params(axis='both', 
        axs.tick_params(axis='both', width=0.25, length=1.0, labelsize=5, pad=1)
        
        axs.set_title(f'# Hidden Neurons: {model_size}', fontsize=5, pad=2)
        axs.set_xlabel('Iterations', fontsize=5, labelpad=1)
        axs.set_ylabel('Avg performance', fontsize=5, labelpad=1)
        axs.legend(loc='lower right', bbox_to_anchor=(1.02, 0.02), frameon=False, fontsize=5)
        
        # 调整布局
        plt.tight_layout()

        plt.savefig(f"./figures/Fig3/Fig3a/Fig3a_case{args.case}_{m_idx}.svg", format='svg')
        plt.savefig(f"./figures/Fig3/Fig3a/Fig3a_case{args.case}_{m_idx}.jpg", format='jpg')
