import numpy as np
import bct
import torch
from functions.utils.eval_utils import lock_random_seed
import matplotlib.pyplot as plt
import pdb
import matplotlib
from matplotlib import font_manager 

fonts_path = '/home/wyuhang/.conda/myfonts'
font_files = font_manager.findSystemFonts(fontpaths=fonts_path)

for file in font_files:
    font_manager.fontManager.addfont(file)

matplotlib.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams["font.sans-serif"] = 'Myriad Pro'
matplotlib.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    lock_random_seed(2024)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model_size = 16
    seed = 100
    step = 38000
    
    task_num_list = [3, 6, 11, 16, 20]
    for task_num in task_num_list:
        file_name = f'./runs/Fig2bcde_data/n_rnn_{model_size}_task_{task_num}_seed_{seed}/RNN_interleaved_learning_{step}.pth'
        model = torch.load(file_name, device)   
        weights = model.recurrent_conn.weight.data.detach().cpu().numpy()
        cluster_id, sc_qvalue = bct.modularity_dir(np.abs(weights))
        weights = np.abs(weights)
        sorted_indices = np.argsort(cluster_id)
        sorted_matrix = weights[sorted_indices][:, sorted_indices]
        
        # axs, fig = plt.subplots(figsize=(1.3, 1.3))
        fig = plt.figure(figsize=(1.3, 1.3))

        # plt.imshow(sorted_matrix, cmap='Oranges_r', interpolation='nearest')  
        # plt.imshow(sorted_matrix, cmap='YlOrBr', interpolation='nearest')  
        im = plt.imshow(sorted_matrix, interpolation='nearest')
        

        ticks_range = np.arange(0, len(sorted_matrix), 2)
        plt.xticks(ticks_range, ticks_range, fontsize=5)
        plt.yticks(ticks_range, ticks_range, fontsize=5)
        
        plt.xlabel('Neurons', fontsize=5, labelpad=0)
        plt.ylabel('Neurons', fontsize=5, labelpad=0)
        
        # 手动创建 colorbar 并指定其位置，调整 [left, bottom, width, height] 参数
        # cbar_ax = fig.add_axes([0.93, 0.28, 0.035, 0.6]) 
        # cbar = fig.colorbar(im, cax=cbar_ax)

        cbar = plt.colorbar(im, fraction=0.0435, pad=0.10)  # 使用 fraction 和 pad 调整大小和位置
                
        tick_values = [0.2, 0.4, 0.6]  
        cbar.set_ticks(tick_values)
        cbar.ax.yaxis.set_tick_params(labelsize=5)  # 控制 colorbar 刻度字体大小
        cbar.outline.set_linewidth(0.25)  # 设置边框的线宽为2
        cbar.ax.yaxis.set_tick_params(width=0.25, length=1.0)
        cbar.ax.yaxis.set_tick_params(pad=0)

        cbar.set_label('Weight Magnitude', labelpad=0, fontsize=5)
        cbar.ax.yaxis.set_label_position('left')

        plt.tight_layout()
        axs = plt.gca()
        axs.tick_params(axis='both', width=0.25, length=1.0)
        axs.tick_params(axis='x', pad=0)
        axs.tick_params(axis='y', pad=0)
        axs.spines['top'].set_linewidth(0.25)    
        axs.spines['bottom'].set_linewidth(0.25) 
        axs.spines['left'].set_linewidth(0.25)  
        axs.spines['right'].set_linewidth(0.25) 
        
        
        plt.savefig(f'./figures/Fig2/Fig2f_{task_num}.jpg', format='jpg', dpi=300)
        plt.savefig(f'./figures/Fig2/Fig2f_{task_num}.svg', format='svg', dpi=300)
        print(f'step:{step}, task_num:{task_num}, modularity:{sc_qvalue}')