import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import pdb
import seaborn as sns
import tensorflow as tf

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
n_rnn = 8

step = 10000


plt.figure(figsize=(2.0, 2.0))

# paths = list_files(f"./runs/seed_search_0.1_relu_{n_rnn}")
paths = list_files(f"./runs/Fig3c_seed_search_0.1_relu_{n_rnn}")


modularity_array = []
perf_avg_array = []

for idx, events_file in enumerate(paths):
    
    modularity = -1
    perf_avg = -1
    
    for e in tf.compat.v1.train.summary_iterator(events_file):

        if e.step != step:
            continue
        
        for v in e.summary.value:
            if v.tag == 'SC_Qvalue':
                modularity = v.simple_value
            if v.tag == 'perf_avg':
                perf_avg = v.simple_value
                
    if modularity != -1:
        modularity_array.append(modularity)
        perf_avg_array.append(perf_avg)


modularity_array = np.array(modularity_array)
perf_avg_array = np.array(perf_avg_array)

print(perf_avg_array.mean())

r, p = stats.pearsonr(modularity_array, perf_avg_array)
print(f'step:{step}, r:{r:.4f}, p:{p:.6f}, len:{len(modularity_array)}')


# 设置颜色和样式
sns.regplot(
    x=modularity_array, 
    y=perf_avg_array, 
    scatter_kws={'s':1, 'color':'#2ca02c'},  # 柔和的浅绿色
    line_kws={'color':'#1f77ff', 'linewidth':0.5, 'alpha':1},  # 天蓝色，去掉阴影
    ci=None  # 去掉回归线的置信区间阴影
)

# 添加标题和标签
# plt.title('Modularity vs Performance: Correlation Analysis')
plt.xlabel('Modularity', fontsize=6)
plt.ylabel('Performance', fontsize=6)

axs = plt.gca()


axs.spines['top'].set_linewidth(0.25)    
axs.spines['bottom'].set_linewidth(0.25) 
axs.spines['left'].set_linewidth(0.25)  
axs.spines['right'].set_linewidth(0.25)  
axs.tick_params(axis='both', width=0.25, length=1.0, labelsize=5, pad=1)

# 添加 Pearson r 和 p 值信息
plt.text(
    0.70, 0.95,  # x, y 位置（相对坐标）
    f'r = {r:.5f}\np = {p:.2e}',  # 显示的文本
    transform=plt.gca().transAxes,  # 使用相对坐标
    verticalalignment='top',  # 文本顶部对齐
    fontsize=5,  # 这里指定文本的字体大小
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', linewidth=0.25)  # 边框设置
)
# 调整布局


plt.tight_layout()

plt.savefig("./figures/Fig3/Fig3c/Correlation_Scatter_Plot.svg", format='svg', dpi=300)
plt.savefig("./figures/Fig3/Fig3c/Correlation_Scatter_Plot.jpg", format='jpg', dpi=300)
