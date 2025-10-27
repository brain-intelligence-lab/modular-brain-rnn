import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import pdb
import tensorflow as tf
import seaborn as sns
import matplotlib
from matplotlib import font_manager 
import os

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


def plot_scatter(x_array, y_array):
    plt.figure(figsize=(2.0, 2.0))

    r, p = stats.pearsonr(x_array, y_array)
    print(f'step:{step}, r:{r:.4f}, p:{p:.6f}, len:{len(y_array)}')

    # 设置颜色和样式
    sns.regplot(
        x=x_array, 
        y=y_array, 
        scatter_kws={'s':1, 'color':'#2ca02c'},  # 柔和的浅绿色
        line_kws={'color':'#1f77ff', 'linewidth':0.5, 'alpha':1},  # 天蓝色，去掉阴影
        ci=None  # 去掉回归线的置信区间阴影
    )

    # 添加标题和标签
    # plt.title('Modularity vs Performance: Correlation Analysis')
    plt.xlabel('Modularity', fontsize=6)
    plt.ylabel('-Loss', fontsize=6)

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

    return 1


task_num = 20

model_size_list = [16, 8]
for m_idx, n_rnn in enumerate(model_size_list):
    exp_name = f'Fig3c_seed_search_0.1_tanh_{n_rnn}'
    paths = list_files(f"./runs/{exp_name}")
    # paths = paths[:96]
    print(len(paths))
    
    r_list = []
    p_list = []
    avg_loss_list = []
    avg_mod_list = []
    step_range = range(20, 40010, 20)

    modularity_array = np.zeros((len(paths), len(step_range)))
    perf_avg_array = np.zeros((len(paths), len(step_range)))
    loss_array = np.zeros((len(paths), len(step_range)))

    all_task_mp = {}
    all_task_perf = []
    
    for idx, events_file in enumerate(paths):
        tag_mp = {}

        for e in tf.compat.v1.train.summary_iterator(events_file):
            if e.step in step_range:
                for v in e.summary.value:
                    if v.tag not in tag_mp:
                        tag_mp[v.tag] = {}
                    
                    tag_mp[v.tag][e.step] = v.simple_value

        for _, (tag, mp) in enumerate(tag_mp.items()):
            if tag == 'perf_min':
                continue

            if tag == 'SC_Qvalue':
                for _idx, (key, value) in enumerate(mp.items()):
                    modularity_array[idx, _idx] = value

            if tag == 'Loss':
                for _idx, (key, value) in enumerate(mp.items()):
                    loss_array[idx, _idx] = value
            
            elif tag[:4] == 'perf':
                if tag == 'perf_avg':
                    for _idx, (key, value) in enumerate(mp.items()):
                        perf_avg_array[idx, _idx] = value
                else:
                    
                    if tag not in all_task_mp:
                        all_task_mp[tag] = np.zeros((len(paths), len(step_range)))
                    

                    for _idx, (key, value) in enumerate(mp.items()):
                        all_task_mp[tag][idx, _idx] = value

    for tag, array in all_task_mp.items():
        all_task_perf.append(array)
    
    all_task_perf = np.array(all_task_perf)
    perf_var_perf = np.var(all_task_perf, axis=0)

    sorted_mod = np.sort(modularity_array, axis=1)

    max_modularity = np.mean(sorted_mod[:, -50:], axis=1)
    # final_perf = []
    learning_speed = []

    cumulative_loss = np.cumsum(loss_array, axis=1)
    lens = np.arange(1, loss_array.shape[1] + 1)    
    average_loss_up_to_now = cumulative_loss / lens  


    for j in range(perf_avg_array.shape[0]):
        visited = set()
        score_list = []
        if loss_array[j].max() < 0.05:
            pdb.set_trace()
            print(f"j:  {j}")
            continue
        
        # max_modularity.append(np.mean(sorted_mod[j, -100:])
        tmp = []

        for idx, step in enumerate(step_range):
            for milestone in [6.0]:
                if average_loss_up_to_now[j, idx] <= milestone  and milestone not in visited:
                    # tmp.append(step * average_loss_up_to_now[j, idx])
                    learning_speed.append(step)
                    visited.add(milestone)
                    # score_list.append(average_loss_up_to_now[j, idx])

        # learning_speed.append(np.sum(tmp))
        # learning_speed[-1] *= np.sum(score_list)

    # max_modularity = np.array(max_modularity)
    learning_speed = np.array(learning_speed)

    r, p = stats.pearsonr(max_modularity, learning_speed)
    print(f"r:{r:.4f}, p:{p:.4f}, len:{max_modularity.shape[0]}")
    pre_loss_sum = 0.0

    max_r = -1.0
    max_r_arrays=None

    for idx, step in enumerate(step_range):
        if step % (500) !=0:
            continue
    
        # r, p = stats.pearsonr(modularity_array[:, idx], perf_avg_array[:, idx])
        loss_delta = cumulative_loss[:, idx] - pre_loss_sum
        pre_loss_sum = cumulative_loss[:, idx]
        
        r, p = stats.pearsonr(modularity_array[:, idx], -loss_delta)
        # r, p = stats.pearsonr(modularity_array[:, idx], cumulative_loss[:, idx])
        # r, p = stats.pearsonr(modularity_array[:, idx], -loss_array[:, idx])

        perf = perf_avg_array[:, idx].mean()
        mod = modularity_array[:, idx].mean()

        if p <= 0.05:
            print(f'step:{step}, r:{r:.4f}, p:{p}, len:{modularity_array.shape[0]} mod:{mod:.4f} perf:{perf:.4f} -----------------')
        else:
            print(f'step:{step}, r:{r:.4f}, p:{p}, len:{modularity_array.shape[0]} mod:{mod:.4f} perf:{perf:.4f}')

        if r > max_r:
            max_r = r
            max_r_arrays = (modularity_array[:, idx], -loss_delta)
    
        r_list.append(r)
        p_list.append(p)
        avg_loss_list.append(np.mean(average_loss_up_to_now[:, idx]))
        avg_mod_list.append(np.mean(modularity_array[:, idx]))


    plot_scatter(max_r_arrays[0], max_r_arrays[1])
    

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
    

    print(fdr_threshold)
    # 生成要显示的标签位置
    x_ticks = [ i for i in range(39, len(r_list)+1, 40)]

    x_tick_labels = [500*(i+1) for i in x_ticks]
    # 定义行数
    rows = 2

    # -- 核心修改 --
    # figsize 现在可以更自由地设置，因为子图形状由 set_box_aspect 控制
    # 我们只需要保证有足够的垂直空间即可
    fig_width = 2.8
    fig_height = 3.5 # 给标题、标签和间距留出一些额外空间
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(fig_width, fig_height))

    # --- 子图 1: 相关性 (Correlation) ---
    axs[0].bar(range(len(r_list)), r_list, color='lightsteelblue')
    axs[0].set_title(f'# Hidden Neurons: {n_rnn}', fontsize=7)
    axs[0].set_ylabel('Correlation', fontsize=6, labelpad=1)

    # --- 子图 2: P-value ---
    logp_value_threshold = -np.log10(fdr_threshold)
    p_list_log = [-np.log10(p) for p in p_list]

    axs[1].bar(range(len(p_list_log)), p_list_log, color='lightgray')
    axs[1].axhline(y=logp_value_threshold, color='green', linestyle='--', linewidth=0.5)
    axs[1].set_ylim([0, 5])
    axs[1].set_ylabel('-log(p)', fontsize=6, labelpad=1)


    for ax in axs:
        # --- 这是保证正方形的关键 ---
        ax.set_box_aspect(1)  # 设置绘图框的高宽比为1
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        
        # 设置坐标轴样式
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
            
        ax.tick_params(axis='both', width=0.25, length=1.0, labelsize=5, pad=1)
        ax.set_xlabel('Iterations', fontsize=6, labelpad=1)

    # 移除顶部图的 x 轴标签
    axs[0].set_xlabel('')

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.6) # 增加垂直间距，防止标题和上图重叠

    # 注意：当手动设置 aspect 后，tight_layout() 的效果可能会受限
    # plt.tight_layout() # 可以尝试使用，但有时会与 set_box_aspect 冲突，subplots_adjust 更可靠

    # --- 保存图像 ---
    figure_path = './figures/Fig3/Fig3c'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    plt.savefig(f"{figure_path}/Correlation_bar_{n_rnn}_{exp_name}.svg", format='svg', dpi=300)
    plt.savefig(f"{figure_path}/Correlation_bar_{n_rnn}_{exp_name}.jpg", format='jpg', dpi=300)

    # plt.show()