import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import pdb
import tensorflow as tf
import seaborn as sns
import matplotlib
import os

matplotlib.rcParams['pdf.fonttype'] = 42


def list_files(directory):
    path_list = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()  # Sort directory list in-place
        files.sort()  # Sort file list in-place
        files.reverse()
        for file in files:
            if '.pth' in file or '.txt' in file:
                continue
            path_list.append(os.path.join(root, file))
            break

    return path_list


def plot_scatter(x_array, y_array, xlabel='Modularity', ylabel='Performance', file_name='performance_vs_modularity'):
    plt.figure(figsize=(1.8, 1.8))

    r, p = stats.pearsonr(x_array, y_array)
    print(f'step:{step}, r:{r:.4f}, p:{p:.6f}, len:{len(y_array)}')

    # Set colors and styles
    sns.regplot(
        x=x_array, 
        y=y_array, 
        scatter_kws={'s':1, 'color':'#2ca02c'},  # Soft light green
        line_kws={'color':'#1f77ff', 'linewidth':0.5, 'alpha':1},  # Sky blue, remove shadow
        ci=None  # Remove confidence interval shadow of regression line
    )

    # Add title and labels
    # plt.title('Modularity vs Performance: Correlation Analysis')
    plt.xlabel(f'{xlabel}', fontsize=6)
    plt.ylabel(f'{ylabel}', fontsize=6)

    axs = plt.gca()

    axs.spines['top'].set_linewidth(0.25)    
    axs.spines['bottom'].set_linewidth(0.25) 
    axs.spines['left'].set_linewidth(0.25)  
    axs.spines['right'].set_linewidth(0.25)  
    axs.tick_params(axis='both', width=0.25, length=1.0, labelsize=5, pad=1)

    # Add Pearson r and p value information
    plt.text(
        0.70, 0.95,  # x, y position (relative coordinates)
        f'r = {r:.4f}\np = {p:.2e}',  # Text to display
        transform=plt.gca().transAxes,  # Use relative coordinates
        verticalalignment='top',  # Text top alignment
        fontsize=5,  # Specify font size here
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', linewidth=0.25)  # Set edge border
    )

    axs.set_box_aspect(1)  # Set aspect ratio of plotting box to 1
    plt.tight_layout()

    plt.savefig(f"./figures/Fig3/Fig3d/{file_name}.svg", format='svg', dpi=300)
    plt.savefig(f"./figures/Fig3/Fig3d/{file_name}.jpg", format='jpg', dpi=300)
    plt.close()


figure_path = './figures/Fig3/Fig3d'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

task_num = 20

model_size_list = [32, 16, 8]
for m_idx, n_rnn in enumerate(model_size_list):
    runs_name = f'Fig3d_{n_rnn}'
    paths = list_files(f"./runs/{runs_name}")
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

    max_modularity = np.mean(sorted_mod[:, -1:], axis=1)
    learning_speed = []

    cumulative_acc = np.cumsum(perf_avg_array, axis=1)
    lens = np.arange(1, loss_array.shape[1] + 1)    
    average_acc_up_to_now = cumulative_acc / lens

    for j in range(perf_avg_array.shape[0]):
        visited = set()
        if loss_array[j].max() < 0.05:
            pdb.set_trace()
            print(f"j:  {j}")
            continue
        
        tmp = []

        for idx, step in enumerate(step_range):
            for milestone in [0.30]:
                if average_acc_up_to_now[j, idx] >= milestone  and milestone not in visited:
                    learning_speed.append(step)
                    visited.add(milestone)


    learning_speed = np.array(learning_speed)

    r, p = stats.pearsonr(max_modularity, learning_speed)
    print(f"r:{r:.4f}, p:{p:.4f}, len:{max_modularity.shape[0]}")
    plot_scatter(max_modularity, learning_speed, xlabel='Max Modularity', \
                 ylabel='Iterations to 30% Acc', file_name=f'learning_speed_vs_modularity_{n_rnn}')

    final_perf = perf_avg_array[:, -1]
    r, p = stats.pearsonr(max_modularity, final_perf)
    print(f"Final Perf -> r:{r:.4f}, p:{p:.4f}, len:{max_modularity.shape[0]}")
    plot_scatter(max_modularity, final_perf, xlabel='Max Modularity', \
                 ylabel='Final Performance', file_name=f'final_performance_vs_modularity_{n_rnn}')

    pre_acc_sum = 0.0
    max_r = -1.0
    max_r_arrays=None

    cnt = 0
    for idx, step in enumerate(step_range):
        if step % (500) !=0:
            cnt +=1
            continue

        acc_delta = (cumulative_acc[:, idx] - pre_acc_sum) / cnt
        pre_acc_sum = cumulative_acc[:, idx]
        r, p = stats.pearsonr(modularity_array[:, idx], acc_delta)
        cnt = 0

        perf = perf_avg_array[:, idx].mean()
        mod = modularity_array[:, idx].mean()

        if p <= 0.05:
            print(f'step:{step}, r:{r:.4f}, p:{p}, len:{modularity_array.shape[0]} mod:{mod:.4f} perf:{perf:.4f} -----------------')
        else:
            print(f'step:{step}, r:{r:.4f}, p:{p}, len:{modularity_array.shape[0]} mod:{mod:.4f} perf:{perf:.4f}')

        if r > max_r:
            max_r = r
            max_r_arrays = (modularity_array[:, idx], acc_delta)
    
        r_list.append(r)
        p_list.append(p)
        avg_mod_list.append(np.mean(modularity_array[:, idx]))


    plot_scatter(max_r_arrays[0], max_r_arrays[1])
    

    sort_p_list = sorted(p_list)

    m = len(p_list)

    # Calculate FDR threshold
    thresholds = np.arange(1, m+1) / m * 0.05

    sort_p_list = np.array(sort_p_list)
    # Find the largest p-value that satisfies p(i) <= threshold
    significant = sort_p_list <= thresholds

    if np.any(significant):
        fdr_threshold = sort_p_list[significant][-1]
    else:
        fdr_threshold = 0.05
    

    print(fdr_threshold)
    # Generate label positions to display
    x_ticks = [ i for i in range(39, len(r_list)+1, 40)]

    x_tick_labels = [500*(i+1) for i in x_ticks]
    # Define number of rows
    rows = 2

    fig_width = 2.8
    fig_height = 3.5 # Reserve some extra space for title, label and spacing
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(fig_width, fig_height))

    # --- subgraph 1: correlation (Correlation) ---
    axs[0].bar(range(len(r_list)), r_list, color='lightsteelblue')
    axs[0].set_title(f'# Hidden Neurons: {n_rnn}', fontsize=7)
    axs[0].set_ylabel('Correlation', fontsize=6, labelpad=1)

    # --- subgraph 2: P-value ---
    logp_value_threshold = -np.log10(fdr_threshold)
    p_list_log = [-np.log10(p) for p in p_list]

    axs[1].bar(range(len(p_list_log)), p_list_log, color='lightgray')
    axs[1].axhline(y=logp_value_threshold, color='green', linestyle='--', linewidth=0.5)
    axs[1].set_ylim([0, 5])
    axs[1].set_ylabel('-log(p)', fontsize=6, labelpad=1)


    for ax in axs:
        # --- This is the key to ensure square shape ---
        ax.set_box_aspect(1)  # Set aspect ratio of plotting box to 1
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        
        # Set coordinate axis style
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
            
        ax.tick_params(axis='both', width=0.25, length=1.0, labelsize=5, pad=1)
        ax.set_xlabel('Iterations', fontsize=6, labelpad=1)

    # Remove x-axis label from top graph
    axs[0].set_xlabel('')

    # Adjust spacing between subgraphs
    plt.subplots_adjust(hspace=0.6) # Increase vertical spacing to prevent title and top graph overlap

    plt.savefig(f"{figure_path}/correlation_bar_{n_rnn}_{runs_name}.svg", format='svg', dpi=300)
    plt.savefig(f"{figure_path}/correlation_bar_{n_rnn}_{runs_name}.jpg", format='jpg', dpi=300)
    plt.close()