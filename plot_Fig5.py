import numpy as np 
import matplotlib.pyplot as plt
import argparse
import pdb
import tensorflow as tf
import matplotlib
import matplotlib.cm as cm
import scipy.stats as stats
from functions.utils.common_utils import list_files, plot_fig, format_p_latex
from functions.utils.math_utils import find_connected_components, get_induced_subgraphs
import bct
import torch
import os
import seaborn as sns
import pandas as pd


matplotlib.rcParams['pdf.fonttype'] = 42

def p_value_to_stars(p_value):
    if p_value <= 0.0001:
        return '****'
    if p_value <= 0.001:
        return '***'
    if p_value <= 0.01:
        return '**'
    if p_value <= 0.05:
        return '*'
    return 'ns'

def start_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--incremental_path', default='./runs/Fig5bc/incremental', type=str)
    parser.add_argument('--normal_path', default='./runs/Fig5bc/all-at-once', type=str)
    parser.add_argument('--random_mask_path', default='./runs/Fig5efg/lottery_ticket_hypo_random', type=str)
    parser.add_argument('--prior_mask_path', default='./runs/Fig5efg/lottery_ticket_hypo_prior_modular', type=str)
    parser.add_argument('--posteriori_mask_path', default='./runs/Fig5efg/lottery_ticket_hypo_posteriori_modular', type=str)
    # parser.add_argument('--Fig5c_plot_style', default='seed', choices=['seed', 'dense'])
    args = parser.parse_args()
    return args

def plot_figure5b(args):
    model_size = 84

    seed_list = [ i for i in range(100, 2100, 100)]

    task_list = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti', 
                 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo',
                'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm']

    task_perf_name_list = [f'perf_{task}' for task in task_list]
    
    
    fig, axs = plt.subplots(figsize=(2.0, 2.0))
    directory_name_list = [args.incremental_path, args.normal_path]
    modularity_by_condition = {}
    
    for directory_name in directory_name_list:
        seed_paths_list = []
        for s_idx, seed_name in enumerate(seed_list):
            file_name = f"n_rnn_{model_size}_seed_{seed_name}"
            paths = list_files(directory_name, file_name)
            seed_paths_list.append(paths)

        modularity_seed_array = []
        task_perf_seed_array_dict = {task:[] for task in task_list}
        
        for ii, events_file in enumerate(seed_paths_list):            
            modularity_list = [0]
            task_perf_list = {task:[0] for task in task_list}
            
            for e in tf.compat.v1.train.summary_iterator(events_file):
                for v in e.summary.value:
                    if v.tag == 'SC_Qvalue':
                        modularity_list.append(v.simple_value)
                    if v.tag in task_perf_name_list:
                        task = v.tag.split('_')[1]
                        task_perf_list[task].append(v.simple_value)
            
            modularity_seed_array.append(np.array(modularity_list))
            for task in task_list:
                task_perf_seed_array_dict[task].append(task_perf_list[task])
        
        modularity_seed_array = np.array(modularity_seed_array)
    
        # Shape: (20 independent model instances, n_checkpoints)
        if 'incremental' in directory_name:
            modularity_by_condition['incremental'] = modularity_seed_array
        else:
            modularity_by_condition['all_at_once'] = modularity_seed_array
    
        for task in task_list:
            task_perf_seed_array_dict[task] = np.array(task_perf_seed_array_dict[task])
        
        task_perf_mean_dict = {}
        task_perf_ste_dict = {}
        for task in task_list:
            task_perf_mean_dict[task] = np.mean(task_perf_seed_array_dict[task], axis=0)
            performance_std = np.std(task_perf_seed_array_dict[task], axis=0)
            task_perf_ste_dict[task] = performance_std / np.sqrt(task_perf_seed_array_dict[task].shape[0])
        
        
        modularity_mean = np.mean(modularity_seed_array, axis=0)
        modularity_std = np.std(modularity_seed_array, axis=0)
        modularity_ste = modularity_std / np.sqrt(modularity_seed_array.shape[0])
        
        print(f'n_rnn:{model_size}, avg_moduarlity:{modularity_mean.mean():.4f}')

        # Generate label positions to display
        x_ticks = [i for i in range(20, modularity_seed_array.shape[1]+1, 20)]
        # x_ticks = [0] + x_ticks
        x_tick_labels = [500 * i for i in x_ticks]
                
        axs.set_xticks(x_ticks)
        axs.set_xticklabels(x_tick_labels, rotation=45, fontsize=6)
        axs.tick_params(axis='both', labelsize=5)
        axs.tick_params(axis='both', width=0.25)
        # axs.set_xlim(0, 90)
        # axs.axvline(x=80, color='green', linestyle='--', linewidth=0.75)
        
        color_list = ['#2171A8', '#DA762A']
        axs.set_ylim(0, 0.5)
        if 'incremental' in directory_name:
            color = color_list[1]
            axs.plot(modularity_mean, label='incremental', linewidth=0.5, color=color)
        else:
            color = color_list[0]
            axs.plot(modularity_mean, label='all-at-once', linewidth=0.5, color=color)
        
        axs.fill_between(range(modularity_seed_array.shape[1]), modularity_mean - modularity_ste, \
            modularity_mean + modularity_ste, alpha=0.2, color=color)
        
        

    # Statistical comparison: incremental vs all-at-once
    # ============================================================

    incremental_array = modularity_by_condition['incremental']
    all_at_once_array = modularity_by_condition['all_at_once']

    print("incremental:", incremental_array.shape)
    print("all-at-once:", all_at_once_array.shape)

    assert incremental_array.shape == all_at_once_array.shape
    assert incremental_array.shape[0] == 20
    assert all_at_once_array.shape[0] == 20

    # Two-sided Welch's two-sample t-test at each checkpoint
    t_stat, p_value = stats.ttest_ind(
        incremental_array,
        all_at_once_array,
        axis=0,
        equal_var=False,
        alternative='two-sided'
    )    
        

    # The array starts at iteration 0 and is recorded every 500 iterations
    iterations = np.arange(incremental_array.shape[1]) * 500

    # Checkpoints currently reported in Supplementary Table 4:
    selected_iterations = np.arange(15000, 45000, 5000)

    first_line = "Iterations"
    second_line = "P value"

    for iteration in selected_iterations:
        idx = iteration // 500

        first_line += f" & {iteration}"
        second_line += f" & {format_p_latex(p_value[idx])}"

    print(first_line)
    print(second_line)
        
    axs.spines['left'].set_position('zero')
    axs.spines['bottom'].set_position('zero') 
    axs.spines['top'].set_linewidth(0.25)    
    axs.spines['bottom'].set_linewidth(0.25) 
    axs.spines['left'].set_linewidth(0.25)  
    axs.spines['right'].set_linewidth(0.25)  
    
    axs.set_xlabel('Iterations', fontsize=6, labelpad=2)

    plt.legend(loc='lower right', bbox_to_anchor=(1.00, 0.02), frameon=False, ncol=1, fontsize=5, title_fontsize=6)
    axs.set_ylabel('Modularity', fontsize=6, labelpad=2)
    
    plt.tight_layout()
    plt.savefig(f"./figures/Fig5/Fig5b.svg", format='svg', dpi=300)
    plt.savefig(f"./figures/Fig5/Fig5b.jpg", format='jpg', dpi=300)

def plot_figure5c(args):
    n_rnn = 84
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    subgraph_source_path = './figures/Fig5/Fig5c_subgraph_level_repeated_measurements.csv'
    seed_source_path = './figures/Fig5/Fig5c_seed_level_source_data.csv'
    expected_seed_list = list(range(100, 2100, 100))

    if os.path.exists(subgraph_source_path):
        source_df = pd.read_csv(subgraph_source_path)
        print(f'Loaded cached Fig5c induced-subgraph data from {subgraph_source_path}')
    else:
        source_data_rows = []
        for seed in expected_seed_list:
            incremental_seed_values = []
            all_at_once_seed_values = []

            for step in range(3000, 40000, 500):
                incremental_model = torch.load(
                    os.path.join(args.incremental_path, f'n_rnn_{n_rnn}_seed_{seed}', f'RNN_interleaved_learning_{step}.pth'),
                    map_location=device)
                components = find_connected_components(incremental_model.mask)
                incremental_weights = np.abs(incremental_model.recurrent_conn.weight.data.detach().cpu().numpy())
                incremental_subgraphs = get_induced_subgraphs(incremental_weights, components)

                if len(incremental_subgraphs) < 2:
                    if incremental_seed_values:
                        break
                    continue

                all_at_once_model = torch.load(
                    os.path.join(args.normal_path, f'n_rnn_{n_rnn}_seed_{seed}', f'RNN_interleaved_learning_{step}.pth'),
                    map_location=device)
                all_at_once_weights = np.abs(all_at_once_model.recurrent_conn.weight.data.detach().cpu().numpy())
                all_at_once_subgraphs = get_induced_subgraphs(all_at_once_weights, components)

                for subgraph_idx, (incremental_subgraph, all_at_once_subgraph) in enumerate(zip(incremental_subgraphs, all_at_once_subgraphs)):
                    _, incremental_qvalue = bct.modularity_dir(incremental_subgraph)
                    _, all_at_once_qvalue = bct.modularity_dir(all_at_once_subgraph)
                    incremental_seed_values.append(incremental_qvalue)
                    all_at_once_seed_values.append(all_at_once_qvalue)
                    source_data_rows.append({
                        'Seed': seed,
                        'Step': step,
                        'Subgraph index': subgraph_idx,
                        'Incremental modularity': incremental_qvalue,
                        'All-at-once modularity': all_at_once_qvalue,
                    })

            if not incremental_seed_values or not all_at_once_seed_values:
                print(f'Skipping seed {seed}: no Fig5c induced-subgraph modularity values found.')

        source_df = pd.DataFrame(source_data_rows)
        source_df.to_csv(subgraph_source_path, index=False)

    seed_summary = (
        source_df.groupby('Seed')
        .agg({
            'Incremental modularity': 'mean',
            'All-at-once modularity': 'mean',
            'Subgraph index': 'count',
        })
        .rename(columns={'Subgraph index': 'Subgraph observations'})
        .reset_index()
    )
    seed_level_rows = []
    for _, row in seed_summary.iterrows():
        seed_level_rows.append({
            'Seed': int(row['Seed']),
            'Training paradigm': 'incremental',
            'Modularity': row['Incremental modularity'],
            'Subgraph observations': int(row['Subgraph observations']),
        })
        seed_level_rows.append({
            'Seed': int(row['Seed']),
            'Training paradigm': 'all-at-once',
            'Modularity': row['All-at-once modularity'],
            'Subgraph observations': int(row['Subgraph observations']),
        })

    stats_df = pd.DataFrame(seed_level_rows)

    if stats_df['Seed'].nunique() < 3:
        raise RuntimeError('Fig5c requires at least 3 paired seeds for statistical testing.')

    incremental_seed_means = stats_df[stats_df['Training paradigm'] == 'incremental'].sort_values('Seed')['Modularity'].to_numpy()
    all_at_once_seed_means = stats_df[stats_df['Training paradigm'] == 'all-at-once'].sort_values('Seed')['Modularity'].to_numpy()
    paired_seed_n = len(incremental_seed_means)
    wilcoxon_stat, p_value = stats.wilcoxon(incremental_seed_means, all_at_once_seed_means, alternative='two-sided')
    stars = p_value_to_stars(p_value)

    print(f'Fig5c Wilcoxon signed-rank test, two-sided, n={paired_seed_n} paired seeds: W={wilcoxon_stat:.6g}, p={p_value:.6g}, stars={stars}')

    summary_source_df = stats_df.pivot(index='Seed', columns='Training paradigm', values='Modularity').reset_index()
    summary_source_df['Incremental minus all-at-once'] = summary_source_df['incremental'] - summary_source_df['all-at-once']
    summary_source_df['Wilcoxon statistic'] = wilcoxon_stat
    summary_source_df['P value'] = p_value
    summary_source_df['Test'] = 'Two-sided paired Wilcoxon signed-rank test'
    summary_source_df['Multiple-comparison correction'] = 'None; one comparison'
    summary_source_df.to_csv(seed_source_path, index=False)

    palette = {'incremental': '#DA762A', 'all-at-once': '#2171A8'}
    names_order = ['incremental', 'all-at-once']

    group = 'Training paradigm'
    column = 'Modularity'
    # if args.Fig5c_plot_style == 'dense':
    #     plot_df = pd.concat([
    #         source_df[['Seed', 'Step', 'Subgraph index', 'Incremental modularity']]
    #             .rename(columns={'Incremental modularity': 'Modularity'})
    #             .assign(**{'Training paradigm': 'incremental'}),
    #         source_df[['Seed', 'Step', 'Subgraph index', 'All-at-once modularity']]
    #             .rename(columns={'All-at-once modularity': 'Modularity'})
    #             .assign(**{'Training paradigm': 'all-at-once'}),
    #     ], ignore_index=True)
    #     point_size = 0.8
    # else:
    plot_df = stats_df
    point_size = 1.5

    fig, ax = plt.subplots(figsize=(1.5, 1.8))
    ax = sns.boxplot(x=group, y=column, data=plot_df, order=names_order, ax=ax, palette=palette,
                boxprops=dict(facecolor='none', linewidth=0.25), width=0.12, 
                flierprops={
                                'markersize': 1,      # Size of outliers
                                'markeredgewidth': 0.25,  # Edge line width of outliers
                                }, 
                whiskerprops={'linewidth': 0.25}, medianprops={'linewidth': 0.25}, capprops={'linewidth': 0.25})
    np.random.seed(2026)
    ax = sns.stripplot(x=group, y=column, data=plot_df, order=names_order,
                dodge=False, ax=ax, palette=palette, jitter=0.05, size=point_size, alpha=0.7)

    y_min = plot_df[column].min()
    y_max = plot_df[column].max()
    y_range = y_max - y_min if y_max > y_min else 1.0
    line_y = y_max + 0.08 * y_range
    line_h = 0.03 * y_range
    ax.plot([0, 0, 1, 1], [line_y, line_y + line_h, line_y + line_h, line_y],
            linewidth=0.5, color='0.2')
    ax.text(0.5, line_y + line_h, stars, ha='center', va='bottom',
            fontsize=5, color='0.2')
    ax.set_ylim(top=line_y + 0.18 * y_range)

    ax.tick_params(axis='both', labelsize=5)
    ax.tick_params(axis='both', width=0.25)
    ax.spines['top'].set_linewidth(0.25)    
    ax.spines['bottom'].set_linewidth(0.25) 
    ax.spines['left'].set_linewidth(0.25)  
    ax.spines['right'].set_linewidth(0.25)  

    ax.set_title('Induced subgraphs comparison', fontsize=6)
    ax.set_xlabel('Training paradigm', fontsize=6, labelpad=2)
    ax.set_ylabel('Modularity', fontsize=6, labelpad=2)

    # plt.tight_layout()
    fig.subplots_adjust(
        left=0.21,
        right=0.98,
        bottom=0.18,
        top=0.90
    )
    output_prefix = f'./figures/Fig5/Fig5c'
    plt.savefig(f'{output_prefix}.jpg', format='jpg', dpi=300)
    plt.savefig(f'{output_prefix}.svg', format='svg', dpi=300)


def plot_figure5e(args):
    model_size_list = [8, 16, 32]

    for model_size in model_size_list:

        fig = plt.figure(figsize=(2.0, 2.0))
        task_num_list = [20]

        directory_name = args.random_mask_path
        seed_list = [ i for i in range(100, 2100, 100)]

        color_map = cm.get_cmap('Blues')
        color_indices = np.linspace(0.4, 0.9, len(model_size_list))  
        color_dict = {model_size: color_map(ci) for model_size, ci in zip(sorted(model_size_list), color_indices)}


        random_modularity_array, random_perf_array = \
            plot_fig(directory_name, seed_list, task_num_list, [model_size], ylabel='Performance', \
                plot_perf=True, linelabel=f'# random', color_dict=color_dict, y_lim_perf=0.8)
        

        color_map = cm.get_cmap('Reds')
        color_indices = np.linspace(0.4, 0.9, len(model_size_list)) 
        color_dict = {model_size: color_map(ci) for model_size, ci in zip(sorted(model_size_list), color_indices)}
        
        directory_name = args.posteriori_mask_path
        postriori_modularity_array, postriori_perf_array = \
            plot_fig(directory_name, seed_list, task_num_list, [model_size], ylabel='Avg performance', \
            plot_perf=True, linelabel=f'# posteriori_modular', color_dict=color_dict, y_lim_perf=0.8)


        color_map = cm.get_cmap('Greens')
        color_indices = np.linspace(0.4, 0.9, len(model_size_list)) 
        color_dict = {model_size: color_map(ci) for model_size, ci in zip(sorted(model_size_list), color_indices)}
        
        directory_name = args.prior_mask_path
        prior_modularity_array, prior_perf_array = \
            plot_fig(directory_name, seed_list, task_num_list, [model_size], ylabel='Avg performance', \
        plot_perf=True, linelabel=f'# prior_modular', color_dict=color_dict, y_lim_perf=0.8)

        # iterations = np.arange(0, 40500, 500) 

        # t_stat, p_value = stats.ttest_ind(
        #     postriori_modularity_array,
        #     random_modularity_array,
        #     axis=0,
        #     equal_var=False,
        #     alternative='two-sided'
        # )    

        # first_line = "Iterations"
        # second_line = "P value"

        # for i in range(0, len(iterations), 6):
        #     if iterations[i] > 10000 and iterations[i] <= 42000:
        #         first_line += f" & {iterations[i]}"
        #         second_line += f" & {p_value[i]:.4f}"
    
        # print(first_line)
        # print(second_line)

        plt.title(f'# Hidden Neurons: {model_size}', fontsize=6)
        plt.tight_layout()
        
        fig.savefig(f'./figures/Fig5/Fig5e_{model_size}.jpg', format='jpg', dpi=300)
        fig.savefig(f'./figures/Fig5/Fig5e_{model_size}.svg', format='svg', dpi=300)


if __name__ == '__main__':
    figures_path = './figures/Fig5'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    args = start_parse()
    plot_figure5b(args)
    plot_figure5c(args)
    plot_figure5e(args)
