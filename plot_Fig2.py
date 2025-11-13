import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pdb
import bct
import torch
import matplotlib.cm as cm
from functions.utils.plot_utils import plot_fig, get_seed_avg
import scipy.stats as stats
import pandas as pd
import pickle
import seaborn as sns
import os

matplotlib.rcParams['pdf.fonttype'] = 42

    
def plot_fig2a(model_size_list, color_dict):
    fig = plt.figure(figsize=(2.0, 2.0))
    directory_name = "./runs/Fig2a_data_RNN_relu"
    seed_list = [ i for i in range(100, 1100, 100)]
    task_name_list = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
    
    for flag in [False]:
        plot_fig(directory_name, seed_list, task_name_list, model_size_list, \
            ylabel='Avg performance', plot_perf=True, \
                color_dict=color_dict, chance_flag=flag)

    plt.title('Single Task Learning', fontsize=7)
    plt.tight_layout()

    fig.savefig(f'./figures/Fig2/Fig2a.svg', format='svg', dpi=300)
    fig.savefig(f'./figures/Fig2/Fig2a.jpg', format='jpg', dpi=300)

def plot_fig2b(model_size_list, color_dict):
    fig = plt.figure(figsize=(2.0, 2.0))
    directory_name = "./runs/Fig2bcde_data_relu"
    seed_list = [ i for i in range(100, 2100, 100)]
    task_num_list = [20]
    for flag in [False]:
        plot_fig(directory_name, seed_list, task_num_list, model_size_list, \
            ylabel='Avg performance', color_dict=color_dict, chance_flag=flag)

    plt.title('Multi-task Learning', fontsize=7)
    plt.tight_layout()
        
    fig.savefig(f'./figures/Fig2/Fig2b.svg', format='svg', dpi=300)
    fig.savefig(f'./figures/Fig2/Fig2b.jpg', format='jpg', dpi=300)

def plot_fig2c(color_dict, N=15):
    fig = plt.figure(figsize=(2.0, 2.0))
    model_size_list = [N]
    task_num_list = [20]
    directory_name = "./runs/Fig2bcde_data_relu"
    seed_list = [ i for i in range(100, 2100, 100)]
    multitask_modularity_array, multitask_perf_array = \
        plot_fig(directory_name, seed_list, task_num_list, model_size_list, ylabel='Modularity', \
             plot_perf=False, linelabel=f'# Multi-task', color_dict=color_dict)
    
    task_name_list = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

    directory_name = "./runs/Fig2a_data_RNN_relu"

    singletask_modularity_array, singletask_perf_array = \
        plot_fig(directory_name, seed_list, task_name_list, model_size_list, ylabel='Modularity', \
        plot_perf=False, linelabel=f'# Single task', color_dict=color_dict)

    iterations = np.arange(0, 47000, 500) 

    t_stat, p_value = stats.ttest_ind(multitask_modularity_array, singletask_modularity_array)

    first_line = "Iterations"
    second_line = "P value"

    for i in range(0, len(iterations), 6):
        if iterations[i] > 10000 and iterations[i] <= 42000:
            first_line += f" & {iterations[i]}"
            second_line += f" & {p_value[i]:.4f}"

    print(first_line)
    print(second_line)

    # plt.title(f'Single task vs Multi-task\n(# Hidden Neurons: {N})', fontsize=6)
    plt.title(f'Single task vs Multi-task', fontsize=6)
    plt.tight_layout()
        
    fig.savefig(f'./figures/Fig2/Fig2c_{N}.jpg', format='jpg', dpi=300)
    fig.savefig(f'./figures/Fig2/Fig2c_{N}.svg', format='svg', dpi=300)
    
def plot_fig2de(model_size_list, task_num_list, color_dict):
    directory_name = "./runs/Fig2bcde_data_relu"
    seed_list = [ i for i in range(100, 2100, 100)]

    for model_size in model_size_list:
        fig, axs = plt.subplots(figsize=(2.0, 2.0))
        for task_set_id, task_num in enumerate(task_num_list):

            modularity_array, _ = get_seed_avg(directory_name, model_size, task=task_num, seed_list=seed_list)
            modularity_mean = np.mean(modularity_array, axis=0)
            modularity_std = np.std(modularity_array, axis=0)
            modularity_ste = modularity_std / np.sqrt(modularity_array.shape[0])

            x_ticks = [ i for i in range(20, modularity_array.shape[1]+1, 20)]
            x_ticks = [0] + x_ticks
            x_tick_labels = [500*i for i in x_ticks]
            axs.set_xticks(x_ticks)
            axs.set_xticklabels(x_tick_labels, rotation=45, fontsize=5)
            
            color = color_dict[task_set_id]
            line_label = f'# Tasks: {task_num:2}'
            axs.plot(modularity_mean, label=line_label, color=color, linewidth=0.25)
            axs.fill_between(range(modularity_array.shape[1]), modularity_mean - modularity_ste, \
                modularity_mean + modularity_ste, color=color, alpha=0.2)
                
        axs.spines['left'].set_position('zero')
        axs.spines['bottom'].set_position('zero') 
        axs.spines['top'].set_linewidth(0.25)    
        axs.spines['bottom'].set_linewidth(0.25) 
        axs.spines['left'].set_linewidth(0.25)  
        axs.spines['right'].set_linewidth(0.25)  
        axs.tick_params(axis='both', labelsize=5)
        axs.tick_params(axis='both', width=0.25)

        axs.set_title(f'# Hidden Neurons: {model_size}', fontsize=6)  
        axs.set_xlabel('Iterations', fontsize=6)    
        axs.set_ylabel('Modularity', fontsize=6)    
        axs.legend(loc='lower right', bbox_to_anchor=(0.98, 0.05), frameon=False, fontsize=6)

        plt.tight_layout()
        fig.savefig(f'./figures/Fig2/Fig2de_{model_size}.jpg', format='jpg', dpi=300)
        fig.savefig(f'./figures/Fig2/Fig2de_{model_size}.svg', format='svg', dpi=300)


def plot_interaction_fig(model_size_list, task_num_list):
    """
    Plots the "Interaction Plot":
    - X-axis: Number of Tasks
    - Y-axis: Final Modularity
    - Lines: One line per model_size
    """
    directory_name = "./runs/Fig2bcde_data_relu"
    seed_list = [i for i in range(100, 2100, 100)]
    
    # Define a color dict for the model sizes
    # You can change these colors
    model_color_dict = {
        64: '#003f5c',  # Dark Blue
        32: '#4472c4', # Blue
        16: '#5cb85c', # Green
        8: '#ffa600'  # Orange/Yellow
    }

    fig, axs = plt.subplots(figsize=(2.5, 2.2)) # Single plot

    # Outer loop is Model Size (each size is a line)
    for model_size in model_size_list:
        
        mean_modularities = []
        ste_modularities = []

        # Inner loop is Task Number (each task num is a point on the X-axis)
        for task_num in task_num_list:
            # Load the data for this specific condition
            modularity_array, _ = get_seed_avg(directory_name, model_size, task=task_num, seed_list=seed_list)
            
            # Get the final modularity value for EACH seed
            # Shape: (num_seeds,)
            # mod_values_per_seed = modularity_array[:, -1]
            mod_values_per_seed = modularity_array.mean(1)
            # mod_values_per_seed = modularity_array.max(1)
            
            # Calculate the mean and STE of these final values
            mean_final_mod = np.mean(mod_values_per_seed)
            ste_final_mod = np.std(mod_values_per_seed) / np.sqrt(mod_values_per_seed.shape[0])
            
            mean_modularities.append(mean_final_mod)
            ste_modularities.append(ste_final_mod)

        # After iterating through all tasks, plot the line for this model_size
        color = model_color_dict.get(model_size, 'black') # Get color or default to black
        mean_modularities = np.array(mean_modularities)
        ste_modularities = np.array(ste_modularities)
        
        # Use a label based on your paper's parameter scaling, as the reviewer suggested
        recurrent_params = model_size * model_size
        line_label = f'{model_size} Neurons ({recurrent_params} Params)'
        
        axs.plot(task_num_list, mean_modularities, label=line_label, color=color, linewidth=1.0, marker='o', markersize=3)
        axs.fill_between(task_num_list, 
                         mean_modularities - ste_modularities,
                         mean_modularities + ste_modularities, 
                         color=color, alpha=0.2)

    # --- Styling the new plot ---
    axs.set_xlabel('Number of Tasks', fontsize=6)
    # axs.set_ylabel('Final Modularity', fontsize=6)
    axs.set_ylabel('Mean Modularity', fontsize=6)
    axs.set_title('Modularity vs. Task Load and Network Size', fontsize=7)
    
    # Set X-axis ticks to match your task numbers
    axs.set_xticks(task_num_list, fontsize=6)
    axs.set_xticklabels(task_num_list, fontsize=6)

    # Apply your styling
    axs.spines['top'].set_linewidth(0.25)
    axs.spines['bottom'].set_linewidth(0.25)
    axs.spines['left'].set_linewidth(0.25)
    axs.spines['right'].set_linewidth(0.25)
    axs.tick_params(axis='both', labelsize=5, width=0.25)

    axs.legend(loc='upper left', frameon=False, fontsize=5)
    plt.tight_layout()

    fig.savefig(f'./figures/Fig2/Fig2_InteractionPlot.jpg', format='jpg', dpi=300)
    fig.savefig(f'./figures/Fig2/Fig2_InteractionPlot.svg', format='svg', dpi=300)
    print(f"Saved new interaction plot to ./figures/Fig2/Fig2_InteractionPlot.svg")


# --- Main Plotting Function ---
def plot_representational_strain_fig(model_size_list, task_num_list):
    from functions.utils.eval_utils import ActivationHook
    from functions.utils.math_utils import calculate_effective_dimensionality
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    directory_name = "./runs/Fig2bcde_data_relu"
    seed_list = [i for i in range(100, 2100, 100)] # Assuming 20 seeds
    

    task_list_all = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
    
    # Assuming the 'task' module is importable
    import datasets.multitask as task
    
    results_filename = f'./runs/representational_strain.pkl'
    if os.path.exists(results_filename):
        print(f"Found serialized results file '{results_filename}'. Loading data...")
            
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)
    else:
        results = [] # Store all data for plotting
    
    if not results :
        num_batches_per_task = 5 # How many batches to sample for ED
        batch_size = 64
        
        for model_size in model_size_list:
            for task_num in task_num_list:
                
                # --- Get Modularity Data ---
                # Load modularity for all seeds in this condition at once
                mod_array, _ = get_seed_avg(directory_name, model_size, task=task_num, seed_list=seed_list)
                # Get the *final* modularity for each seed
                final_mod_per_seed = mod_array[:, -1] # Shape: (len(seed_list),)

                
                for seed_idx, seed in enumerate(seed_list):
                    print(f"Processing: Size={model_size}, Tasks={task_num}, Seed={seed}")
                    
                    # --- Load Model ---
                    model_path = f'{directory_name}/n_rnn_{model_size}_task_{task_num}_seed_{seed}/RNN_interleaved_learning_{46500}.pth'
                    if not os.path.exists(model_path):
                        print(f"Warning: Model not found, skipping: {model_path}")
                        continue
                    
                    model = torch.load(model_path, map_location=device) 
                    model.eval() # Set to evaluation mode
                    hp = model.hp
                    
                    # THIS IS THE CRITICAL LINE YOU WROTE
                    task_list_part = task_list_all[:task_num]
            
                    # --- Setup Hook ---
                    hook_container = ActivationHook()
                    # Register the hook on the readout layer
                    handle = model.readout.register_forward_hook(hook_container)        
                    
                    # --- Run Trials ---
                    with torch.no_grad(): 
                        for task_name_now in task_list_part:                
                            for _ in range(num_batches_per_task):
                                trial = task.generate_trials(task_name_now, hp, 'random', batch_size=batch_size)
                                input_tensor = torch.from_numpy(trial.x).to(device)

                                # Running the forward pass now automatically saves the hidden states via the hook
                                _ = model(input_tensor) 
                                
                    # --- Remove Hook ---
                    handle.remove()

                    # --- Process Collected Activations ---
                    if not hook_container.activations:
                        # print(f"Warning: No activations collected for {model_path}")
                        continue
                        
                    # Concatenate all collected (T, B, H) tensors and reshape to (Total_Samples, H)
                    all_activations_stacked = torch.cat(hook_container.activations, dim=0)
                    all_activations_for_this_seed = all_activations_stacked.reshape(-1, model_size)
                    
                    # --- Calculate ED and Strain ---
                    ed_capacity = calculate_effective_dimensionality(all_activations_for_this_seed)
                    task_load = task_num 
                    strain = task_load / ed_capacity.item()
                    
                    # --- Store Result ---
                    results.append({
                        'model_size': model_size,
                        'task_num': task_num,
                        'seed': seed,
                        'strain': strain,
                        'modularity': final_mod_per_seed[seed_idx],
                        'ed_capacity': ed_capacity.item()
                    })
        
        with open(results_filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Computation finished and results saved to '{results_filename}'.")
        
    df = pd.DataFrame(results).dropna(subset=['strain', 'modularity'])
    
    model_color_dict = { 64: '#003f5c', 32: '#4472c4', 16: '#5cb85c', 8: '#ffa600' }
    
    fig, ax = plt.subplots(figsize=(2.5, 2.2))
    
    sns.scatterplot(
        data=df, x='strain', y='modularity', hue='model_size',
        palette=model_color_dict, ax=ax, alpha=0.7, s=20
    )
    
    sns.regplot(
        data=df, x='strain', y='modularity',
        scatter=False, ax=ax, color='black',
        line_kws={'linestyle':'--', 'linewidth': 1.0}
    )
    
    # --- NEWLY ADDED STATISTICAL ANNOTATION ---
    # Calculate r and p
    r, p = stats.pearsonr(df['strain'], df['modularity'])
    
    # Format the text using LaTeX for academic style
    stat_text = (
        f"$r = {r:.3f}$\n"
        f"$p = {p:.2e}$"
    )
    
    # Add text to the plot in the top-left corner
    ax.text(0.05, 0.95, stat_text, 
            transform=ax.transAxes, 
            fontsize=7, 
            verticalalignment='top',
            # Add a white box for readability
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5, edgecolor='none'))
    # --- END OF NEW CODE ---
            
    ax.set_title('Modularity Increases with Representational Strain', fontsize=7)
    ax.set_xlabel('Representational Strain\n(Task Num / Effective Dimensionality)', fontsize=6)
    ax.set_ylabel('Final Modularity', fontsize=6)
    
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [f'{int(l)} Neurons ({int(l)*int(l)} Params)' for l in labels if l.isdigit()]
    
    ax.legend(handles, new_labels, title='Model Size', fontsize=5, title_fontsize=6, frameon=False, loc='upper right')
    
    ax.spines['top'].set_linewidth(0.25)
    ax.spines['bottom'].set_linewidth(0.25)
    ax.spines['left'].set_linewidth(0.25)
    ax.spines['right'].set_linewidth(0.25)
    ax.tick_params(axis='both', labelsize=6, width=0.25)
    
    plt.tight_layout()
        
    fig.savefig(f'./figures/Fig2/Fig2_StrainPlot.jpg', format='jpg', dpi=300)
    fig.savefig(f'./figures/Fig2/Fig2_StrainPlot.svg', format='svg', dpi=300)
    print("Saved new 'Strain Plot' to ./figures/Fig2/Fig2_StrainPlot.svg")
    print(f"Correlation: r={r:.4f}, p={p:.2e}")
    return df



def plot_fig2f(model_size, seed, step, task_num_list):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    directory_name = "./runs/Fig2bcde_data_relu"

    for task_num in task_num_list:
        file_name = f'{directory_name}/n_rnn_{model_size}_task_{task_num}_seed_{seed}/RNN_interleaved_learning_{step}.pth'
        model = torch.load(file_name, device)   
        weights = model.recurrent_conn.weight.data.detach().cpu().numpy()
        cluster_id, sc_qvalue = bct.modularity_dir(np.abs(weights))
        weights = np.abs(weights)
        sorted_indices = np.argsort(cluster_id)
        sorted_matrix = weights[sorted_indices][:, sorted_indices]
        fig = plt.figure(figsize=(1.5, 1.5))

        # plt.imshow(sorted_matrix, cmap='Oranges_r', interpolation='nearest')  
        # plt.imshow(sorted_matrix, cmap='YlOrBr', interpolation='nearest')  
        im = plt.imshow(sorted_matrix, interpolation='nearest')
        
        ticks_range = np.arange(0, len(sorted_matrix), 2)
        plt.xticks(ticks_range, ticks_range, fontsize=5)
        plt.yticks(ticks_range, ticks_range, fontsize=5)
        
        plt.xlabel('Neurons', fontsize=5, labelpad=0)
        plt.ylabel('Neurons', fontsize=5, labelpad=0)

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


if __name__ == '__main__':

    figures_path = './figures/Fig2'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    model_size_list = [8, 16, 32, 64]
    task_num_list = [3, 6, 11, 16, 20]
    num_curves = len(model_size_list)

    color_map = cm.get_cmap('Blues')
    color_indices = np.linspace(0.4, 0.9, len(model_size_list))  
    color_dict = {model_size: color_map(ci) for model_size, ci in zip(sorted(model_size_list), color_indices)}

    plot_interaction_fig(model_size_list, task_num_list)
    plot_representational_strain_fig(model_size_list, task_num_list)

    plot_fig2a(model_size_list, color_dict)
    plot_fig2b(model_size_list, color_dict)
    for N in model_size_list:
        plot_fig2c(color_dict, N)

    num_curves = len(task_num_list)
    color_map = cm.get_cmap('winter')
    color_indices = np.linspace(0.00, 1.0, len(task_num_list))  
    color_indices = color_indices[::-1]
    color_dict = {idx: color_map(ci) for idx, ci in zip(range(num_curves), color_indices)}

    plot_fig2de(model_size_list, task_num_list, color_dict)

    plot_fig2f(model_size=8, seed=100, step=40000, task_num_list=task_num_list)
