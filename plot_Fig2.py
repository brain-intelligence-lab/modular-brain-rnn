import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pdb
import bct
import torch
import matplotlib.cm as cm
from functions.utils.plot_utils import plot_fig, get_seed_avg
import scipy.stats as stats
import statsmodels.formula.api as smf
import pandas as pd
import pickle
import seaborn as sns
import os

matplotlib.rcParams['pdf.fonttype'] = 42

    
def plot_fig2a(model_size_list, color_dict):
    fig = plt.figure(figsize=(2.0, 2.0))
    directory_name = "./runs/Fig2a"
    seed_list = [ i for i in range(100, 2100, 100)]
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
    directory_name = "./runs/Fig2b-h"
    seed_list = [ i for i in range(100, 2100, 100)]
    task_num_list = [20]
    for flag in [False]:
        plot_fig(directory_name, seed_list, task_num_list, model_size_list, \
            ylabel='Avg performance', color_dict=color_dict, chance_flag=flag)

    plt.title('Multi-task Learning', fontsize=7)
    plt.tight_layout()
        
    fig.savefig(f'./figures/Fig2/Fig2b.svg', format='svg', dpi=300)
    fig.savefig(f'./figures/Fig2/Fig2b.jpg', format='jpg', dpi=300)

def plot_fig2d(color_dict, N=15):
    fig = plt.figure(figsize=(2.0, 2.0))
    model_size_list = [N]
    task_num_list = [20]
    directory_name = "./runs/Fig2b-h"
    seed_list = [ i for i in range(100, 2100, 100)]
    multitask_modularity_array, multitask_perf_array = \
        plot_fig(directory_name, seed_list, task_num_list, model_size_list, ylabel='Modularity', \
             plot_perf=False, linelabel=f'# Multi-task', color_dict=color_dict)
    
    task_name_list = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

    directory_name = "./runs/Fig2a"

    singletask_modularity_array, singletask_perf_array = \
        plot_fig(directory_name, seed_list, task_name_list, model_size_list, ylabel='Modularity', \
        plot_perf=False, linelabel=f'# Single task', color_dict=color_dict)

    # iterations = np.arange(0, 47000, 500) 

    # t_stat, p_value = stats.ttest_ind(multitask_modularity_array, singletask_modularity_array)

    # first_line = "Iterations"
    # second_line = "P value"

    # for i in range(0, len(iterations), 6):
    #     if iterations[i] > 10000 and iterations[i] <= 42000:
    #         first_line += f" & {iterations[i]}"
    #         second_line += f" & {p_value[i]:.4f}"

    # print(first_line)
    # print(second_line)

    plt.title(f'Single task vs Multi-task\n(# Hidden Neurons: {N})', fontsize=6)
    # plt.title(f'Single task vs Multi-task', fontsize=6)
    plt.tight_layout()
        
    fig.savefig(f'./figures/Fig2/Fig2d_{N}.jpg', format='jpg', dpi=300)
    fig.savefig(f'./figures/Fig2/Fig2d_{N}.svg', format='svg', dpi=300)
    
def plot_fig2efg(model_size_list, task_num_list, color_dict):
    directory_name = "./runs/Fig2b-h"
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
            axs.plot(modularity_mean, label=line_label, color=color, linewidth=1.0)
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
        
        axs.set_title(f'# Hidden Neurons: {model_size}\n({model_size**2} Params)', fontsize=6)  
        axs.set_xlabel('Iterations', fontsize=6)    
        axs.set_ylabel('Modularity', fontsize=6)   
        # plt.ylim([0.0, 0.40]) 
        axs.legend(loc='lower right', bbox_to_anchor=(0.99, 0.01), frameon=False, fontsize=6)

        plt.tight_layout()
        fig.savefig(f'./figures/Fig2/Fig2efg_{model_size}.jpg', format='jpg', dpi=300)
        fig.savefig(f'./figures/Fig2/Fig2efg_{model_size}.svg', format='svg', dpi=300)


def plot_2h(model_size_list, task_num_list, load_step=46500):
    from functions.utils.eval_utils import ActivationHook
    from functions.utils.math_utils import calculate_effective_dimensionality
    import datasets.multitask as task

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    directory_name = "./runs/Fig2b-h"
    seed_list = [i for i in range(100, 2100, 100)] 
    
    task_list_all = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
                     'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
                     'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                     'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
    

    results_filename = f'./runs/Fig2b-h/TDR.pkl'
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
                try:
                    mod_array, _ = get_seed_avg(directory_name, model_size, task=task_num, seed_list=seed_list)
                    final_mod_per_seed = mod_array[:, -1] # Shape: (len(seed_list),)
                except Exception as e:
                    print(f"Error getting modularity data, skipping {model_size}/{task_num}: {e}")
                    continue # Skip this model_size/task_num combo
                
                for seed_idx, seed in enumerate(seed_list):
                    print(f"Processing: Size={model_size}, Tasks={task_num}, Seed={seed}")
                    
                    # --- Load Model ---
                    model_path = f'{directory_name}/n_rnn_{model_size}_task_{task_num}_seed_{seed}/RNN_interleaved_learning_{load_step}.pth'
                    assert os.path.exists(model_path), f"Warning: Model not found: {model_path}"
                    
                    model = torch.load(model_path, map_location=device) 
                    model.eval() # Set to evaluation mode
                    hp = model.hp
                    
                    task_list_part = task_list_all[:task_num]
            
                    # --- Setup Hook ---
                    hook_container = ActivationHook()
                    handle = model.readout.register_forward_hook(hook_container)
                    
                    # --- Run Trials ---
                    with torch.no_grad(): 
                        for task_name_now in task_list_part:
                            for _ in range(num_batches_per_task):
                                trial = task.generate_trials(task_name_now, hp, 'random', batch_size=batch_size)
                                input_tensor = torch.from_numpy(trial.x).to(device)
                                _ = model(input_tensor) 
                                
                    # --- Remove Hook ---
                    handle.remove()

                    # --- Process Collected Activations ---
                    if not hook_container.activations:
                        print(f"Warning: No activations collected for {model_path}")
                        continue
                        
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
    
    if df.empty:
        print("DataFrame is empty after loading/computation. Cannot plot.")
        return None
        
    model_color_dict = { 64: '#003f5c', 32: '#4472c4', 16: '#5cb85c', 8: '#ffa600' }
    
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    
    sns.scatterplot(
        data=df, x='strain', y='modularity', hue='model_size',
        palette=model_color_dict, ax=ax, alpha=0.7, s=5
    )
    
    # --- START: NEW PLOTTING LOGIC ---
    # Loop through each model size to plot its own regression line
    for model_size in sorted(df['model_size'].unique()):
        group_df = df[df['model_size'] == model_size]
        
        if group_df.shape[0] > 1:
            sns.regplot(
                data=group_df,
                x='strain',
                y='modularity',
                scatter=False, # We already have the main scatterplot
                ax=ax,
                color=model_color_dict.get(model_size, 'gray'), # Use the same color as the dots
                line_kws={'linestyle': '--', 'linewidth': 0.75}
            )
    # --- END: NEW PLOTTING LOGIC ---
    
# --- START: NEW MIXED-EFFECT MODEL ANALYSIS (R-BRIDGE) ---
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    # Initialize variables
    stat_text = "Stats failed"
    slope = 0.0
    p_value_fixed = 1.0
    p_value_re = 1.0 
    method_name = "None"

    print("Running Statistical Analysis via R (lme4 + RLRsim)...")

    try:
        # 1. Define the R function script
        r_script = """
        library(lme4)
        library(RLRsim)
        
        function(data) {
            # Ensure model_size is a factor
            data$model_size <- as.factor(data$model_size)

            # Fit model
            m_full <- lmer(modularity ~ strain + (1 | model_size), data=data, REML=TRUE)
            
            # 1. Fixed Effect (Slope)
            coefs <- summary(m_full)$coefficients
            slope <- coefs["strain", "Estimate"]
            t_val <- coefs["strain", "t value"]
            p_fixed <- 2 * (1 - pnorm(abs(t_val)))
            
            # 2. Random Effect (exactRLRT)
            rlrt_res <- exactRLRT(m_full)
            p_re <- rlrt_res$p.value
            
            # RETURN ORDER MATTERS: 1=slope, 2=p_fixed, 3=p_re
            return(list(slope=slope, p_fixed=p_fixed, p_re=p_re))
        }
        """
        
        r_func = ro.r(r_script)
        
        # 2. Execute with Data Conversion
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)
            results = r_func(r_df)
            
            slope = results[0][0]          # 1st element (slope)
            p_value_fixed = results[1][0]  # 2nd element (p_fixed)
            p_value_re = results[2][0]     # 3rd element (p_re)
            
        method_name = "R:exactRLRT"

        print(f"R Analysis Successful.")
        print(f"Slope: {slope}")
        print(f"Fixed Effect p: {p_value_fixed}")
        print(f"Random Effect (exactRLRT) p: {p_value_re}")

    except Exception as e:
        print(f"R Bridge Failed: {e}")
        print("Ensure 'lme4' and 'RLRsim' are installed in your R environment.")

    # --- Format Text for Plot ---
    if method_name == "R:exactRLRT":
        def fmt_p(p): return "< 0.001" if p < 0.001 else f"{p:.3f}"
        
        stat_text = (
            f"$\\beta_{{TDR}} = {slope:.3f}$ ($p_{{fix}} {fmt_p(p_value_fixed)}$)\n"
            f"Rand Effect Test: $p_{{rand}} {fmt_p(p_value_re)}$"
        )
    else:
        stat_text = "R Stats Failed"

    # --- END: STATISTICAL ANALYSIS ---

    # Add text to the plot in the top-left corner
    ax.text(0.05, 0.95, stat_text, 
            transform=ax.transAxes, 
            fontsize=5, 
            verticalalignment='top',
            # Add a white box for readability
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5, edgecolor='none'))
            
    ax.set_title('Modularity Capacity Interaction', fontsize=6)
    ax.set_xlabel('Task-to-Dimensionality Ratio', fontsize=6)
    ax.set_ylabel('Final Modularity', fontsize=6)
    
    handles, labels = ax.get_legend_handles_labels()
    # Filter out any non-numeric labels that might appear
    new_labels = [f'{int(float(l))} Neurons\n({int(float(l))**2} Params)' for l in labels if l.replace('.', '', 1).isdigit()]
    # Ensure handles match the new labels
    new_handles = [h for h, l in zip(handles, labels) if l.replace('.', '', 1).isdigit()]


    ax.legend(
        new_handles, new_labels, fontsize=4, frameon=False,
        bbox_to_anchor=(0.99, 0.05), # <--- Adjust x and y
        loc='lower right',
        handletextpad=0.02)

    
    ax.spines['top'].set_linewidth(0.25)
    ax.spines['bottom'].set_linewidth(0.25)
    ax.spines['left'].set_linewidth(0.25)
    ax.spines['right'].set_linewidth(0.25)
    ax.tick_params(axis='both', labelsize=6, width=0.25)
    
    plt.tight_layout()
        
    fig.savefig(f'./figures/Fig2/Fig2h_TDR.jpg', format='jpg', dpi=300)
    fig.savefig(f'./figures/Fig2/Fig2h_TDR.svg', format='svg', dpi=300)
    print("Saved new 'Strain Plot' to ./figures/Fig2/Fig2h_TDR.svg")
    
    # --- UPDATED FINAL PRINT ---
    print(f"Mixed-Effect Model: Slope (strain)={slope:.4f}, p={p_value_fixed:.5e}")
    return df


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

    # plot_2h(model_size_list, task_num_list)

    plot_fig2a(model_size_list, color_dict)
    plot_fig2b(model_size_list, color_dict)
    for N in model_size_list:
        plot_fig2d(color_dict, N)

    num_curves = len(task_num_list)
    color_map = cm.get_cmap('Blues')
    color_indices = np.linspace(0.3, 0.90, len(task_num_list))  
    color_dict = {idx: color_map(ci) for idx, ci in zip(range(num_curves), color_indices)}

    plot_fig2efg(model_size_list, task_num_list, color_dict)
