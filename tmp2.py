import torch
import numpy as np
import seaborn as sns  # Import seaborn
from functions.utils.eval_utils import do_eval
from datasets.multitask import rules_dict, Multitask_Batches_Realtime_Gen
import matplotlib.pyplot as plt # Import matplotlib
import pandas as pd # Import pandas for data processing
import matplotlib.ticker as mticker
import pickle
import argparse
import os
from tqdm import tqdm  
import pdb


def start_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--step', default=500, type=int)
    parser.add_argument('--load_model_path', type=str, default='./runs/Fig2b-h')
    parser.add_argument('--snr_db', type=float, default=20)
    args = parser.parse_args()
    return args


def get_data_loss(data_loader, model, device):
    loss_list = []
    # Use torch.no_grad() to disable gradient computation
    with torch.no_grad():
        for input_data, target, c_mask in data_loader:
            input_data = input_data.to(device)
            target = target.to(device)
            c_mask = c_mask.to(device)

            output = model(input_data)
            output = torch.sigmoid(output)
            loss = torch.mean(torch.square((target - output) * c_mask))
            loss_list.append(loss.item())

    return np.sum(loss_list)


def run_perturbation_analysis_for_checkpoint(args, model_size, seed, alphas):
    """
    Perform perturbation analysis at multiple perturbation levels (alphas) for a single model checkpoint.

    Args:
        model_size (int): Number of hidden units in the model.
        seed (int): Random seed.
        alphas (list of float): List of perturbation intensity levels.

    Returns:
        list of dict: List containing results for this checkpoint at different perturbation levels.
                      Returns empty list if model file does not exist.
    """
    train_loader = args.train_loader
    device = args.device
    step = args.step

    task_num = 20
    file_name = f'{DIRECTORY_NAME}/n_rnn_{model_size}_task_{task_num}_seed_{seed}/RNN_interleaved_learning_{step}.pth'

    # 1. Robustness check: ensure model file exists
    if not os.path.exists(file_name):
        print(f"Warning: File not found, skipping. {file_name}")
        return []

    # 2. Load model and evaluate original performance (efficiency optimization: execute only once)
    try:
        model = torch.load(file_name, map_location=device)
        model.to(device)
    except Exception as e:
        print(f"Error loading model {file_name}: {e}")
        return []
        
    original_W_rec = model.recurrent_conn.weight.data.clone().detach()
    pre_loss = get_data_loss(train_loader, model, device)
    # log = do_eval(model, rule_train=rules_dict['all'], verbose=False)
    # original_perf = log['perf_avg'][-1]

    # 3. Perform perturbation analysis for each alpha level
    results_for_checkpoint = []
    with torch.no_grad():
        for alpha in alphas:
            # Apply perturbation
 
            for _ in range(10):

                # Calculate signal power (energy)
                # Power is defined as mean of squared signal, but using total energy (sum of squares) is more convenient here as the ratio is the same
                signal_power = torch.sum(original_W_rec ** 2)

                # Convert SNR from dB to linear scale
                # SNR(linear) = 10^(SNR(dB) / 10)
                snr_linear = 10 ** (alpha / 10.0)

                # Calculate required noise power based on SNR
                #    SNR = Power_signal / Power_noise  =>  Power_noise = Power_signal / SNR
                noise_power = signal_power / snr_linear

                # Generate random noise with the same shape as weights
                noise = torch.randn_like(original_W_rec)

                # Calculate the power of the currently generated random noise
                current_noise_power = torch.sum(noise ** 2)
                
                # Calculate scaling factor to bring noise to desired power
                #    Power(k*N) = k^2 * Power(N) = desired_power
                #    => k = sqrt(desired_power / current_power)
                scaling_factor = torch.sqrt(noise_power / current_noise_power)

                # Scale noise and add to original weights
                noise = scaling_factor * noise

                # print(f"  - Signal energy: {signal_power:.4f}")
                # print(f"  - Target noise energy: {noise_power:.4f}")
                # print(f"  - Actual injected noise energy: {torch.sum(noise**2):.4f}")
                # final_snr = 10 * torch.log10(signal_power / torch.sum(noise**2))
                # print(f"  - Final SNR (dB): {final_snr:.2f}")

                model.recurrent_conn.weight.data = original_W_rec + noise
                perturbed_loss = get_data_loss(train_loader, model, device)
                delta_loss = perturbed_loss - pre_loss
                
                # Record results
                results_for_checkpoint.append({
                    'model_size': model_size,
                    'seed': seed+_,  # Add seed record
                    'step': step,
                    'snr': snr_linear,
                    'delta_loss': delta_loss
                })
                
    return results_for_checkpoint


def main(args):
    """
    Main function for executing the entire experiment, data collection, and plotting.
    """
    step = args.step

    # --- Experiment Parameters ---
    MODEL_SIZES = [8, 16, 32, 64]
    SEEDS = [i for i in range(100, 2100, 100)]
    # ALPHAS = [1 * i for i in range(1, 7)]
    ALPHAS = [1, 2, 4, 6, 8, 10]

    # --- Define serialization filename ---
    results_filename = f'./runs/all_results_step_{step}.pkl'

    # --- Data Collection ---
    # Check if serialization file exists, if yes load directly, otherwise perform computation
    if os.path.exists(results_filename):
        print(f"Found serialized results file '{results_filename}'. Loading data...")
        with open(results_filename, 'rb') as f:
            all_results = pickle.load(f)
    else:
        print("Serialized results file not found. Starting new computation...")
        all_results = []
        param_combinations = [(size, seed, step) for size in MODEL_SIZES for seed in SEEDS]
        for model_size, seed, step in tqdm(param_combinations, desc="Running Perturbation Analysis"):
            results = run_perturbation_analysis_for_checkpoint(args, model_size, seed, ALPHAS)
            all_results.extend(results)

        # --- Serialize computation results ---
        with open(results_filename, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Computation finished and results saved to '{results_filename}'.")


    df = pd.DataFrame(all_results).round(2)

    # --- Data Visualization --- #
    print(f"--- Generating plot for training step {step} ---")

    # 1. Filter data for current step
    df_step = df[df['step'] == step]

    # 2. Create new figure
    plt.figure(figsize=(1.8, 1.8))

    # Define outlier style: adjust size and transparency
    flierprops = dict(marker='o', markersize=1.2, alpha=0.8, markeredgecolor='green')


    # 3. Use seaborn to draw boxplot
    sns.boxplot(x='snr', y='delta_loss', hue='model_size', flierprops=flierprops,
                data=df_step, palette='viridis', linewidth=0.25)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)  # Set all frame line widths

    # 4. Set step-specific title and labels
    ax.tick_params(axis='both', which='major', labelsize=5, width=0.5)
    plt.title(f'Perturbation for Training Step {step}', fontsize=6)
    plt.xlabel('Signal Noise Ratio', fontsize=6)
    plt.ylabel('Change in Loss (Δ Loss)',fontsize=6)
    plt.grid(axis='y', linestyle='--', linewidth=0.25)
    plt.legend(title='Model Size', loc='upper right', fontsize=5, title_fontsize=5)

    plt.tight_layout()
    # 5. Save independent image file with step information
    output_filename = f'perturbation_analysis_boxplot_step_{step}.png'
    plt.savefig(output_filename, dpi=300)

    output_filename = f'perturbation_analysis_boxplot_step_{step}.svg'

    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as '{output_filename}'")
    plt.close()

if __name__ == '__main__':
    args = start_parse()

    # --- Global Parameter Settings ---
    DIRECTORY_NAME = args.load_model_path 
    device = torch.device(f'cuda:{args.gpu}' if args.gpu>=0 else 'cpu')
    args.device = device

    for n_rnn in [8, 16, 32, 64]:
        std_list = []
        for seed in range(100, 2100, 100):
            model = torch.load(f'{DIRECTORY_NAME}/n_rnn_{n_rnn}_task_20_seed_{seed}/RNN_interleaved_learning_{10000}.pth', map_location=device)
            w_rec = model.recurrent_conn.weight.data.clone().detach()
            std_list.append(torch.std(w_rec).cpu().sum())
        print(np.sum(std_list))
    train_dataset = Multitask_Batches_Realtime_Gen(model.hp, num_batches=25, batch_size=32)

    train_loader = torch.utils.data.DataLoader(train_dataset, \
        batch_size = None, num_workers = 2)
    
    args.train_loader = train_loader

    main(args)
