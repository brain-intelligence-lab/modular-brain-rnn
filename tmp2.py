import torch
import numpy as np
import seaborn as sns  # 导入 seaborn
from functions.utils.eval_utils import do_eval
from datasets.multitask import rules_dict, Multitask_Batches_Realtime_Gen
import matplotlib.pyplot as plt # 导入 matplotlib
import pandas as pd # 导入 pandas 以便进行数据处理
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
    # 使用 torch.no_grad() 禁用梯度计算
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
    对单个模型检查点在多个扰动水平(alphas)下进行分析。
    
    Args:
        model_size (int): 模型的隐藏单元数量。
        seed (int): 随机种子。
        alphas (list of float): 扰动强度列表。
        
    Returns:
        list of dict: 包含该检查点在不同扰动水平下结果的列表。
                      如果模型文件不存在，则返回空列表。
    """
    train_loader = args.train_loader
    device = args.device
    step = args.step

    task_num = 20
    file_name = f'{DIRECTORY_NAME}/n_rnn_{model_size}_task_{task_num}_seed_{seed}/RNN_interleaved_learning_{step}.pth'

    # 1. 稳健性检查：确保模型文件存在
    if not os.path.exists(file_name):
        print(f"Warning: File not found, skipping. {file_name}")
        return []

    # 2. 加载模型并评估原始性能（效率优化：仅执行一次）
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

    # 3. 对每个alpha水平进行扰动分析
    results_for_checkpoint = []
    with torch.no_grad():
        for alpha in alphas:
            # 施加扰动
 
            for _ in range(10):

                # 计算信号功率 (能量)
                # 功率定义为信号平方和的均值，但在这里用总能量（平方和）更方便，因为比例是相同的
                signal_power = torch.sum(original_W_rec ** 2)

                # 将SNR从dB转换为线性比率
                # SNR(linear) = 10^(SNR(dB) / 10)
                snr_linear = 10 ** (alpha / 10.0)

                # 根据SNR计算所需的噪声功率
                #    SNR = Power_signal / Power_noise  =>  Power_noise = Power_signal / SNR
                noise_power = signal_power / snr_linear

                # 生成一个随机噪声，其形状与权重相同
                noise = torch.randn_like(original_W_rec)

                # 计算当前生成的这个随机噪声的功率
                current_noise_power = torch.sum(noise ** 2)
                
                # 计算缩放因子，使噪声达到所需的功率
                #    Power(k*N) = k^2 * Power(N) = desired_power
                #    => k = sqrt(desired_power / current_power)
                scaling_factor = torch.sqrt(noise_power / current_noise_power)

                # 缩放噪声并添加到原始权重上
                noise = scaling_factor * noise

                # print(f"  - 信号能量: {signal_power:.4f}")
                # print(f"  - 目标噪声能量: {noise_power:.4f}")
                # print(f"  - 实际注入噪声能量: {torch.sum(noise**2):.4f}")
                # final_snr = 10 * torch.log10(signal_power / torch.sum(noise**2))
                # print(f"  - 最终的信噪比 (dB): {final_snr:.2f}")

                model.recurrent_conn.weight.data = original_W_rec + noise
                perturbed_loss = get_data_loss(train_loader, model, device)
                delta_loss = perturbed_loss - pre_loss
                
                # 记录结果
                results_for_checkpoint.append({
                    'model_size': model_size,
                    'seed': seed+_,  # 添加 seed 记录
                    'step': step,
                    'snr': snr_linear,
                    'delta_loss': delta_loss
                })
                
    return results_for_checkpoint


def main(args):
    """
    主函数，用于执行整个实验、数据收集和绘图。
    """
    step = args.step

    # --- 实验参数 ---
    MODEL_SIZES = [8, 16, 32, 64]
    SEEDS = [i for i in range(100, 2100, 100)]
    # ALPHAS = [1 * i for i in range(1, 7)]
    ALPHAS = [1, 2, 4, 6, 8, 10]

    # --- 定义序列化文件名 ---
    results_filename = f'./runs/all_results_step_{step}.pkl'

    # --- 数据收集 ---
    # 判断是否有序列化文件，如果有则直接读取，否则进行计算
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
        
        # --- 序列化计算结果 ---
        with open(results_filename, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Computation finished and results saved to '{results_filename}'.")


    df = pd.DataFrame(all_results).round(2)

    # --- 数据可视化 --- #
    print(f"--- Generating plot for training step {step} ---")
    
    # 1. 筛选出当前 step 的数据
    df_step = df[df['step'] == step]
    
    # 2. 创建新的图像
    plt.figure(figsize=(1.8, 1.8))
    
    # 定义异常点样式：调整大小和透明度
    flierprops = dict(marker='o', markersize=1.2, alpha=0.8, markeredgecolor='green')


    # 3. 使用 seaborn 绘制箱形图
    sns.boxplot(x='snr', y='delta_loss', hue='model_size', flierprops=flierprops, 
                data=df_step, palette='viridis', linewidth=0.25)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)  # 设置所有框线宽度

    # 4. 设置特定于该 step 的标题和标签
    ax.tick_params(axis='both', which='major', labelsize=5, width=0.5)
    plt.title(f'Perturbation for Training Step {step}', fontsize=6)
    plt.xlabel('Signal Noise Ratio', fontsize=6)
    plt.ylabel('Change in Loss (Δ Loss)',fontsize=6)
    plt.grid(axis='y', linestyle='--', linewidth=0.25)
    plt.legend(title='Model Size', loc='upper right', fontsize=5, title_fontsize=5)
    
    plt.tight_layout()
    # 5. 保存带有 step 信息的独立图像文件
    output_filename = f'perturbation_analysis_boxplot_step_{step}.png'
    plt.savefig(output_filename, dpi=300)

    output_filename = f'perturbation_analysis_boxplot_step_{step}.svg'
    
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as '{output_filename}'")
    plt.close()

if __name__ == '__main__':
    args = start_parse()

    # --- 全局参数设定 ---
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
