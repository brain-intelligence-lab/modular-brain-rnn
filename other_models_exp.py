import torch
import torchvision
from datasets.diy_metric_learning_task import get_cifar10_class_indices, Cifar10PairsDataset, ContrastiveLoss
from models.cnn import SiameseNet
from tensorboardX import SummaryWriter
from datetime import datetime
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt 
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
import tensorflow as tf
from functions.utils.eval_utils import lock_random_seed
import argparse
import pdb


def start_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels_1', default=32, type=int)
    parser.add_argument('--channels_2', default=32, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--log_dir', type=str, default='./runs/cnn_models_data/')
    parser.add_argument('--analyze_every_n_batches', default=20, type=int)
    parser.add_argument('--r_script_path', type=str, default="LPA_wb_plus.R")
    parser.add_argument('--dataset_dir', default='/data_nv/dataset/cifar10', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--weight_scale_factor', default=0.1, type=float)
    parser.add_argument('--layer_to_scale', type=str, default='conv2')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--class_num', default=2, type=int)
    parser.add_argument('--classes_to_use', nargs='+', help='A set of class id', default=None)
    parser.add_argument('--load_data_plot', action='store_true')
    args = parser.parse_args()
    return args


def calculate_modularity_in_r(weight_matrix: np.ndarray, r_script_path:str, verbose:bool=False):
    assert os.path.exists(r_script_path), f"R script '{r_script_path}' not found."

    numpy2ri.activate() # 激活 numpy -> R matrix 的自动转换
    r = robjects.r

    r.source(r_script_path) # 加载 R 脚本
    lpa_wb_plus_func = robjects.globalenv['LPA_wb_plus']
    dirt_lpa_func = robjects.globalenv['DIRT_LPA_wb_plus']
    r_matrix = robjects.conversion.py2rpy(weight_matrix) # 转换为 R 矩阵

    # 调用 R 函数
    if verbose:
        print(" 正在调用 LPA_wb_plus(MAT)...") 
    mod1_result = lpa_wb_plus_func(r_matrix)
    
    if verbose:
        print(" 正在调用 DIRT_LPA_wb_plus(MAT)...") 
    mod2_result = dirt_lpa_func(r_matrix)

    modularity1 = mod1_result.rx2('modularity')[0]
    modularity2 = mod2_result.rx2('modularity')[0]

    numpy2ri.deactivate() # 关闭转换
    
    return modularity1, modularity2


def list_files(directory, name):
    path_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if '.pth' in file or '.txt' in file:
                continue
            if name == root.split('/')[-1]:
                path_list.append(os.path.join(root, file))

    assert len(path_list) <= 1
    if len(path_list) == 1:
        return path_list[0]

def load_data_and_plot(channels_1, channels_2, class_num, seed_name_list, log_dir="./runs/cnn_models_data/"):

    # --- 结果分析与绘图 ---
    print("\n开始处理和绘制结果...")

    exp_paths_list = []
    for __, seed_name in enumerate(seed_name_list):
        file_name = f"c1_{channels_1}_c2_{channels_2}_class_num_{class_num}_seed_{seed_name}"
        paths = list_files(log_dir, file_name)
        if paths != None:
            exp_paths_list.append(paths)

    modularity_array_1 = []
    modularity_array_2 = []
    loss_avg_array = []

    for ii, events_file in enumerate(exp_paths_list):
        modularity_list_1 = [0]
        modularity_list_2 = [0]
        loss_avg_list = [0]
        
        for e in tf.compat.v1.train.summary_iterator(events_file):
            for v in e.summary.value:
                if v.tag == 'LPA_wb_plus':
                    modularity_list_1.append(v.simple_value)
                if v.tag == 'DIRT_LPA_wb_plus':
                    modularity_list_2.append(v.simple_value)
                if v.tag == 'Loss':
                    loss_avg_list.append(v.simple_value)
        
        modularity_array_1.append(np.array(modularity_list_1))
        modularity_array_2.append(np.array(modularity_list_2))
        loss_avg_array.append(np.array(loss_avg_list))
    

    min_len = min(len(arr) for arr in modularity_array_1)
    print(min_len)
    modularity_array_1 = np.stack(modularity_array_1, axis=0)
    modularity_array_2 = np.stack(modularity_array_2, axis=0)
    loss_avg_array = np.stack(loss_avg_array, axis=0)
    

    modularity_mean_1 = np.mean(modularity_array_1, axis=0)
    modularity_std_1 = np.std(modularity_array_1, axis=0)
    modularity_ste_1 = modularity_std_1 / np.sqrt(modularity_array_1.shape[0])
    
    
    modularity_mean_2 = np.mean(modularity_array_2, axis=0)
    modularity_std_2 = np.std(modularity_array_2, axis=0)
    modularity_ste_2 = modularity_std_2 / np.sqrt(modularity_array_2.shape[0])
    
    loss_avg_mean = np.mean(loss_avg_array, axis=0)
    loss_avg_std = np.std(loss_avg_array, axis=0)
    loss_avg_ste = loss_avg_std / np.sqrt(loss_avg_array.shape[0])
    
    analysis_steps = list(range(args.analyze_every_n_batches, \
    (min_len + 1) * args.analyze_every_n_batches, args.analyze_every_n_batches))

    # --- 绘图 ---
    plt.figure(figsize=(18, 6))

    # 绘制 LPA_wb_plus 的模块度曲线
    plt.subplot(1, 3, 1)   
    plt.plot(analysis_steps, modularity_mean_1, label='Mean Modularity (LPA_wb_plus)', color='blue')
    plt.fill_between(analysis_steps, modularity_mean_1 - modularity_ste_1, modularity_mean_1 + modularity_ste_1, 
                     color='blue', alpha=0.2, label='Ste Dev Range')
    plt.xlabel('Global Training Batches')
    plt.ylabel('Modularity')
    plt.title('Modularity Evolution (LPA_wb_plus)')
    plt.legend()
    plt.grid(True)

    # 绘制 DIRT_LPA_wb_plus 的模块度曲线
    plt.subplot(1, 3, 2) 
    plt.plot(analysis_steps, modularity_mean_2, label='Mean Modularity (DIRT_LPA_wb_plus)', color='red')
    plt.fill_between(analysis_steps, modularity_mean_2 - modularity_ste_2,
                     modularity_mean_2 + modularity_ste_2, 
                     color='red', alpha=0.2, label='Std Dev Range')
    plt.xlabel('Global Training Batches')
    plt.ylabel('Modularity')
    plt.title('Modularity Evolution (DIRT_LPA_wb_plus)')
    plt.legend()
    plt.grid(True)
    
    
    plt.subplot(1, 3, 3)
    plt.plot(analysis_steps, loss_avg_mean, label='Mean Loss', color='green')
    plt.fill_between(analysis_steps, loss_avg_mean - loss_avg_ste, loss_avg_mean
                        + loss_avg_ste, color='green', alpha=0.2, label='Std Dev Range')
    plt.xlabel('Global Training Batches')
    plt.ylabel('Loss')
    plt.title('Training Loss Evolution')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # 调整子图布局，避免重叠
    plt.savefig(f"modularity_evolution_c1_{channels_1}_c2_{channels_2}_class_num_{class_num}.png") # 保存图片
    print("\n模块度演变曲线图已保存为 'modularity_evolution.png' 并已显示。")


if __name__ == "__main__":
    args = start_parse()
    if args.load_data_plot:
        seed_name_list = [i for i in range(100, 1050, 100)]
        load_data_and_plot(args.channels_1, args.channels_2, args.class_num, seed_name_list, args.log_dir)
        exit(0)
        
    lock_random_seed(seed=args.seed)    
    assert args.classes_to_use is None or len(args.classes_to_use) == args.class_num
    if args.classes_to_use is not None:
        args.classes_to_use = set(map(int, args.classes_to_use))
        assert all(0 <= x < 10 for x in args.classes_to_use)
    else:
        args.classes_to_use = set(range(args.class_num))
    print(f"使用的类别ID列表: {args.classes_to_use if args.classes_to_use is not None else '前'+str(args.class_num)+'类'}")
    
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    
    layer_to_analyze = args.layer_to_scale
    writer = SummaryWriter(log_dir=args.log_dir)
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args_file_path = os.path.join(args.log_dir, "args.txt")
    
    with open(args_file_path, 'w') as f:
        f.write(f'current_time: {current_time}\n')
        for arg, value in sorted(vars(args).items()):
            f.write(f'{arg}: {value}\n')


    # --- 数据加载和预处理 ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    
    cifar10_full_train = torchvision.datasets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=transform)
    
    cifar10_class_indices = get_cifar10_class_indices(cifar10_full_train)
    classes_to_use = args.classes_to_use
    print(f"任务将使用以下类别: {classes_to_use}")

    analysis_steps = [] 

    train_dataset = Cifar10PairsDataset(cifar10_full_train, cifar10_class_indices, classes_to_use, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    model = SiameseNet(channels_1=args.channels_1, channels_2=args.channels_2, output_channels=64).to(device)
    model.apply_scale_factor(scale_factor=0.1, layer_to_scale=args.layer_to_scale)

    # --- 模型、损失函数、优化器 (为每个 run 重新初始化) ---
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    # --- 训练循环 ---
    total_batches_per_epoch = len(train_loader)
    global_batch_counter = 0
    loss_sum = 0.0

    for epoch in range(args.epochs):
        for i, (x1, x2, labels) in enumerate(train_loader):

            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)

            # 前向传播
            embd1, embd2 = model(x1, x2)
            loss = criterion(embd1, embd2, labels) 

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_batch_counter += 1
            loss_sum += loss.item()

            # --- 模块度计算 ---
            if global_batch_counter % args.analyze_every_n_batches == 0:
                print("-" * 50)
                if global_batch_counter != 0:
                    print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{total_batches_per_epoch}], Loss: {loss_sum:.4f}")
            
                print(f" 开始分析层 '{layer_to_analyze}' 的权重模块度 (Global Batch: {global_batch_counter})...")

                try:
                    target_layer = dict(model.named_modules())[layer_to_analyze]
                    weights_tensor = target_layer.weight.detach().cpu()
                    
                    # --- 处理权重张量 ---
                    if weights_tensor.dim() == 4: # 卷积层权重
                        # 形状: (out_channels, in_channels, kH, kW) 展开为 2D: (out_channels, in_channels * kH * kW)
                        reshaped_weights = weights_tensor.reshape(weights_tensor.size(0), -1)
                        print(f" 检测到4D卷积层权重 (形状: {weights_tensor.shape}), 已展开为 2D 矩阵 (形状: {reshaped_weights.shape})")
                        weights_tensor = reshaped_weights

                    elif weights_tensor.dim() == 2: # 全连接层权重
                        print(f" 检测到2D全连接层权重 (形状: {weights_tensor.shape})。")
                    else:
                        raise ValueError(f"不支持的权重维度: {weights_tensor.dim()}")
                    
                    weights_numpy = np.abs(weights_tensor.numpy())
                    mod1, mod2 = calculate_modularity_in_r(weights_numpy, args.r_script_path)
                    
                    writer.add_scalar(tag = 'LPA_wb_plus', scalar_value = mod1, global_step = global_batch_counter)
                    writer.add_scalar(tag = 'DIRT_LPA_wb_plus', scalar_value = mod2, global_step = global_batch_counter)
                    writer.add_scalar(tag = 'Loss', scalar_value = loss_sum, global_step = global_batch_counter)
                    loss_sum = 0.0    

                except KeyError:
                    print(f"错误：模型中找不到名为 '{layer_to_analyze}' 的层。")
                    print(f"可用层包括: {[name for name, _ in model.named_modules() if '.' not in name and name != '']}")
                
                print("-" * 50)

    print("\n所有训练重复实验完成。")