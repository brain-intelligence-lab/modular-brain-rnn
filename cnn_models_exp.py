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
import tensorflow as tf
from functions.utils.eval_utils import calculate_modularity_in_r, calculate_modularity_for_fc_layer
from functions.utils.math_utils import lock_random_seed
import argparse
import bct
import pdb


def start_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels_1', default=8, type=int)
    parser.add_argument('--channels_2', default=8, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--logdir', type=str, default='./runs/cnn_models_data/')
    parser.add_argument('--analyze_every_n_batches', default=20, type=int)
    parser.add_argument('--r_script_path', type=str, default="LPA_wb_plus.R")
    parser.add_argument('--dataset_dir', default='./datasets/cifar10', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--weight_scale_factor', default=0.1, type=float)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--class_num', default=3, type=int)
    parser.add_argument('--classes_to_use', nargs='+', help='A set of class id', default=None)
    parser.add_argument('--load_data_plot', action='store_true')
    args = parser.parse_args()
    return args



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


def load_data_and_plot(channels_1, channels_2, class_num_list, seed_name_list, logdir="./runs/cnn_models_data/"):

    print("\nStarting to process and plot results...")

    # --- Create plotting window ---
    fig = plt.figure(figsize=(12.0, 4.0))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    # Define color list to assign different colors for different class_num
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, class_num in enumerate(class_num_list):
        exp_paths_list = []
        for __, seed_name in enumerate(seed_name_list):
            file_name = f"c1_{channels_1}_c2_{channels_2}_class_num_{class_num}_seed_{seed_name}"

            paths = list_files(logdir, file_name)

            if paths:
                exp_paths_list.append(paths)

        assert exp_paths_list

        dirt_lpa_mod_array = []
        weight_in_mod_array = []
        weight_out_mod_array = []
        loss_avg_array = []

        for ii, events_file in enumerate(exp_paths_list):
            dirt_lpa_mod_list = []
            weight_in_mod_list = []
            weight_out_mod_list = []
            loss_avg_list = []
            
            try:
                for e in tf.compat.v1.train.summary_iterator(events_file):
                    for v in e.summary.value:
                        if v.tag == 'DIRT_LPA_wb_plus':
                            dirt_lpa_mod_list.append(v.simple_value)
                        if v.tag == 'weight_in_mod':
                            weight_in_mod_list.append(v.simple_value)
                        if v.tag == 'weight_out_mod':
                            weight_out_mod_list.append(v.simple_value)
                        if v.tag == 'Loss':
                            loss_avg_list.append(v.simple_value)
            except Exception as e:
                print(f"Error reading file {events_file}: {e}")
                continue
            
            if dirt_lpa_mod_list:
                dirt_lpa_mod_array.append(np.array(dirt_lpa_mod_list))
            if weight_in_mod_list:
                weight_in_mod_array.append(np.array(weight_in_mod_list))
            if weight_out_mod_list:
                weight_out_mod_array.append(np.array(weight_out_mod_list))
            if loss_avg_list:
                loss_avg_array.append(np.array(loss_avg_list))

        if not dirt_lpa_mod_list or not weight_in_mod_list or not weight_out_mod_list or not loss_avg_array:
            print(f"Warning: class_num={class_num} data incomplete, skipping...")
            continue
        
        # Find the shortest sequence length to align data
        min_len = min(len(arr) for arr in dirt_lpa_mod_array + weight_in_mod_array + weight_out_mod_array + loss_avg_array)
        print(min_len)
        
        # Truncate all arrays to shortest length and stack
        dirt_lpa_mod_array = np.stack([arr[:min_len] for arr in dirt_lpa_mod_array], axis=0)
        weight_in_mod_array = np.stack([arr[:min_len] for arr in weight_in_mod_array], axis=0)
        weight_out_mod_array = np.stack([arr[:min_len] for arr in weight_out_mod_array], axis=0)
        loss_avg_array = np.stack([arr[:min_len] for arr in loss_avg_array], axis=0)

        # --- Calculate mean and standard error ---
        dirt_lpa_mod_mean = np.mean(dirt_lpa_mod_array, axis=0)
        dirt_lpa_mod_ste = np.std(dirt_lpa_mod_array, axis=0) / np.sqrt(dirt_lpa_mod_array.shape[0])

        weight_in_mod_mean = np.mean(weight_in_mod_array, axis=0)
        weight_in_mod_ste = np.std(weight_in_mod_array, axis=0) / np.sqrt(weight_in_mod_array.shape[0])

        weight_out_mod_mean = np.mean(weight_out_mod_array, axis=0)
        weight_out_mod_ste = np.std(weight_out_mod_array, axis=0) / np.sqrt(weight_out_mod_array.shape[0])
        
        # loss_avg_mean = np.mean(loss_avg_array, axis=0)
        # loss_avg_ste = np.std(loss_avg_array, axis=0) / np.sqrt(loss_avg_array.shape[0])
        
        analysis_steps = list(range(args.analyze_every_n_batches, (min_len + 1) * args.analyze_every_n_batches, args.analyze_every_n_batches))
        analysis_steps = analysis_steps[:min_len]

        # --- Plot curves on corresponding subplots ---
        color = color_list[i]
        ax1.plot(analysis_steps, dirt_lpa_mod_mean, label=f'Class Num {class_num}', color=color)
        ax1.fill_between(analysis_steps, dirt_lpa_mod_mean - dirt_lpa_mod_ste, dirt_lpa_mod_mean + dirt_lpa_mod_ste, 
                         color=color, alpha=0.2)

        ax2.plot(analysis_steps, weight_in_mod_mean, label=f'Class Num {class_num}', color=color)
        ax2.fill_between(analysis_steps, weight_in_mod_mean - weight_in_mod_ste,
                         weight_in_mod_mean + weight_in_mod_ste, 
                         color=color, alpha=0.2)

        ax3.plot(analysis_steps, weight_out_mod_mean, label=f'Class Num {class_num}', color=color)
        ax3.fill_between(analysis_steps, weight_out_mod_mean - weight_out_mod_ste,
                         weight_out_mod_mean + weight_out_mod_ste, 
                         color=color, alpha=0.2)
        

    
    # --- Set subplot titles, labels, and legends ---
    ax1.set_xlabel('Global Training Batches')
    ax1.set_ylabel('Modularity')
    ax1.set_title(f'DIRT_LPA_wb_plus')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Global Training Batches')
    ax2.set_ylabel('Modularity')
    ax2.set_title(f'weight_in_modularity')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_xlabel('Global Training Batches')
    ax3.set_ylabel('Modularity')
    ax3.set_title(f'weight_out_modularity')
    ax3.legend()
    ax3.grid(True)

    # --- Adjust layout and save figure ---
    plt.suptitle(f'Modularity Evolution During Contrastive learning\
            \n(# channel_in: {channels_1}, channel_out: {channels_2})', fontsize=16)
    plt.tight_layout()
    save_path = f"./figures/modularity_evolution_c1_{channels_1}_c2_{channels_2}_comparison"

    fig.savefig(f"{save_path}.jpg", format='jpg')
    fig.savefig(f"{save_path}.svg", format='svg')
    


if __name__ == "__main__":
    args = start_parse()
    if args.load_data_plot:
        seed_name_list = [i for i in range(100, 1650, 100)]
        class_num_list = [2, 10]
        load_data_and_plot(args.channels_1, args.channels_2, class_num_list, seed_name_list, args.logdir)
        exit(0)
        
    lock_random_seed(seed=args.seed)    
    assert args.classes_to_use is None or len(args.classes_to_use) == args.class_num
    if args.classes_to_use is not None:
        args.classes_to_use = set(map(int, args.classes_to_use))
        assert all(0 <= x < 10 for x in args.classes_to_use)
    else:
        args.classes_to_use = set(range(args.class_num))
    print(f"使用的class别IDlist: {args.classes_to_use if args.classes_to_use is not None else '前'+str(args.class_num)+'class'}")
    
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    
    layers_to_scale = ['conv1', 'conv2']
    writer = SummaryWriter(log_dir=args.logdir)
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args_file_path = os.path.join(args.logdir, "args.txt")
    
    with open(args_file_path, 'w') as f:
        f.write(f'current_time: {current_time}\n')
        for arg, value in sorted(vars(args).items()):
            f.write(f'{arg}: {value}\n')


    # --- Data loading and preprocessing ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    
    cifar10_full_train = torchvision.datasets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=transform)
    
    cifar10_class_indices = get_cifar10_class_indices(cifar10_full_train)
    classes_to_use = args.classes_to_use
    print(f"Tasks will use the following classes: {classes_to_use}")

    analysis_steps = [] 

    train_dataset = Cifar10PairsDataset(cifar10_full_train, cifar10_class_indices, classes_to_use, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    model = SiameseNet(channels_1=args.channels_1, channels_2=args.channels_2, output_channels=64).to(device)
    model.apply_scale_factor(scale_factor=args.weight_scale_factor, layers_to_scale=layers_to_scale)

    # --- Model, loss function, optimizer (re-initialize for each run) ---
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    # --- Training loop ---
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

            # 反向传播和optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_batch_counter += 1
            loss_sum += loss.item()

            # --- module度Calculate ---
            if global_batch_counter % args.analyze_every_n_batches == 0:
                print("-" * 50)
                if global_batch_counter != 0:
                    print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{total_batches_per_epoch}], Loss: {loss_sum:.4f}")
            
                weights_list = []
                for idx, layer_to_analyze in enumerate(layers_to_scale):
                    print(f" Startanalysislayer '{layer_to_analyze}' 的weightmodule度 (Global Batch: {global_batch_counter})...")

                    target_layer = None
                    try:
                        target_layer = dict(model.named_modules())[layer_to_analyze]

                    except KeyError:
                        print(f"error：model中找不到名为 '{layer_to_analyze}' 的layer。")
                        print(f"可用layer包括: {[name for name, _ in model.named_modules() if '.' not in name and name != '']}")
                

                    weight_tensor = target_layer.weight.detach()
                
                    # --- Processweight张量 ---
                    if weight_tensor.dim() == 4: # 卷积layerweight

                        if idx == 0: 
                            # 形状: (out_channels, in_channels, kH, kW) 展开转置为 2D: (in_channels * kH * kW, out_channels)
                            weight_tensor_new = weight_tensor.reshape(weight_tensor.size(0), -1).T
                        else:
                            # (out_channels, in_channels, kH, kW) ---> (in_channels, out_channels,  kH,  kW) ----> (in_channels, out_channels * kH * kW)
                            weight_tensor_new = weight_tensor.permute(1, 0, 2, 3)
                            weight_tensor_new = weight_tensor_new.reshape(weight_tensor_new.size(0), -1)

                        print(f" 检测到4D卷积layerweight (形状: {weight_tensor.shape}), 已展开为 2D matrix并转置 (形状: {weight_tensor_new.shape})")

                    elif weight_tensor.dim() == 2: # 全connectionlayerweight
                        weight_tensor_new = weight_tensor
                        print(f" 检测到2D全connectionlayerweight (形状: {weight_tensor.shape})")
                    else:
                        raise ValueError(f"不支持的weightdimension: {weight_tensor.dim()}")
                    
                    weights_list.append(weight_tensor_new.cpu().numpy())

                    if idx == 1:

                        if weight_tensor.dim() == 4: # 卷积layerweight
                            # 形状: (out_channels, in_channels, kH, kW) 展开为 2D: (out_channels, in_channels * kH * kW)
                            reshaped_weight = weight_tensor.reshape(weight_tensor.size(0), -1)
                            print(f" 检测到4D卷积layerweight (形状: {weight_tensor.shape}), 已展开为 2D matrix (形状: {reshaped_weight.shape})")
                            weight_tensor = reshaped_weight

                        elif weight_tensor.dim() == 2: # 全connectionlayerweight
                            print(f" 检测到2D全connectionlayerweight (形状: {weight_tensor.shape})。")
                        else:
                            raise ValueError(f"不支持的weightdimension: {weight_tensor.dim()}")


                        weight_numpy = np.abs(weight_tensor.cpu().numpy())
                        mod1, mod2 = calculate_modularity_in_r(weight_numpy, args.r_script_path)
                        
                        writer.add_scalar(tag = 'LPA_wb_plus', scalar_value = mod1, global_step = global_batch_counter)
                        writer.add_scalar(tag = 'DIRT_LPA_wb_plus', scalar_value = mod2, global_step = global_batch_counter)
                        
                        

                mod1, mod2 = calculate_modularity_for_fc_layer(weights_list[0], weights_list[1])
                writer.add_scalar(tag = 'weight_in_mod', scalar_value = mod1, global_step = global_batch_counter)
                writer.add_scalar(tag = 'weight_out_mod', scalar_value = mod2, global_step = global_batch_counter)

                writer.add_scalar(tag = 'Loss', scalar_value = loss_sum, global_step = global_batch_counter)
                loss_sum = 0.0    
                print("-" * 50)

    print("\n所有trainingComplete...")