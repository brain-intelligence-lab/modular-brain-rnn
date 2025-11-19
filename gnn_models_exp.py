import torch
from models.gnn import GCN
from functions.utils.eval_utils import calculate_modularity_in_r, calculate_modularity_for_fc_layer
from functions.utils.math_utils import lock_random_seed
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
import argparse
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import tensorflow as tf


def start_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', default=16, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--logdir', type=str, default='./runs/gnn_models_data/')
    parser.add_argument('--analyze_every_n_batches', default=20, type=int)
    parser.add_argument('--r_script_path', type=str, default="LPA_wb_plus.R")
    parser.add_argument('--dataset_dir', default='./datasets/', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--weight_scale_factor', default=0.1, type=float)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--task_idx', default=-1, type=int, help='-1 for Multi Task Learning, [0-11] for Single Task Learning')
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


def test(loader, model, task_idx=None):
    model.eval()
    all_y_true = []
    all_y_pred = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.x = data.x.float()
            out = model(data.x, data.edge_index, data.batch)
            all_y_true.append(data.y.cpu())
            all_y_pred.append(out.cpu())
            
    y_true_tensor = torch.cat(all_y_true, dim=0)
    y_pred_tensor = torch.cat(all_y_pred, dim=0)
    
    auc_scores = []

    if task_idx != -1:
        # **STL 模式 (单任务)**
        # 此时 y_pred_tensor 只有一列 [N, 1]， y_true_tensor 有多列 [N, 12]
        
        # 1. 选取正确的真实标签 (task_idx 列)
        y_true_task = y_true_tensor[:, task_idx]
        # 2. 选取模型输出 (STL模型只有一个输出，即第0列)
        y_pred_task = y_pred_tensor[:, 0]
        
        # 计算该单任务的AUC
        is_labeled = ~torch.isnan(y_true_task)
        y_true_labeled = y_true_task[is_labeled].numpy()
        y_pred_labeled = y_pred_task[is_labeled].numpy()
        
        if len(np.unique(y_true_labeled)) > 1:
            try:
                auc = roc_auc_score(y_true_labeled, y_pred_labeled)
                auc_scores.append(auc)
            except ValueError:
                pass
    else:
        # **MTL 模式 (多任务)** - 保持原始逻辑
        num_tasks = y_true_tensor.shape[1]
        
        for i in range(num_tasks):
            y_true_task = y_true_tensor[:, i]
            y_pred_task = y_pred_tensor[:, i]
            
            # 过滤掉NaN标签
            is_labeled = ~torch.isnan(y_true_task)
            y_true_labeled = y_true_task[is_labeled].numpy()
            y_pred_labeled = y_pred_task[is_labeled].numpy()
            
            # 必须同时存在正负样本才能计算AUC
            if len(np.unique(y_true_labeled)) > 1:
                try:
                    auc = roc_auc_score(y_true_labeled, y_pred_labeled)
                    auc_scores.append(auc)
                except ValueError:
                    # 某些罕见情况下，即使unique > 1，也可能出错
                    pass

    return np.mean(auc_scores) if auc_scores else 0.0


def load_data_and_plot(hidden_channels, seed_name_list, logdir="./runs/cnn_models_data/"):

    print("\n开始处理和绘制结果...")
    global_min = 1000000

    # --- 创建绘图窗口 ---
    fig = plt.figure(figsize=(12.0, 4.0))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    # 定义颜色列表，以便为不同的 class_num 分配不同的颜色
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    task_idx_list = [i for i in range(-1, 12)]

    mtl_exp_paths_list = []
    stl_exp_paths_list = []
    label_name = ['Single Task Learning', 'Multiple Tasks Learning']

    for i, task_idx in enumerate(task_idx_list):
        exp_paths_list = []
        for __, seed_name in enumerate(seed_name_list):
            file_name = f"hidden_channels_{hidden_channels}_task_idx_{task_idx}_seed_{seed_name}"

            paths = list_files(logdir, file_name)

            if paths:
                exp_paths_list.append(paths)

        assert exp_paths_list
        if task_idx == -1:
            mtl_exp_paths_list.extend(exp_paths_list)
        else:
            stl_exp_paths_list.extend(exp_paths_list)
    
    for i, exp_paths_list in enumerate([stl_exp_paths_list, mtl_exp_paths_list]):
    
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
                print(f"读取文件 {events_file} 时出错: {e}")
                continue
            
            if dirt_lpa_mod_list:
                dirt_lpa_mod_array.append(np.array(dirt_lpa_mod_list))
            if weight_in_mod_list:
                weight_in_mod_array.append(np.array(weight_in_mod_list))
            if weight_out_mod_list:
                weight_out_mod_array.append(np.array(weight_out_mod_list))
            if loss_avg_list:
                loss_avg_array.append(np.array(loss_avg_list))

        assert dirt_lpa_mod_list and weight_in_mod_list and weight_out_mod_list and loss_avg_array
        
        # 找到最短的序列长度，以对齐数据
        min_len = min(len(arr) for arr in dirt_lpa_mod_array + weight_in_mod_array + weight_out_mod_array + loss_avg_array)
        global_min = min(global_min, min_len)
        min_len = global_min
        print(min_len)
        

        # 截断所有数组到最短长度并堆叠
        dirt_lpa_mod_array = np.stack([arr[:min_len] for arr in dirt_lpa_mod_array], axis=0)
        weight_in_mod_array = np.stack([arr[:min_len] for arr in weight_in_mod_array], axis=0)
        weight_out_mod_array = np.stack([arr[:min_len] for arr in weight_out_mod_array], axis=0)
        loss_avg_array = np.stack([arr[:min_len] for arr in loss_avg_array], axis=0)

        # --- 计算均值和标准误差 ---
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

        # --- 在对应的子图上绘制曲线 ---
        color = color_list[i]
        ax1.plot(analysis_steps, dirt_lpa_mod_mean, label=label_name[i], color=color)
        ax1.fill_between(analysis_steps, dirt_lpa_mod_mean - dirt_lpa_mod_ste, dirt_lpa_mod_mean + dirt_lpa_mod_ste, 
                         color=color, alpha=0.2)

        ax2.plot(analysis_steps, weight_in_mod_mean, label=label_name[i], color=color)
        ax2.fill_between(analysis_steps, weight_in_mod_mean - weight_in_mod_ste,
                         weight_in_mod_mean + weight_in_mod_ste, 
                         color=color, alpha=0.2)

        ax3.plot(analysis_steps, weight_out_mod_mean, label=label_name[i], color=color)
        ax3.fill_between(analysis_steps, weight_out_mod_mean - weight_out_mod_ste,
                         weight_out_mod_mean + weight_out_mod_ste, 
                         color=color, alpha=0.2)
        

    # --- 设置子图的标题、标签和图例 ---
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

    # --- 调整布局并保存图片 ---
    plt.suptitle(f'Modularity Evolution During Training on Tox21 Dataset\
            \n(# hidden_channels: {hidden_channels})', fontsize=16)
    plt.tight_layout()

    save_path = f"./figures/modularity_evolution_hidden_channels_{hidden_channels}_comparison"
    
    fig.savefig(f"{save_path}.jpg", format='jpg')
    fig.savefig(f"{save_path}.svg", format='svg')


if __name__ == "__main__":
    args = start_parse()

    if args.load_data_plot:
        seed_name_list = [i for i in range(100, 1650, 100)]
        load_data_and_plot(args.hidden_channels, seed_name_list, args.logdir)
        exit(0)


    lock_random_seed(seed=args.seed)    
    
    # --- 数据集加载与准备 ---
    dataset = MoleculeNet(root=args.dataset_dir, name="Tox21")
    shuffled_indices = torch.randperm(len(dataset))

    train_size = int(0.9 * len(dataset))
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')

    layers_to_scale = ['conv1', 'conv2']
    writer = SummaryWriter(log_dir=args.logdir)
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args_file_path = os.path.join(args.logdir, "args.txt")
    
    with open(args_file_path, 'w') as f:
        f.write(f'current_time: {current_time}\n')
        for arg, value in sorted(vars(args).items()):
            f.write(f'{arg}: {value}\n')

    criterion = torch.nn.BCEWithLogitsLoss()

    epochs = args.epochs
    task_idx = args.task_idx

    if task_idx == -1:
        print("\n--- 开始多任务学习 (MTL) 实验 ---")

        # 根据索引创建数据集

        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")

        model = GCN(
            hidden_channels=args.hidden_channels,
            num_node_features=dataset.num_node_features,
            num_classes=dataset.num_classes
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    else:

        print("\n--- 开始单任务学习 (STL) 实验 ---")
        all_tasks_val_aucs = []

        # 每次都重新初始化一个新模型
        model = GCN(
            hidden_channels=args.hidden_channels,
            num_node_features=dataset.num_node_features,
            num_classes=1
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            
        task_train_indices = [i for i, data in enumerate(train_dataset) if not torch.isnan(data.y[0, task_idx])]
        task_val_indices = [i for i, data in enumerate(val_dataset) if not torch.isnan(data.y[0, task_idx])]
            
        # 从已经划分好的训练/验证集中再次筛选
        task_train_subset = torch.utils.data.Subset(train_dataset, task_train_indices)
        task_val_subset = torch.utils.data.Subset(val_dataset, task_val_indices)

        train_loader = DataLoader(task_train_subset, batch_size=64, shuffle=True, num_workers=2)
        val_loader = DataLoader(task_val_subset, batch_size=64, shuffle=False, num_workers=2)
        

    model.apply_scale_factor(scale_factor=args.weight_scale_factor, layers_to_scale=layers_to_scale)

    global_batch_counter = 0
    loss_sum = 0.0
    total_batches_per_epoch = len(train_loader)
    for epoch in range(1, epochs + 1):
        model.train()
        for idx, data in enumerate(train_loader):
            
            data = data.to(device)
            optimizer.zero_grad()
            data.x = data.x.float()
            out = model(data.x, data.edge_index, data.batch)
            y = data.y.to(torch.float32)
            
            if task_idx != -1:
                y = y[:, task_idx].unsqueeze(1)
            
            is_labeled = y == y
            if is_labeled.sum() == 0: continue
                
            loss = criterion(out[is_labeled], y[is_labeled])
            loss.backward()
            optimizer.step()

            global_batch_counter += 1
            loss_sum += loss.item()
                        
            if global_batch_counter % args.analyze_every_n_batches == 0:
                print("-" * 50)
                if global_batch_counter != 0:
                    print(f"Epoch [{epoch}/{args.epochs}], Batch [{idx+1}/{total_batches_per_epoch}], Loss: {loss_sum:.4f}")
            
                weights_list = []
                for idx, layer_to_analyze in enumerate(layers_to_scale):
                    print(f" 开始分析层 '{layer_to_analyze}' 的权重模块度 (Global Batch: {global_batch_counter})...")

                    target_layer = None
                    try:
                        target_layer = dict(model.named_modules())[layer_to_analyze]

                    except KeyError:
                        print(f"错误：模型中找不到名为 '{layer_to_analyze}' 的层。")
                        print(f"可用层包括: {[name for name, _ in model.named_modules() if '.' not in name and name != '']}")
                

                    weight_tensor = target_layer.lin.weight.detach()
                
                    # --- 处理权重张量 ---
                    if weight_tensor.dim() == 4: # 卷积层权重

                        if idx == 0: 
                            # 形状: (out_channels, in_channels, kH, kW) 展开转置为 2D: (in_channels * kH * kW, out_channels)
                            weight_tensor_new = weight_tensor.reshape(weight_tensor.size(0), -1).T
                        else:
                            # (out_channels, in_channels, kH, kW) ---> (in_channels, out_channels,  kH,  kW) ----> (in_channels, out_channels * kH * kW)
                            weight_tensor_new = weight_tensor.permute(1, 0, 2, 3)
                            weight_tensor_new = weight_tensor_new.reshape(weight_tensor_new.size(0), -1)

                        print(f" 检测到4D卷积层权重 (形状: {weight_tensor.shape}), 已展开为 2D 矩阵并转置 (形状: {weight_tensor_new.shape})")

                    elif weight_tensor.dim() == 2: # 全连接层权重
                        weight_tensor_new = weight_tensor
                        print(f" 检测到2D全连接层权重 (形状: {weight_tensor.shape})")
                    else:
                        raise ValueError(f"不支持的权重维度: {weight_tensor.dim()}")
                    
                    weights_list.append(weight_tensor_new.cpu().numpy())

                    if idx == 1:

                        if weight_tensor.dim() == 4: # 卷积层权重
                            # 形状: (out_channels, in_channels, kH, kW) 展开为 2D: (out_channels, in_channels * kH * kW)
                            reshaped_weight = weight_tensor.reshape(weight_tensor.size(0), -1)
                            print(f" 检测到4D卷积层权重 (形状: {weight_tensor.shape}), 已展开为 2D 矩阵 (形状: {reshaped_weight.shape})")
                            weight_tensor = reshaped_weight

                        elif weight_tensor.dim() == 2: # 全连接层权重
                            print(f" 检测到2D全连接层权重 (形状: {weight_tensor.shape})。")
                        else:
                            raise ValueError(f"不支持的权重维度: {weight_tensor.dim()}")

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

        if epoch % 5 == 0:
            val_auc = test(val_loader, model, task_idx)
            print(f'Epoch: {epoch:03d}/{epochs:03d}, Val AUC: {val_auc:.4f}')
    

