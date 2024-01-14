import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import numpy as np
import time
import random
import bct
from spatially_embed.node_distance import sphere_distance, cube_distance, squre_distance
from spatially_embed.loss_function import SpatiallyLoss
from spatially_embed.dvs_dataset import load_n_mnist, load_dvs128_gesture
# from spatially_embed.spiking_model_new import SNN, LIFNeuron
from spatially_embed.shebbian_model import SNN, HebbReccurrentLayer
from spatially_embed.structural_analysis import *
from spatially_embed.node_plot import node_plot

def seed_all(seed=1029, benchmark=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def dvs_reshape(inputs):
    [B, T, C, H, W] = inputs.shape
    inputs = inputs.reshape(B, T, C * H * W)
    inputs = inputs.transpose(0, 1) # [T, B, C*H*W]
    return inputs

def train(model, data_loader, optimizer, criterion, device):
    train_loss = 0
    total_num = 0
    correct_num = 0
    model.train()
    t1 = time.time()
    for idx, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = dvs_reshape(inputs)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.mean(dim=0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_num += labels.size(0)
        correct_num += predicted.eq(labels).sum().item()
        model.hebb_update() # 更新结构性权重

    t2 = time.time()
    train_loss = train_loss / total_num
    train_acc = correct_num / total_num
    print('train loss: {:.4f}, train acc: {:.4f}, time: {:.2f}'.format(train_loss, train_acc, t2 - t1))
    return train_loss, train_acc

@torch.no_grad()
def test(model, data_loader, criterion, device):
    test_loss = 0
    total_num = 0
    correct_num = 0
    model.eval()
    for idx, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = dvs_reshape(inputs)
        outputs = model(inputs)
        outputs = outputs.mean(dim=0)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_num += labels.size(0)
        correct_num += predicted.eq(labels).sum().item()

    test_loss = test_loss / total_num
    test_acc = correct_num / total_num
    print('test loss: {:.4f}, test acc: {:.4f}'.format(test_loss, test_acc))
    return test_loss, test_acc

def save_weight(model, path, epoch):
    if not os.path.exists(path):
        os.makedirs(path)
    for name, param in model.named_parameters():
        if 'recurrent_weight' in name:
            save_name = name + '_epoch_{}.npy'.format(epoch)
            save_path = os.path.join(path, save_name)
            np.save(save_path, param.cpu().detach().numpy())

    for m in model.modules():
        if isinstance(m, HebbReccurrentLayer):
            save_name = 'structural_weight_epoch_{}.npy'.format(epoch)
            save_path = os.path.join(path, save_name)
            np.save(save_path, m.structural_weight.cpu().detach().numpy())




if __name__ == '__main__':
    seed_all(1000)
    batch_size = 64
    T = 30
    epochs = 200
    lr = 0.001
    hidden_size = 400
    radius = 1.0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. 创建数据集
    # train_set, test_set, _, _ = load_n_mnist(distributed=False, T=T,
    #                                         data_dir = '/data1/dsk/dataset/n_mnist')

    train_set, test_set, _, _ = load_dvs128_gesture(distributed=False, T=T,
                                                    data_dir = '/data1/dsk/dataset/dvs128_gesture')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                shuffle=False, num_workers=4)

    [T, C, H, W] = train_set[0][0].shape

    # 2. 创建模型

    dis_mat = 1.0 - np.load('Schaefer_dis_mat.npy')
    dis_mat = torch.from_numpy(dis_mat).float()
    model = SNN(input_size=C*H*W, hidden_size=hidden_size, output_size=11, layer_num=1,
                tau=0.5, v_th=1.0, gamma=1.0, structural_weight=dis_mat,
                ).to(device)

    # 3. 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. 创建损失函数
    criterion = nn.CrossEntropyLoss()

    # # 5. 创建正则化损失函数
    # # 5.1 创建距离矩阵
    # # dis_mat, node_coordinate = sphere_distance(hidden_size, radius)
    # dis_mat, node_coordinate = cube_distance(hidden_size)
    # # 5.2 创建正则化损失函数
    # reg_names = []
    # distances = {}
    # for name, param in model.named_parameters():
    #     if 'recurrent_weight' in name:
    #         reg_names.append(name)
    #         distances[name] = dis_mat
    #
    # reg_lambda = 0.1
    # reg_type = 'l2'
    # criterion = SpatiallyLoss(criterion, model, reg_names, distances, reg_type, reg_lambda)

    # # 6. 训练
    best_acc = 0
    best_loss = 100
    train_losses = []
    for epochi in range(epochs):
        print('Epoch: {}'.format(epochi))
        train_loss, train_acc =train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        save_weight(model, 'recurrent_weight', epochi)
        train_losses.append(train_loss)
        if test_acc > best_acc:
            best_acc = test_acc
        if test_loss < best_loss:
            best_loss = test_loss

    # save_weight(model, 'recurrent_weight', epochi)
    print('best acc: {:.4f}, best loss: {:.4f}'.format(best_acc, best_loss))


