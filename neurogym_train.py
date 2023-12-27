import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import numpy as np
import time
import random
from spatially_embed.node_distance import sphere_distance
from spatially_embed.loss_function import SpatiallyLoss
from spatially_embed.neurogym_dataset import create_dataset
from spatially_embed.spiking_model import SNN, LIFNeuron

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

def train(model, dataset, epochs, optimizer, criterion, device):
    model.train()
    t1 = time.time()
    for epochi in range(epochs):
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (epochi + 1)% 100 == 0:
            t2 = time.time()
            print('epoch: {}, loss: {:.4f}, time: {:.2f}'.format(epochi + 1, loss.item(), t2 - t1))

def performance_analysis(model, env, num_trials, device):
    acc = 0
    model.eval()
    for _ in range(num_trials):
        env.new_trial()
        ob, gt = env.ob, env.gt
        ob = ob[:, np.newaxis, :]
        inputs = torch.from_numpy(ob).type(torch.float).to(device)
        action_pred = model(inputs)
        action_pred = action_pred.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1)
        acc += (gt[-1] == action_pred[-1, 0])
    acc = acc / num_trials
    print('Average performance in {:2f} trials'.format(acc))


class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.lstm(x)
        x = self.linear(out)
        return x

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_all(1001)
    # 1. 创建数据集
    task = 'PerceptualDecisionMaking-v0'
    seq_len = 100
    batch_size = 32
    kwargs = {'dt': 100}
    dataset, ob_size, ac_size = create_dataset(task, seq_len, batch_size, kwargs)

    # 2. 创建模型
    input_size = ob_size
    hidden_size = 50
    output_size = ac_size
    layer_num = 1
    tau = 0.8
    v_th = 1.0
    gamma = 1.0
    radius = 1.0
    reg_lambda = 0.1
    reg_type = 'Euc_space'

    model = SNN(input_size, hidden_size, output_size, layer_num, tau, v_th, gamma).to(device)
    # model = ANN(input_size, hidden_size, output_size).to(device)

    # 3. 创建优化器
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. 创建损失函数
    criterion = nn.CrossEntropyLoss()

    distance_matrix = sphere_distance(hidden_size, radius)

    reg_names = []
    distances = {}
    for name, param in model.named_parameters():
        if 'recurrent_weight' in name:
            reg_names.append(name)
            distances[name] = distance_matrix

    criterion = SpatiallyLoss(criterion, model, reg_names, distances, reg_type, reg_lambda)

    # # 5. 训练
    epochs = 1000
    train(model, dataset, epochs, optimizer, criterion, device)

    # 6. 测试
    env = dataset.env
    num_trials = 10000
    performance_analysis(model, env, num_trials, device)

    # 7. 记录脉冲频率
    total_neuron_num = 0
    fire_num = 0
    for m in model.modules():
        if isinstance(m, LIFNeuron):
            total_neuron_num += m.total_neuron_num
            fire_num += m.fire_neuron_num
    fire_rate = fire_num / (total_neuron_num + 1e-6)
    print('fire_rate: {:.2f}'.format(fire_rate))

