import numpy as np 
import matplotlib.pyplot as plt
from functions.utils.eval_utils import do_eval
import datasets.multitask as task
import os
import pdb
import bct
import torch
import time
import networkx as nx


def fibonacci_sphere(samples=1, randomize=True):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment
        if randomize:
            theta += np.random.uniform(0, 2*np.pi)  # randomize within each module

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))
    return points

def assign_module_positions(modules, n):
    # Create random but distinct center points for each module on a large sphere
    module_centers = fibonacci_sphere(len(set(modules)), randomize=False)
    positions = {}
    for i in range(n):
        module_index = modules[i] - 1
        center_x, center_y, center_z = module_centers[module_index]
        # Scale points around the center based on module
        scale = 0.1  # Adjust scale to manage intra-module density
        positions[i] = (center_x + np.random.normal(scale=scale), 
                        center_y + np.random.normal(scale=scale), 
                        center_z + np.random.normal(scale=scale))
    return positions



def visualize_rnn_weights_3d_modular(W, modules, save_name):
    G = nx.DiGraph()
    n = W.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if W[i, j] != 0:
                G.add_edge(i, j, weight=abs(W[i, j]))

    pos = assign_module_positions(modules, n)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Different color for each module
    colors = plt.cm.jet(np.linspace(0, 1, len(set(modules))))

    for key, value in pos.items():
        module_index = modules[key] - 1
        ax.scatter(*value, color=colors[module_index], s=100)

    for edge in G.edges(data=True):
        start, end, weight = edge
        xs, ys, zs = zip(pos[start], pos[end])
        ax.plot(xs, ys, zs, 'gray', linewidth=weight['weight'])

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlim([-1, 1])

    plt.title('3D Modular RNN Weight Visualization')
    plt.savefig(f'{save_name}.jpg', format='jpg')
    plt.savefig(f'{save_name}.svg', format='svg')


start_time = time.time()  # 获取当前时间

device = torch.device(f'cuda:{7}')

ruleset = 'all'

hp = {'activation': 'softplus', 'use_snn':False}

default_hp = task.get_default_hp(ruleset)
if hp is not None:
    default_hp.update(hp)

hp = default_hp
hp['seed'] = 2024
hp['rng'] = np.random.RandomState(hp['seed'])
hp['rule_trains'] = task.rules_dict[ruleset]
hp['rules'] = hp['rule_trains']


def load_and_test(model_path):
    perf_avg = -1
    model = torch.load(model_path, map_location=device)
    log = do_eval(model, rule_train=hp['rules'])
    perf_avg = log['perf_avg'][-1]    
    return model, perf_avg


seed_list = [ i for i in range(1, 101)]
n = 30
m = 80
task_num = 20
step = 30000

seed2model = {}

N = 30

for _, seed in enumerate(seed_list):
    if seed > N:
        break
    
    model_dir = f'./runs/Fig4_topology_task/{n}_{m}_{task_num}/'
    sub_dir = f'n_rnn_{n}_task_{task_num}_seed_{seed}_rule_random_mode_fix_conn_num_{m}/'
    file_name = f'RNN_interleaved_learning_{step}.pth'
    
    model_path = os.path.join(model_dir, sub_dir, file_name)
    model, perf = load_and_test(model_path=model_path)
    print(f'{perf:.4f}')

    seed2model[seed] = (perf, model)
    
end_time = time.time()  # 再次获取当前时间
elapsed_time = end_time - start_time  # 计算两次时间的差值，得到代码运行的时间
print(f"代码运行时间：{elapsed_time}秒")


sorted_model_perf = sorted(seed2model.items(), key=lambda item: item[1][0], reverse=True)

sorted_model_perf_mp = dict(sorted_model_perf)

# To print or use the sorted dictionary
for model, perf in sorted_model_perf_mp.items():
    print(model, perf)


end_time = time.time()  # 再次获取当前时间
elapsed_time = end_time - start_time  # 计算两次时间的差值，得到代码运行的时间
print(f"代码运行时间：{elapsed_time}秒")

_, model = sorted_model_perf[0][1]

recurrent_conn = model.recurrent_conn.weight.data.detach().cpu().numpy()

recurrent_conn = model.mask.cpu().numpy() * recurrent_conn

ci, sc_qvalue = bct.modularity_dir(np.abs(recurrent_conn))



visualize_rnn_weights_3d_modular(recurrent_conn, ci, f'top1_perf_W')


_, model = sorted_model_perf[-1][1]

recurrent_conn = model.recurrent_conn.weight.data.detach().cpu().numpy()

recurrent_conn = model.mask.cpu().numpy() * recurrent_conn

ci, sc_qvalue = bct.modularity_dir(np.abs(recurrent_conn))

visualize_rnn_weights_3d_modular(recurrent_conn, ci, f'top{N}_perf_W')

pdb.set_trace()
