import numpy as np
import bct
import torch
import torch.nn as nn
from models.recurrent_models import RNN
from functions.utils.eval_utils import lock_random_seed, do_eval
from functions.generative_network_modelling.generative_network_modelling import Gen_one_connection
from collections import defaultdict
import datasets.multitask as task
from torch.utils.tensorboard import SummaryWriter
import pdb

def sub_graph_weight(weight, mask):
    # 找出所有被选中的边
    selected_edges = np.where(mask == 1)

    # 获取所有被选中的边的端点（即行和列索引）
    selected_nodes = np.unique(np.hstack((selected_edges[0], selected_edges[1])))

    # 创建一个新的子矩阵，初始值为无穷大（或者任意一个表示无边的值）
    weight1 = np.full((len(selected_nodes), len(selected_nodes)), 1e-3)

    # 节点到新矩阵索引的映射
    node_to_new_index = {node: index for index, node in enumerate(selected_nodes)}

    # 使用被选中的节点填充新矩阵
    for edge in zip(*selected_edges):
        new_row = node_to_new_index[edge[0]]
        new_col = node_to_new_index[edge[1]]
        weight1[new_row, new_col] = weight[edge[0], edge[1]]
        
    return weight1
    

def train(model, max_trials, writer:SummaryWriter, iter_idx):
    device = next(model.parameters()).device
    step = 0
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []
    log = defaultdict(list)

    fc_list = []
    while step * hp['batch_size_train'] <= max_trials:
        # Training
        rule_train_now = hp['rng'].choice(hp['rule_trains'],
                                            p=hp['rule_probs'])
        # Generate a random batch of trials.
        # Each batch has the same trial length
        trial = task.generate_trials(
                rule_train_now, hp, 'random',
                batch_size=hp['batch_size_train'])
        
        input = torch.from_numpy(trial.x).to(device)
        target = torch.from_numpy(trial.y).to(device)

        output, hidden_states = model(input)
        if hp['loss_type'] == 'lsq' and not hp['use_snn']:
            output = torch.sigmoid(output)
            
        hidden_states_mean = hidden_states.detach().mean(1).cpu().numpy()
        fc = np.corrcoef(hidden_states_mean, rowvar=False)
        fc_list.append(fc)

        loss = criterion(output, target)
        loss_list.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        if step % 500 == 0:
            global_step = iter_idx * (max_trials / hp['batch_size_train']) +  step

            num_steps = step * hp['batch_size_train']
            weight = model.recurrent_conn.weight.data.detach().cpu().numpy()

            ci, sc_qvalue = bct.modularity_dir(np.abs(weight))

            writer.add_scalar(tag = 'Cluster_Num', scalar_value = ci.max(), global_step = global_step)

            writer.add_scalar(tag = 'SC_Qvalue', scalar_value = sc_qvalue, global_step = global_step)
            
            # sc_qvalue = -1
            # if model.mask.sum() > 0:

            #     weight1 = sub_graph_weight(weight=weight, mask=model.mask.cpu().numpy())
                
            #     _, sc_qvalue = bct.modularity_dir(np.abs(weight1))
            
            fc = np.mean(fc_list, 0)            
            
            fc [fc < 0] = 0
                    
            ci, fc_qvalue = bct.modularity_dir(fc)

            writer.add_scalar(tag = 'FC_Qvalue', scalar_value = fc_qvalue, global_step = global_step)

            loss_sum = sum(loss_list)
            writer.add_scalar(tag = 'Loss', scalar_value = loss_sum, global_step = global_step)
            
            print(f'Step [{num_steps}/{max_trials}], Loss: {loss_sum:.3f}, \
                SC_Qvalue: {sc_qvalue:.3f}, FC_Qvalue: {fc_qvalue:.3f}')
            
            log = do_eval(model, log, rule_train=hp['rule_trains'])

            writer.add_scalar(tag = 'perf_avg', scalar_value = log['perf_avg'][-1], global_step = global_step)
            writer.add_scalar(tag = 'perf_min', scalar_value = log['perf_min'][-1], global_step = global_step)

            loss_list = []
    # print("Optimization finished!")
    fc = np.mean(fc_list, 0) 
    fc [fc < 0] = 0
    return model, fc


def start_parse():
    import argparse
    parser = argparse.ArgumentParser(description='interactive_modelling')
    parser.add_argument('--total_conn_num', default=1500, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=2024)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--max_trials_per_stage', default=32000, type=int)
    parser.add_argument('--add_conn_per_stage', default=10, type=int)
    parser.add_argument('--rec_scale_factor', default=1.0, type=float)
    parser.add_argument('--use_distance', action='store_true')
    parser.add_argument('--use_fc', action='store_true')
    parser.add_argument('--task_num', default=20, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = start_parse()
    assert args.exp_name is not None, "You must specify exp_name."

    device = torch.device(f'cuda:{args.gpu}' if args.gpu>=0 else 'cpu')
    lock_random_seed(args.seed)
    writer = SummaryWriter(comment=args.exp_name)
 
    ruleset = 'all'

    hp = {'activation': 'softplus', 'use_snn':False}

    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}

    default_hp = task.get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)

    hp = default_hp
    hp['seed'] = args.seed
    hp['rng'] = np.random.RandomState(args.seed)
    hp['rule_trains'] = task.rules_dict[ruleset]

    hp['rule_trains'] = hp['rule_trains'][:args.task_num]
    # hp['rule_trains'] = ['dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm']

    hp['rules'] = hp['rule_trains']

    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()
    
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array(
                [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob / np.sum(rule_prob))

    # 创建模型
    model = RNN(hp=hp, device=device, rec_scale_factor=args.rec_scale_factor).to(device)

    # Wtgt = scipy.io.loadmat('/data_nv/dataset/brain_hcp_data/84/structureM_use.mat')['structureM_use'].astype(np.int16)

    # coor = np.load('/data_nv/dataset/brain_hcp_data/84/mean_coordinate.npy')
    
    if args.use_distance:
        Dis = np.load('/data_nv/dataset/brain_hcp_data/84/Raw_dis.npy')
    else:
        Dis = None
    
    # A = np.zeros_like(Wtgt[:,:,0])

    # set model parameters
    typemodeltype = 'matching'
    modelvar = ['powerlaw', 'powerlaw']

    eta = -3.2
    gamma = 0.38
    params = [eta, gamma, 1e-5]
    m = args.total_conn_num
    conn_matrices = np.zeros((m, hp['n_rnn'], hp['n_rnn']))

    # Atgt = np.array(Wtgt > 0, dtype=float)
   
    # interactive 布线-训练

    mask = np.random.randint(2, size=(hp['n_rnn'], hp['n_rnn']))
    mask = np.zeros_like(mask)



    # mask = np.ones_like(mask)
    # modules = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 83)]
    
    # # 为每个模块内的节点创建随机连接
    # for start, end in modules:
    #     for i in range(start, end + 1):
    #         for j in range(start, end + 1):
    #                 mask[i, j] = 1   
    
    iter_idx = 0
    conn_num = 0    
    while conn_num < m:
        add_conn = 0

        if not args.use_fc:
            fc = None

        while iter_idx > 0 and (add_conn < args.add_conn_per_stage and (conn_num + add_conn) < m):

            mask = Gen_one_connection(mask, params, modelvar, D=Dis, Fc=fc, device=device)
            add_conn += 1
        
        conn_num += add_conn
        model.set_mask(torch.tensor(mask))
        model, fc = train(model=model, max_trials=args.max_trials_per_stage, writer=writer, iter_idx=iter_idx)
        iter_idx += 1
        torch.save(model, f'./runs/rnn_{args.exp_name}.pth')
        print(f'now num of recurrent connections:{conn_num}')
        
        # if conn_num > 100:
        #     mask = np.ones_like(mask)

    writer.close()
        
