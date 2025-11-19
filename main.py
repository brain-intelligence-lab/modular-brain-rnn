import torch
import torch.nn.functional as F
from functions.utils.math_utils import lock_random_seed
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import datasets.multitask as task
from models.recurrent_models import RNN, GRU, LSTM
from functions.utils.eval_utils import do_eval, \
    do_eval_with_dataset_torch_fast
from functions.utils.math_utils import generate_adj_matrix
import numpy as np
from collections import Counter
import bct
import argparse
import pdb

def start_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_rnn', default=84, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--display_step', default=500, type=int)
    parser.add_argument('--max_trials', default=3e6, type=int)
    parser.add_argument('--max_steps_per_stage', default=500, type=int)
    parser.add_argument('--add_conn_per_stage', default=0, type=int)
    parser.add_argument('--rec_scale_factor', default=0.5, type=float)
    parser.add_argument('--load_model_path', type=str)
    parser.add_argument('--conn_num', type=int, default=-1)
    parser.add_argument('--conn_mode',choices=['fixed', 'grow', 'full'], default='full')
    parser.add_argument('--rnn_type', choices=['RNN', 'GRU', 'LSTM'], default='RNN')
    parser.add_argument('--wiring_rule', choices=['distance', 'random'], default='distance')
    parser.add_argument('--eta', type=float, default=-3)
    parser.add_argument('--loss_type', choices=['lsq', 'ce'], default='lsq')
    parser.add_argument('--rule_set', choices=['all', 'mante', 'oicdmc'], default='all')
    parser.add_argument('--non_linearity', choices=['tanh', 'softplus', 'relu', 'leakyrelu'], default='relu')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--continual_learning', action='store_true')
    parser.add_argument('--task_num', default=20, type=int)
    parser.add_argument('--init_mode', choices=['randortho', 'diag', 'one_init'], default='randortho')
    parser.add_argument('--task_list', nargs='+', help='A list of tasks', default=None)
    parser.add_argument('--mask_type', choices=['prior_modular', 'random', 'posteriori_modular'], default='random')
    parser.add_argument('--eval_perf', action='store_true')
    parser.add_argument('--easy_task', action='store_true')
    parser.add_argument('--eval_verbose', action='store_true')
    parser.add_argument('--mod_lottery_hypo', action='store_true')
    parser.add_argument('--reg_term', action='store_true')
    parser.add_argument('--read_from_file', action='store_true')
    parser.add_argument('--get_chance_level', action='store_true')
    parser.add_argument('--gen_dataset_files', action='store_true')
    args = parser.parse_args()
    return args


def gen_hp(args):
    hp = task.get_default_hp(args.rule_set)
    hp['w_rec_init'] = args.init_mode
    
    if args.task_list is None:
        hp['rule_trains'] = task.rules_dict[args.rule_set]
        hp['rule_trains'] = hp['rule_trains'][:args.task_num]
        hp['rules'] = hp['rule_trains']
    else:
        assert set(args.task_list) <= set(task.rules_dict[args.rule_set]), "Invalid task_list!"
        hp['rule_trains'] = args.task_list
        hp['rules'] = hp['rule_trains']
    
    hp['reg_term'] = args.reg_term
    hp['wiring_rule'] = args.wiring_rule
    hp['conn_num'] = args.conn_num
    hp['n_rnn'] = args.n_rnn
    hp['learning_rate'] = 1e-3
    hp['seed'] = args.seed
    hp['rng'] = np.random.RandomState(args.seed)
    hp['activation'] = args.non_linearity
    hp['loss_type'] = args.loss_type
    hp['device'] = args.device

    if hp['n_rnn'] == 84 and hp['reg_term']:
        Distance = np.load('/modular-brain-rnn/datasets/brain_hcp_data/84/Raw_dis.npy')
        Distance = torch.from_numpy(Distance).to(args.device)
        min_val = Distance.min()
        max_val = Distance.max()
        hp['Distance'] = ((Distance - min_val) / (max_val - min_val)).float()
    
    return hp 

# https://github.com/gyyang/multitask/blob/master/train.py
def _run_training_loop(args, hp, model, writer: SummaryWriter):
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
    device = args.device
    log_dir = writer.logdir
    conn_mode = args.conn_mode 
    if conn_mode in ['fixed', 'grow']:
        model.gen_conn_matrix(wiring_rule=args.wiring_rule, eta=args.eta)
        if conn_mode == 'fixed':
            model.fix_connections()
        else:
            model.empty_connections()
        
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array(
                [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob / np.sum(rule_prob))

    step = 0
    loss_list = []

    batch_size = hp['batch_size_train']

    NUM_OF_BATCHES = int(args.max_trials // hp['batch_size_train'])
    if args.read_from_file:
        train_dataset = task.Multitask_Batched(hp, batch_size * NUM_OF_BATCHES, \
            batch_size, data_dir = './datasets/multitask/train')
    else:
        train_dataset = task.Multitask_Batches_Realtime_Gen(hp, NUM_OF_BATCHES, batch_size)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, \
        batch_size = None, num_workers = 4)
    
    if args.eval_perf and args.read_from_file:
        test_set = task.Get_Testset(hp, \
            data_dir='./datasets/multitask/test', n_rep=32, batch_size=16)
        
        big_batch_test_data = \
            task.preprocess_dataset_for_gpu_global(test_set, hp['rules'], device)

    while step * hp['batch_size_train'] <= args.max_trials:
        # for input, target, c_mask, task_name in data_list:
        for input, target, c_mask in train_loader:
            if step * hp['batch_size_train'] > args.max_trials:
                break
    
            input = input.to(device)
            target = target.to(device)
            c_mask = c_mask.to(device)

            output = model(input)
            if hp['loss_type'] == 'lsq':
                output = torch.sigmoid(output)
                loss = torch.mean(torch.square((target - output) * c_mask))
            else:
                _, _, n_output = target.shape
                target = target.view(-1, n_output)
                output = output.view(-1, n_output)
            
                loss = F.cross_entropy(output, target, reduction='none')      
                loss = torch.mean(loss * c_mask)
                
            if args.reg_term:
                loss += model.comm_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            step += 1
            if conn_mode == 'grow' and step % args.max_steps_per_stage == 0:
                model.grow_connections(args.add_conn_per_stage)
                optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])

            if step % args.display_step == 0:
                num_trials = step * hp['batch_size_train']

                weight = model.get_layer_to_analyze()
                ci, sc_qvalue = bct.modularity_dir(np.abs(weight))
                cluster_sizes = Counter(ci)
                average_cluster_size = sum(cluster_sizes.values()) / ci.max()
                writer.add_scalar(tag = 'Avg_Cluster_Size', scalar_value = average_cluster_size, global_step = step)
                writer.add_scalar(tag = 'Cluster_Num', scalar_value = ci.max(), global_step = step)
                writer.add_scalar(tag = 'SC_Qvalue', scalar_value = sc_qvalue, global_step = step)
                
                weight = torch.from_numpy(weight).to(device) 
                _, s, _ = torch.linalg.svd(weight)
                eff_dim = (torch.sum(s)**2) / torch.sum(s**2)
                writer.add_scalar(tag = 'eff_dim', scalar_value = eff_dim, global_step = step)

                loss_sum = sum(loss_list)
                writer.add_scalar(tag = 'Loss', scalar_value = loss_sum, global_step = step)
            
                print(f'Trials [{num_trials}/{args.max_trials}], Loss: {loss_sum:.3f}, \
                    SC_Qvalue: {sc_qvalue:.3f}')
                

                if args.eval_perf:
                    if args.read_from_file:
                        log = do_eval_with_dataset_torch_fast(model, \
                            hp['rules'], big_batch_test_data, verbose=args.eval_verbose)
                    else:
                        log = do_eval(model, rule_train=hp['rule_trains'], verbose=args.eval_verbose)

                    writer.add_scalar(tag = 'perf_avg', scalar_value = log['perf_avg'][-1], global_step = step)
                    writer.add_scalar(tag = 'perf_min', scalar_value = log['perf_min'][-1], global_step = step)
                    
                    for list_name, perf_list in log.items():
                        if not 'min' in list_name and not 'avg' in list_name:
                            writer.add_scalar(tag=list_name, scalar_value=perf_list[-1], global_step=step)
                
                if args.save_model:
                    torch.save(model, os.path.join(log_dir, f'RNN_interleaved_learning_{step}.pth'))
                
                loss_list = []

    # handle.remove()
    print("Optimization finished!")


def train(args, writer:SummaryWriter):
    device = args.device
    hp = gen_hp(args)
    if args.rnn_type == 'RNN':
        model = RNN(hp=hp, device=device, rec_scale_factor=args.rec_scale_factor).to(device)
    elif args.rnn_type == 'GRU':
        model = GRU(hp=hp, device=device, rec_scale_factor=args.rec_scale_factor).to(device)
    elif args.rnn_type == 'LSTM':
        model = LSTM(hp=hp, device=device, rec_scale_factor=args.rec_scale_factor).to(device)
    else:
        raise NotImplementedError
    _run_training_loop(args, hp, model, writer)


def get_chance_level(args, writer:SummaryWriter):
    device = args.device
    hp = gen_hp(args)
    model = RNN(hp=hp, device=device, rec_scale_factor=args.rec_scale_factor).to(device)
            
    step = args.display_step
    while step * hp['batch_size_train'] <= args.max_trials:
        if step % args.display_step == 0:
            weight = model.recurrent_conn.weight.data.detach().cpu().numpy()
            ci, sc_qvalue = bct.modularity_dir(np.abs(weight))
            
            log = do_eval(model, rule_train=hp['rule_trains'])
            writer.add_scalar(tag = 'SC_Qvalue', scalar_value = sc_qvalue, global_step = step)
            writer.add_scalar(tag = 'perf_avg', scalar_value = log['perf_avg'][-1], global_step = step)
            writer.add_scalar(tag = 'perf_min', scalar_value = log['perf_min'][-1], global_step = step)
            
            for list_name, perf_list in log.items():
                if not 'min' in list_name and not 'avg' in list_name:
                    writer.add_scalar(tag = list_name , scalar_value = perf_list[-1], global_step = step)
                    
        step += args.display_step
        
    print("Chance level evaluation finished!")


def module_lottery_ticket_hypo(args, writer):
    device = args.device
    hp = gen_hp(args)

    if args.rnn_type == 'RNN':
        init_model = RNN(hp=hp, device=device, rec_scale_factor=args.rec_scale_factor).to(device)
    elif args.rnn_type == 'GRU':
        init_model = GRU(hp=hp, device=device, rec_scale_factor=args.rec_scale_factor).to(device)
    elif args.rnn_type == 'LSTM':
        init_model = LSTM(hp=hp, device=device, rec_scale_factor=args.rec_scale_factor).to(device)
    else:
        raise NotImplementedError

    n_rnn = args.n_rnn
    max_modular_step = -1
    max_qvalue = -1

    load_model_path = args.load_model_path
    
    mask = None
    for load_step in range(500, 47000, 500):
        # trained_model = torch.load(f'runs/Fig2bcde_data/n_rnn_{n_rnn}_task_{task_num}_seed_{original_seed}/RNN_interleaved_learning_{load_step}.pth', device)
        path = os.path.join(load_model_path, f"RNN_interleaved_learning_{load_step}.pth")
        trained_model = torch.load(path, device) 
        
        weight = trained_model.recurrent_conn.weight.data.detach().cpu().numpy()
        abs_weight = np.abs(weight)
        ci, sc_qvalue = bct.modularity_dir(abs_weight)
        
        if sc_qvalue > max_qvalue:
            max_qvalue = sc_qvalue
            max_modular_step = load_step
            
            tmp_mask = np.zeros((n_rnn, n_rnn), dtype=np.float32)
            for i in range(n_rnn):
                for j in range(n_rnn):
                    if ci[i] == ci[j]: 
                        tmp_mask[i, j] = 1.0
                    
            # 针对 tmp_mask = 1 的部分，取前 100%
            mask_1_indices = np.where(tmp_mask == 1)
            mask_1_values = abs_weight[mask_1_indices]
            num_elements_1 = int(np.ceil(mask_1_values.size))
            top_1_indices_sorted = np.argpartition(-mask_1_values, num_elements_1 - 1)[:num_elements_1]

            # 针对 tmp_mask = 0 的部分，取前 5%
            mask_0_indices = np.where(tmp_mask == 0)
            mask_0_values = abs_weight[mask_0_indices]
            num_elements_0 = int(np.ceil(0.05 * mask_0_values.size))
            top_0_indices_sorted = np.argpartition(-mask_0_values, num_elements_0 - 1)[:num_elements_0]

            mask = np.zeros((n_rnn, n_rnn), dtype=bool)
            # 合并两部分的索引到 mask 中
            for idx in top_1_indices_sorted:
                mask[mask_1_indices[0][idx], mask_1_indices[1][idx]] = True
            for idx in top_0_indices_sorted:
                mask[mask_0_indices[0][idx], mask_0_indices[1][idx]] = True

    print(f'step:{max_modular_step} max_qvalue: {max_qvalue}')

    lottery_mask = np.zeros((n_rnn, n_rnn), dtype=np.float32)
    for i in range(n_rnn):
        for j in range(n_rnn):
            if mask[i, j] == True:
                lottery_mask[i, j] = 1.0

    if args.mask_type == 'posteriori_modular':

        init_model.set_mask(lottery_mask)
        assert(lottery_mask.sum() < n_rnn * n_rnn)

    elif args.mask_type == 'prior_modular':
        
        num_ones = int(np.sum(lottery_mask))
        print(f'num_ones:{num_ones}')
        core_list = [ n_rnn // 4 for _ in range(2)]
        periphery_size = n_rnn - np.sum(core_list)
        adj_matrix = generate_adj_matrix(core_sizes=core_list, periphery_size=periphery_size, \
            total_connections=num_ones, seed=args.seed)
        
        init_model.set_mask(adj_matrix)
    
    elif args.mask_type == 'random':
        
        num_ones = np.sum(lottery_mask) 
        random_mask = np.zeros((n_rnn, n_rnn), dtype=np.float32)
        indices = [(i, j) for i in range(n_rnn) for j in range(n_rnn)]
        chosen_indices = np.random.choice(len(indices), size=int(num_ones), replace=False)
        for idx in chosen_indices:
            random_mask[indices[idx]] = 1.0
    
        init_model.set_mask(random_mask)
        
    model = init_model
    _run_training_loop(args, hp, model, writer)

if __name__ == '__main__':
    args = start_parse()
    lock_random_seed(seed=args.seed)
    assert args.conn_mode =='full' or args.conn_num != -1
    
    if args.gen_dataset_files:
        hp = task.get_default_hp(args.rule_set)
        batch_size = hp['batch_size_train']
        NUM_OF_BATCHES = int(args.max_trials // batch_size)
        train_dataset = task.Multitask_Batched(hp, batch_size * NUM_OF_BATCHES, \
            batch_size, data_dir = './datasets/multitask/train')

        test_set = task.Get_Testset(hp, \
            data_dir='./datasets/multitask/test', n_rep=32, batch_size=16)
        exit(0)

    
    writer = SummaryWriter(log_dir=args.log_dir)
    log_dir = writer.logdir
    device = torch.device(f'cuda:{args.gpu}' if args.gpu>=0 else 'cpu')
    args.device = device

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args_file_path = os.path.join(log_dir, "args.txt")
    
    with open(args_file_path, 'w') as f:
        f.write(f'current_time: {current_time}\n')
        for arg, value in sorted(vars(args).items()):
            f.write(f'{arg}: {value}\n')
    
    if args.mod_lottery_hypo:
        module_lottery_ticket_hypo(args, writer=writer)

    elif args.get_chance_level:
        get_chance_level(args, writer=writer)
        
    else:
        train(args, writer=writer)    
    
    writer.close()
        