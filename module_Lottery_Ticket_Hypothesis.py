import argparse
import torch
from functions.utils.eval_utils import lock_random_seed
from tensorboardX import SummaryWriter
from multitask_train import gen_hp, train
from models.recurrent_models import RNN
from functions.utils.eval_utils import do_eval
import torch.nn.functional as F
from main import start_parse
from datetime import datetime
import datasets.multitask as task
import bct
import os
import numpy as np
import pdb

def readout_hook(module, input, output):
    global hidden_states
    hidden_states = input[0].clone()  # 使用 clone() 以保留当前状态
    

if __name__ == '__main__':
    args = start_parse()
    original_seed=100
    lock_random_seed(original_seed)
    device = torch.device(f'cuda:{args.gpu}' if args.gpu>=0 else 'cpu')
    args.device = device
    
    n_rnn = args.n_rnn
    task_num = args.task_num
    
    hp = gen_hp(args)
    original_model = RNN(hp=hp, device=device, rec_scale_factor=args.rec_scale_factor).to(device)
    
    max_modular_step = -1
    max_qvalue = -1
    max_modular_ci = None
    
    for load_step in range(500, 47000, 500):
        
        trained_model = torch.load(f'runs/Fig2bcde_data/n_rnn_{n_rnn}_task_{task_num}_seed_{original_seed}/RNN_interleaved_learning_{load_step}.pth', device) 
        
        weight = trained_model.recurrent_conn.weight.data.detach().cpu().numpy()
        abs_weight = np.abs(weight)
        ci, sc_qvalue = bct.modularity_dir(np.abs(abs_weight))
        
        if sc_qvalue > max_qvalue:
            max_qvalue = sc_qvalue
            max_modular_step = load_step
            max_modular_ci = ci
            
            tmp_mask = np.zeros((n_rnn, n_rnn), dtype=np.float32)
            for i in range(n_rnn):
                for j in range(n_rnn):
                    if ci[i] == ci[j]: 
                        tmp_mask[i, j] = 1.0
                    
            # print(tmp_mask.sum())            

            # 针对 tmp_mask = 1 的部分，取前 100%
            mask_1_indices = np.where(tmp_mask == 1)
            mask_1_values = abs_weight[mask_1_indices]
            num_elements_1 = int(np.ceil(mask_1_values.size))
            top_1_indices_sorted = np.argpartition(-mask_1_values, num_elements_1 - 1)[:num_elements_1]

            # 针对 tmp_mask = 0 的部分，取前 15%
            mask_0_indices = np.where(tmp_mask == 0)
            mask_0_values = abs_weight[mask_0_indices]
            num_elements_0 = int(np.ceil(0.15 * mask_0_values.size))
            top_0_indices_sorted = np.argpartition(-mask_0_values, num_elements_0 - 1)[:num_elements_0]

            
            mask = np.zeros((n_rnn, n_rnn), dtype=bool)
            # 合并两部分的索引到 mask 中
            for idx in top_1_indices_sorted:
                mask[mask_1_indices[0][idx], mask_1_indices[1][idx]] = True
            for idx in top_0_indices_sorted:
                mask[mask_0_indices[0][idx], mask_0_indices[1][idx]] = True

            # abs_weight[~mask] = 0.0
            # ci, sc_qvalue = bct.modularity_dir(np.abs(abs_weight))
            # print(f'masked_sc_qvalue:{sc_qvalue}, step:{load_step}')
            
            # community_in = 0
            # community_out = 0
            
            # for i in range(n_rnn):
            #     for j in range(n_rnn):
            #         if mask[i,j] == True:
            #             if ci[i] == ci[j]:
            #                 community_in += 1
            #             else:
            #                 community_out += 1
            
            # print(f'community_in:{community_in}, community_out:{community_out}')
            # num_true = int(mask.sum())
            # print(f'num_true:{num_true}')
            # qvalue_list = []

            # for _ in range(100):
            #     random_mask = np.zeros_like(mask.flatten(), dtype=bool)
            #     random_indices = np.random.choice(mask.size, size=num_true, replace=False)
            #     random_mask[random_indices] = True
            #     random_mask = random_mask.reshape(weight.shape)
            #     abs_weight = np.abs(weight)
            #     abs_weight[~random_mask] = 0.0
            #     ci, sc_qvalue = bct.modularity_dir(np.abs(abs_weight))
            #     qvalue_list.append(sc_qvalue)
            # print(f'qvalue_mean:{np.mean(qvalue_list)}, var:{np.var(qvalue_list)}')   
    
    
    print(f'step:{max_modular_step} max_qvalue: {max_qvalue}')

    lottery_mask = np.zeros((n_rnn, n_rnn), dtype=np.float32)
    for i in range(n_rnn):
        for j in range(n_rnn):
            if mask[i, j] == True:
                lottery_mask[i, j] = 1.0

    
    if args.seed == original_seed:
        original_model.set_mask(lottery_mask)
    else:
        lock_random_seed(args.seed)
        
        num_ones = np.sum(lottery_mask) 
        random_mask = np.zeros((n_rnn, n_rnn), dtype=np.float32)
        
        indices = [(i, j) for i in range(n_rnn) for j in range(n_rnn)]
        
        chosen_indices = np.random.choice(len(indices), size=int(num_ones), replace=False)
        for idx in chosen_indices:
            random_mask[indices[idx]] = 1.0
    
        original_model.set_mask(random_mask)
        masked_weight = original_model.recurrent_conn.weight.data.detach().cpu().numpy() * random_mask
        print(bct.modularity_dir(np.abs(masked_weight))[1])
        
    model = original_model

    # weight = model.recurrent_conn.weight.data.detach().cpu().numpy()
    # ci, sc_qvalue = bct.modularity_dir(np.abs(weight))
    # print(f'init_qvalue: {sc_qvalue}')

    handle = model.readout.register_forward_hook(readout_hook)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    
    writer = SummaryWriter(log_dir=args.log_dir)
    log_dir = writer.logdir
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args_file_path = os.path.join(log_dir, "args.txt")

    with open(args_file_path, 'w') as f:
        f.write(f'current_time: {current_time}\n')
        for arg, value in sorted(vars(args).items()):
            f.write(f'{arg}: {value}\n')
    
    log_dir = writer.logdir
    
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array(
                [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob / np.sum(rule_prob))

    step = 0
    loss_list = []

    fc_list = []
    while step * hp['batch_size_train'] <= args.max_trials:
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
        c_mask = torch.from_numpy(trial.c_mask).to(device)

        if torch.numel(c_mask) == torch.numel(target):
            c_mask = c_mask.reshape(target.shape[0], target.shape[1], -1)

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

        global hidden_states

        random_value = torch.rand_like(hidden_states) 
        hidden_states =  hidden_states + random_value * 1e-7
        hidden_states_mean = hidden_states.detach().mean(1).cpu().numpy()
        fc = np.corrcoef(hidden_states_mean, rowvar=False)
        fc_list.append(fc)

        loss_list.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
            
        if step % args.display_step == 0:
            num_trials = step * hp['batch_size_train']
            
            weight = model.recurrent_conn.weight.data.detach().cpu().numpy()
            ci, sc_qvalue = bct.modularity_dir(np.abs(weight))

            writer.add_scalar(tag = 'Cluster_Num', scalar_value = ci.max(), global_step = step)
            
            writer.add_scalar(tag = 'SC_Qvalue', scalar_value = sc_qvalue, global_step = step)
            
            fc = np.mean(fc_list, 0)
            fc[ fc < 0 ] = 0
            _, fc_qvalue = bct.modularity_dir(fc)
            writer.add_scalar(tag = 'FC_Qvalue', scalar_value = fc_qvalue, global_step = step)
            
            loss_sum = sum(loss_list)
            writer.add_scalar(tag = 'Loss', scalar_value = loss_sum, global_step = step)
            
            print(f'Trials [{num_trials}/{args.max_trials}], Loss: {loss_sum:.3f}, \
                SC_Qvalue: {sc_qvalue:.3f}, FC_Qvalue: {fc_qvalue:.3f}')
            
            log = do_eval(model, rule_train=hp['rule_trains'])
            
            writer.add_scalar(tag = 'perf_avg', scalar_value = log['perf_avg'][-1], global_step = step)
            writer.add_scalar(tag = 'perf_min', scalar_value = log['perf_min'][-1], global_step = step)
            
            for list_name, perf_list in log.items():
                if not 'min' in list_name and not 'avg' in list_name:
                    writer.add_scalar(tag=list_name, scalar_value=perf_list[-1], global_step=step)
            
            if args.save_model:
                torch.save(model, os.path.join(log_dir, f'RNN_interleaved_learning_{step}.pth'))
            
            loss_list = []
            fc_list = []
            
    print("Optimization finished!")
    
    handle.remove()
    writer.close()
        