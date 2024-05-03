import torch
import datasets.multitask as task
from models.recurrent_models import RNN
from functions.utils.eval_utils import do_eval
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
import os
from tensorboardX import SummaryWriter
import bct

hidden_states = None

def readout_hook(module, input, output):
    global hidden_states
    hidden_states = input[0].clone()  # 使用 clone() 以保留当前状态

def gen_hp(args):
    hp = task.get_default_hp(args.rule_set)
    hp['w_rec_init'] = 'randortho'
    
    if args.continual_learning: 
        assert args.max_trials != 3000000, "too many trials for continual learning!"
        
        hp['easy_task'] = True
    else:
        # hp['w_rec_init'] = 'diag'
        hp['rule_trains'] = task.rules_dict[args.rule_set]
        hp['rule_trains'] = hp['rule_trains'][:args.task_num]
        hp['rules'] = hp['rule_trains']
    
    hp['wiring_rule'] = args.wiring_rule
    hp['conn_num'] = args.conn_num
    hp['n_rnn'] = args.n_rnn
    hp['learning_rate'] = 1e-3
    hp['seed'] = args.seed
    hp['rng'] = np.random.RandomState(args.seed)
    hp['activation'] = args.non_linearity
    hp['c_intsyn'] = args.reg_factor
    hp['loss_type'] = args.loss_type
    hp['ksi_intsyn'] = args.ksi

    return hp

# https://github.com/gyyang/multitask/blob/master/train.py
            
def train(args, writer:SummaryWriter):
    device = args.device
    hp = gen_hp(args)
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    
    model = RNN(hp=hp, device=device, rec_scale_factor=args.rec_scale_factor).to(device)
    handle = model.readout.register_forward_hook(readout_hook)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
    
    log_dir = writer.logdir
    conn_mode = args.conn_mode 
    
    if conn_mode in ['fix', 'grow']:
        model.gen_conn_matrix()
        if conn_mode == 'fix':
            model.fix_connections()
        else:
            model.empty_connections()
        
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
        
        if conn_mode == 'fix':
            model.gen_mask_for_control()
    
        optimizer.step()
        
        step += 1
        if conn_mode == 'grow' and step % args.max_steps_per_stage == 0:
            model.grow_connections(args.add_conn_per_stage)
        
        if conn_mode == 'fix' and step % args.max_steps_per_stage == 0:
            model.update_conn_num(args.add_conn_per_stage)
            
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
            
    handle.remove()
    print("Optimization finished!")


def get_current_param_list(model, clone=True):
    v_list = []
    for _, param in model.named_parameters():
        if clone:
            v_list.append(param.clone())
        else:
            v_list.append(param)
    return v_list

def train_sequential(args, writer:SummaryWriter, rule_trains=None):
    device = args.device
    hp = gen_hp(args)
    assert rule_trains is not None, "rule trains is empty!"
    log_dir = writer.logdir
    
    model = RNN(hp=hp, device=device, rec_scale_factor=args.rec_scale_factor).to(device)
    handle = model.readout.register_forward_hook(readout_hook)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])

    conn_mode = args.conn_mode 
    if conn_mode in ['fix', 'grow']:
        model.gen_conn_matrix()
        if conn_mode == 'fix':
            model.fix_connections()
        else:
            model.empty_connections()
    
    hp['rule_trains'] = rule_trains
    hp['rules'] = [r for rs in rule_trains for r in rs]

    # Number of training iterations for each rule
    rule_train_iters = [len(r) * args.max_trials for r in rule_trains]

    # Using continual learning or not
    c, ksi = hp['c_intsyn'], hp['ksi_intsyn']

    loss_list = []
    global_step = 0
    
    for i_rule_train, rule_train in enumerate(hp['rule_trains']):
        step = 0
        
        v_current = get_current_param_list(model)

        if i_rule_train == 0:
            v_anc0 = v_current
            Omega0 = [torch.zeros_like(v) for v in v_anc0]
            omega0 = [torch.zeros_like(v) for v in v_anc0]
            v_delta = [torch.zeros_like(v) for v in v_anc0]
        else:
            v_anc0_prev = v_anc0
            v_anc0 = v_current
            v_delta = [v-v_prev for v, v_prev in zip(v_anc0, v_anc0_prev)]

            # Make sure all elements in omega0 are non-negative
            # Penalty
            Omega0 = [F.relu(O + o / (v_d ** 2 + ksi))
                        for O, o, v_d in zip(Omega0, omega0, v_delta)]
            
        # Reset
        omega0 = [torch.zeros_like(v) for v in v_anc0]

        fc_list = []
        while (step * hp['batch_size_train'] <=
                   rule_train_iters[i_rule_train]):
    
            # Training
            rule_train_now = hp['rng'].choice(rule_train)
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

            # Continual learning with intelligent synapses
            v_prev = v_current

            output = model(input)

            if hp['loss_type'] == 'lsq' and not hp['use_snn']:
                output = torch.sigmoid(output)
            
            global hidden_states
            random_value = torch.rand_like(hidden_states) 
            hidden_states =  hidden_states + random_value * 1e-7
            hidden_states_mean = hidden_states.detach().mean(1).cpu().numpy()
            fc = np.corrcoef(hidden_states_mean, rowvar=False)
            fc_list.append(fc)


            # Update cost 
            cost_reg = 0.
            now_v = get_current_param_list(model, clone=False)
            
            for v, w, v_val in zip(now_v, Omega0, v_anc0):
                cost_reg += c * torch.sum(
                    w.detach() * (v - v_val.detach())**2)

            loss = torch.mean(torch.square((target - output) * c_mask)) + cost_reg
            loss_list.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()

            # get the grad
            v_grad = []
            for _, param in model.named_parameters():
                v_grad.append(param.grad)

            optimizer.step()

            v_current = get_current_param_list(model)

            # Update synaptic importance
            omega0 = [
                o - (v_c - v_p) * v_g for o, v_c, v_p, v_g in
                zip(omega0, v_current, v_prev, v_grad)
            ]

            step += 1
            global_step +=1
            if conn_mode == 'grow' and global_step % args.max_steps_per_stage == 0:
                model.grow_connections(args.add_conn_per_stage)
    
            if step % args.display_step == 0:
                num_trails = step * hp['batch_size_train']
                print(f'Trials [{num_trails}/{rule_train_iters[i_rule_train]}], Loss: {sum(loss_list):.4f}')

                weight = model.recurrent_conn.weight.data.detach().cpu().numpy()
                ci, sc_qvalue = bct.modularity_dir(np.abs(weight))

                writer.add_scalar(tag = 'Cluster_Num', scalar_value = ci.max(), global_step = global_step)

                writer.add_scalar(tag = 'SC_Qvalue', scalar_value = sc_qvalue, global_step = global_step)

                fc = np.mean(fc_list, 0) 
                fc [fc < 0] = 0
                _, fc_qvalue = bct.modularity_dir(fc)
                writer.add_scalar(tag = 'FC_Qvalue', scalar_value = fc_qvalue, global_step = global_step)

                loss_sum = sum(loss_list)
                writer.add_scalar(tag = 'Loss', scalar_value = loss_sum, global_step = global_step)
                
                log = do_eval(model, rule_train=rule_train)
                
                for list_name, perf_list in log.items():
                    if not 'min' in list_name and not 'avg' in list_name:
                        writer.add_scalar(tag=list_name, scalar_value=perf_list[-1], global_step=global_step)
                
                if args.save_model:
                    torch.save(model, os.path.join(log_dir, f'RNN_continual_learning_{step}.pth'))
                loss_list = []
                fc_list = []
                
    handle.remove()
    print("Optimization finished!")
