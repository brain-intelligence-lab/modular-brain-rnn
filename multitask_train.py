import torch
import datasets.multitask as task
from models.recurrent_models import RNN
from functions.utils.eval_utils import do_eval, lock_random_seed
from functions.generative_network_modelling.generative_network_modelling import Gen_one_connection
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
import argparse
import pdb
from torch.utils.tensorboard import SummaryWriter
import bct


# https://github.com/gyyang/multitask/blob/master/train.py
            
def train(ruleset, device, exp_name, hp=None, max_trials=3e6, seed=2024, rule_prob_map=None, rule_trains=None, rec_scale_factor=1.0):
    writer = SummaryWriter(comment=exp_name)
    
    default_hp = task.get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)
    if hp['rule_trains'] is None:
        hp['rule_trains'] = task.rules_dict[ruleset]
        if rule_trains is not None:
            hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']

    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()
    
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array(
                [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob/np.sum(rule_prob))

    model = RNN(hp=hp, device=device, rec_scale_factor=rec_scale_factor).to(device)

    step = 0
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
        c_mask = torch.from_numpy(trial.c_mask).to(device)

        if torch.numel(c_mask) == torch.numel(target):
            c_mask = c_mask.reshape(target.shape[0], target.shape[1], -1)

        output, hidden_states = model(input)
        if hp['loss_type'] == 'lsq':
            output = torch.sigmoid(output)
            loss = torch.mean(torch.square((target - output) * c_mask))
        else:
            _, _, n_output = target.shape
            target = target.view(-1, n_output)
            output = output.view(-1, n_output)
           
            loss = F.cross_entropy(output, target, reduction='none')      
            loss = torch.mean(loss * c_mask)

        #TODO: 这里tmpsum会特别靠近0，是为什么
        # tmpsum = output[-1,:,1:].detach().cpu().numpy().sum(axis=-1).mean()
        # print(f'mean_tmpsum: {tmpsum:.4f}')
        
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
        if step % 500 == 0:
            num_steps = step * hp['batch_size_train']
            
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
            
            print(f'Step [{num_steps}/{max_trials}], Loss: {loss_sum:.3f}, \
                SC_Qvalue: {sc_qvalue:.3f}, FC_Qvalue: {fc_qvalue:.3f}')
            
            log = do_eval(model, log, rule_train=hp['rule_trains'])
            
            writer.add_scalar(tag = 'perf_avg', scalar_value = log['perf_avg'][-1], global_step = step)
            writer.add_scalar(tag = 'perf_min', scalar_value = log['perf_min'][-1], global_step = step)
            
            for list_name, perf_list in log.items():
                if not 'min' in list_name and not 'avg' in list_name:
                    writer.add_scalar(tag=list_name, scalar_value=perf_list[-1], global_step=step)
            
            # torch.save(model, f'./runs/rnn_interleaved.pth')
            
            loss_list = []
            fc_list = []

    print("Optimization finished!")



# https://github.com/gyyang/multitask/blob/master/train.py

def train_sequential(ruleset, device, rule_trains, exp_name, hp=None, max_trials=1e4, seed=2024, rec_scale_factor=1.0):
    '''Train the network sequentially.

    Args:
        rule_trains: a list of list of tasks to train sequentially
        max_trials: int, maximum number of training steps for each list of tasks
        ruleset: the set of rules to train
        seed: int, random seed to be used
    '''
    writer = SummaryWriter(comment=exp_name)
    
    default_hp = task.get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)
    hp['rule_trains'] = rule_trains
    hp['rules'] = [r for rs in rule_trains for r in rs]

    # Number of training iterations for each rule
    rule_train_iters = [len(r)*max_trials for r in rule_trains]

    model = RNN(hp=hp, device=device, rec_scale_factor=rec_scale_factor).to(device)

    def get_current_param_list(model, clone=True):
        v_list = []
        for _, param in model.named_parameters():
            if clone:
                v_list.append(param.clone())
            else:
                v_list.append(param)
        return v_list

    # Using continual learning or not
    c, ksi = hp['c_intsyn'], hp['ksi_intsyn']

    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
    loss_list = []
    log = defaultdict(list)
    global_step = 0
    
    # eta = -3.2
    # gamma = 0.38
    # params = [eta, gamma, 1e-5]
    # modelvar = ['powerlaw', 'powerlaw']
    
    # mask = np.zeros((hp['n_rnn'], hp['n_rnn']), dtype=np.float32)
    # now_total_num = 0
    # model.set_mask(torch.tensor(mask))

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

            output, hidden_states = model(input)

            if hp['loss_type'] == 'lsq' and not hp['use_snn']:
                output = torch.sigmoid(output)
            
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

            # loss = criterion(output, target) + cost_reg
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
            if step % 500 == 0:
                num_trails = step * hp['batch_size_train']
                print(f'Step [{num_trails}/{rule_train_iters[i_rule_train]}], Loss: {sum(loss_list):.4f}')

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
                
                log = do_eval(model, log, rule_train=rule_train)
                
                for list_name, perf_list in log.items():
                    if not 'min' in list_name and not 'avg' in list_name:
                        writer.add_scalar(tag=list_name, scalar_value=perf_list[-1], global_step=global_step)
                
                # torch.save(model, f'./runs/rnn_cintsyn={c}.pth')
                loss_list = []
                fc_list = []

                # add_conn_num = 10
                # add_conn_num = min(add_conn_num, (hp['n_rnn']**2) - now_total_num )
                
                # for _ in range(add_conn_num):
                #     mask = Gen_one_connection(mask, params, modelvar, Fc=None, device=device, undirected=False)
                #     now_total_num += 1
                
                # model.set_mask(torch.tensor(mask))
                # print(f'now_conn_num:{now_total_num}')
    
    writer.close()
    print("Optimization finished!")



def estimate_fisher(model, rule_train, n_trials, ewc_gamma=1.):
    '''Estimate diagonal of Fisher Information matrix for [model] on [dataset] using [n_samples].'''
    hp = model.hp

    # Prepare <dict> to store estimated Fisher Information matrix
    est_fisher_info = {}
    
    for n, p in model.named_parameters():
        device = p.device
        n = n.replace('.', '__')
        est_fisher_info[n] = p.detach().clone().zero_()

    # Set model to evaluation mode
    # criterion = nn.MSELoss()
    mode = model.training
    model.eval()

    for index in range(n_trials):

    # Estimate the FI-matrix for [n_samples] batches of size 1
        rule_train_now = hp['rng'].choice(rule_train)
        trial = task.generate_trials(
                rule_train_now, hp, 'random',
                batch_size=1)
        
        input = torch.from_numpy(trial.x).to(device)
        target = torch.from_numpy(trial.y).to(device)
        c_mask = torch.from_numpy(trial.c_mask).to(device)
        c_mask = c_mask.reshape(input.shape[0], input.shape[1], -1)

        # Run forward pass of model
        output = model(input)

        output = torch.sigmoid(output)
        
        # Calculate the MSE loss for this output
        # mse_loss = criterion(output, target)
        mse_loss = torch.mean(torch.square((target - output) * c_mask))
        # mse_loss = F.mse_loss(output, target, reduction='sum')
        
        # Calculate gradient of MSE loss
        model.zero_grad()
        mse_loss.backward()

        # Square gradients and keep running sum
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None:
                    n = n.replace('.', '__')
                    est_fisher_info[n] += (p.grad.detach() ** 2)

    # Normalize by sample size used for estimation
    est_fisher_info = {n: p/(index + 1) for n, p in est_fisher_info.items()}

    # Store new values in the network
    for n, p in model.named_parameters():
        n = n.replace('.', '__')
        # -mode (=MAP parameter estimate)
        model.register_buffer('{}_EWC_param_values'.format(n), p.detach().clone())
        # -precision (approximated by diagonal Fisher Information matrix)
        if hasattr(model, '{}_EWC_estimated_fisher'.format(n)):
            existing_values = getattr(model, '{}_EWC_estimated_fisher'.format(n))
            est_fisher_info[n] = ewc_gamma * existing_values + est_fisher_info[n]
        model.register_buffer('{}_EWC_estimated_fisher'.format(n), est_fisher_info[n])

    # Set model back to its initial training mode
    model.train(mode=mode)


def train_sequential_ewc(ruleset, device, rule_trains, args, hp=None):

    writer = SummaryWriter(comment=args.exp_name)
    
    default_hp = task.get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = args.seed
    hp['rng'] = np.random.RandomState(args.seed)
    hp['rule_trains'] = rule_trains
    hp['rules'] = [r for rs in rule_trains for r in rs]

    # Number of training iterations for each rule
    rule_train_iters = [len(r)*args.max_trials for r in rule_trains]
    # rule_train_iters[-1] *= 20

    model = RNN(hp=hp, device=device, rec_scale_factor=args.rec_scale_factor).to(device)
    if args.load_model:
        model = torch.load(args.load_model, map_location=device)

    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
    loss_list = []
    log = defaultdict(list)
    global_step = 0

    for i_rule_train, rule_train in enumerate(hp['rule_trains']):
        step = 0
        
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
            c_mask = c_mask.reshape(input.shape[0], input.shape[1], -1)

            output, hidden_states = model(input)

            if hp['loss_type'] == 'lsq' and not hp['use_snn']:
                output = torch.sigmoid(output)
    
            random_value = torch.rand_like(hidden_states) 
            hidden_states =  hidden_states + random_value * 1e-7
            hidden_states_mean = hidden_states.detach().mean(1).cpu().numpy()

            fc = np.corrcoef(hidden_states_mean, rowvar=False)
            fc_list.append(fc)  

            ewc_loss = 0.
            if i_rule_train > 0:
                ewc_losses = []
                for n, p in model.named_parameters():
                    # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                    n = n.replace('.', '__')
                    mean = getattr(model, '{}_EWC_param_values'.format(n))
                    fisher = getattr(model, '{}_EWC_estimated_fisher'.format(n))
                    # Calculate weight regularization loss
                    ewc_losses.append((fisher * (p-mean)**2).sum())
                ewc_loss = (1./2)*sum(ewc_losses)

            # loss = criterion(output, target) + hp['c_intsyn'] * ewc_loss
            loss = torch.mean(torch.square((target - output) * c_mask)) + hp['c_intsyn'] * ewc_loss
            
            loss_list.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            step += 1
            global_step +=1
            if step % 500 == 0:
                num_trails = step * hp['batch_size_train']

                print(f'Step [{num_trails}/{rule_train_iters[i_rule_train]}], Loss: {sum(loss_list):.4f}')

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
                
                log = do_eval(model, log, rule_train=rule_train)
                
                for list_name, perf_list in log.items():
                    if not 'min' in list_name and not 'avg' in list_name:
                        writer.add_scalar(tag=list_name, scalar_value=perf_list[-1], global_step=global_step)

                loss_list = []
                fc_list = []
        estimate_fisher(model, rule_train, args.fisher_samples * len(rule_train), args.ewc_gamma)
        last_trained_rules = "_".join(rule_train)
        torch.save(model, f'./runs/rnn_{last_trained_rules}_trained.pth')
    
    writer.close()
    print("Optimization finished!")


def start_parse():
    parser = argparse.ArgumentParser(description='interactive_modelling')
    parser.add_argument('--n_rnn', default=84, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=2024)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--max_trials', default=4e5, type=int)
    parser.add_argument('--rec_scale_factor', default=0.5, type=float)
    parser.add_argument('--reg_factor', default=1000.0, type=float)
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--loss_type', choices=['lsq', 'ce'], default='lsq')
    parser.add_argument('--use_ewc', action='store_true')
    parser.add_argument('--ksi', default=0.1, type=float)
    parser.add_argument('--non_linearity', choices=['tanh', 'softplus', 'relu', 'leakyrelu'], default='softplus')
    parser.add_argument('--easy_task', action='store_true')
    parser.add_argument('--task_num', default=20, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = start_parse()
    
    args.exp_name = ", ".join(f"{arg}={getattr(args, arg)}" for arg in vars(args))
    
    device = torch.device(f'cuda:{args.gpu}' if args.gpu>=0 else 'cpu')
    lock_random_seed(seed=args.seed)
    
    hp = dict()
    hp['n_rnn'] = args.n_rnn
    hp['easy_task'] = args.easy_task
    hp['learning_rate'] = 1e-3
    hp['w_rec_init'] = 'randortho'
    # hp['w_rec_init'] = 'diag'
    hp['activation'] = args.non_linearity
    hp['c_intsyn'] = args.reg_factor
    hp['loss_type'] = args.loss_type
    hp['ksi_intsyn'] = args.ksi
    
    if args.easy_task:

        # rule_trains = [['fdgo'], ['delaygo'], ['dm1', 'dm2'], ['multidm'],
        #             ['contextdm1', 'contextdm2']]

        rule_trains = [['dm1', 'dm2'], ['delaydm1', 'delaydm2']]
        

        if args.use_ewc:
            train_sequential_ewc(ruleset='all', device=device, rule_trains=rule_trains, \
                            hp=hp, args=args)
        else:
            train_sequential(ruleset='all', device=device, rule_trains=rule_trains, \
                            hp=hp, max_trials=args.max_trials, \
                            exp_name=f'train_sequential_{args.exp_name}', rec_scale_factor=args.rec_scale_factor)
    else:

        rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}

        hp['rule_trains'] = task.rules_dict['all']
        hp['rule_trains'] = hp['rule_trains'][:args.task_num]
        hp['rules'] = hp['rule_trains']

        train(ruleset='all', device=device, exp_name=args.exp_name, rule_prob_map=rule_prob_map, \
            seed=args.seed, max_trials=args.max_trials, hp=hp, rec_scale_factor=args.rec_scale_factor)