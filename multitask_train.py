import torch
import torch.nn as nn
import spatially_embed.multitask as task
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pdb
import os
import random

def lock_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = "myseed"  # str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def gen_ortho_matrix(dim, rng=None):
    """Generate random orthogonal matrix
    Taken from scipy.stats.ortho_group
    Copied here from compatibilty with older versions of scipy
    """
    H = np.eye(dim)
    for n in range(1, dim):
        if rng is None:
            x = np.random.normal(size=(dim-n+1,))
        else:
            x = rng.normal(size=(dim-n+1,))
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0])
        x[0] += D*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = -D*(np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    return H

class RNN(nn.Module):
    def __init__(self, hp):
        super(RNN, self).__init__()
        self.hp = hp
        rule_start, n_rule, n_output =  hp['rule_start'], hp['n_rule'], hp['n_output']
        self.alpha, self.sigma = hp['alpha'], hp['sigma_rec']
        hidden_size = hp['n_rnn']
        self.sensory_readin = nn.Linear(rule_start, hidden_size)
        self.rule_readin = nn.Linear(n_rule, hidden_size, bias=False)
        self.hidden_state = 0
        self.recurrent_conn = nn.Linear(hidden_size, hidden_size)
        # 这样初始化会导致一开始几万个trial训练很慢，但后面就上去了
        scale_factor_q = 0.5
        if hp['w_rec_init'] == 'diag':
            self.recurrent_conn.weight.data = torch.eye(hidden_size) * scale_factor_q        
        elif hp['w_rec_init'] == 'randortho':
            rng = np.random.RandomState()
            self.recurrent_conn.weight.data = torch.from_numpy(scale_factor_q * \
                  gen_ortho_matrix(hidden_size, rng=rng).astype(np.float32))

        self.readout = nn.Linear(hidden_size, n_output)
        # TODO:这里使用relu训不上去，训到后面会出现nan
        if hp['activation'] == 'softplus':
            self.rnn_activation = nn.Softplus()
        elif hp['activation'] == 'relu':
            self.rnn_activation = nn.ReLU() 

    def forward(self, x):
        # x:(T, B, input_size)
        sensory_input = x[:,:,:self.hp['rule_start']]
        rule_input = x[:,:,self.hp['rule_start']:]
        sensory_input = self.sensory_readin(sensory_input)
        rule_input = self.rule_readin(rule_input)
        rnn_inputs = sensory_input + rule_input

        self.hidden_state = torch.zeros_like(rnn_inputs[0])
        T = x.size(0)
        hidden_states = []
        for t in range(T):
            rec_noise = torch.rand_like(rnn_inputs[t]) * self.sigma
            output = self.rnn_activation(rnn_inputs[t] + rec_noise + \
                self.recurrent_conn(self.hidden_state))
            
            self.hidden_state = self.alpha * output + \
                (1 - self.alpha) * self.hidden_state 
            # 这里完全等效于hidden_states.append(self.hidden_state.clone())
            hidden_states.append(self.hidden_state)
        hidden_states = torch.stack(hidden_states, 0)
        out = self.readout(hidden_states)
        return out

def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)
    
def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.2*np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return perf
    
def do_eval(model, log, rule_train):
    """Do evaluation.

    Args:
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hp = model.hp
    for rule_test in hp['rules']:
        n_rep = 16
        batch_size_test_rep = int(hp['batch_size_test']/n_rep)
        perf_tmp = list()
        with torch.no_grad():
            for i_rep in range(n_rep):
                trial = task.generate_trials(
                    rule_test, hp, 'random', batch_size=batch_size_test_rep)
            
                input = torch.from_numpy(trial.x).to(device)
                output = model(input)
                if hp['loss_type'] == 'lsq':
                    y_hat_test = torch.sigmoid(output).cpu().numpy()
                else:
                    y_hat_test = output.cpu().numpy()
                

                # Cost is first summed over time,
                # and averaged across batch and units
                # We did the averaging over time through c_mask
                perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
                perf_tmp.append(perf_test)

        log['perf_'+rule_test].append(np.mean(perf_tmp, dtype=np.float64))

    if hasattr(rule_train, '__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]
    perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_avg'].append(perf_tests_mean)

    perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_min'].append(perf_tests_min)
    print('avg'+'  | perf {:0.3f}'.format(np.mean(perf_tests_mean)))
    print('min'+'  | perf {:0.3f}'.format(np.mean(perf_tests_min)))

    return log


# https://github.com/gyyang/multitask/blob/master/train.py
            
def train(ruleset, device, hp=None, max_steps=1e7, seed=2024, rule_prob_map=None, rule_trains=None, save_name='interleaved.jpg'):
    # 原文的图x-axis坐标是3000，指的是3000*1000trials，而不是3000trails
    default_hp = task.get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)
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

    model = RNN(hp=hp).to(device)

    step = 0
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []
    log = defaultdict(list)

    while step * hp['batch_size_train'] <= max_steps:
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

        output = model(input)
        if hp['loss_type'] == 'lsq':
            output = torch.sigmoid(output)

        #TODO: 这里tmpsum会特别靠近0，是为什么
        # tmpsum = output[-1,:,1:].detach().cpu().numpy().sum(axis=-1).mean()
        # print(f'mean_tmpsum: {tmpsum:.4f}')

        loss = criterion(output, target)
        loss_list.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        if step % 500 == 0:
            num_steps = step * hp['batch_size_train']
            print(f'Step [{num_steps}/{max_steps}], Loss: {sum(loss_list):.4f}')
            log = do_eval(model, log, rule_train=hp['rule_trains'])
            if log['perf_min'][-1] > model.hp['target_perf']:
                print('Perf reached the target: {:0.2f}'.format(
                    hp['target_perf']))
                break
            loss_list = []
    log_plot(log, savename=save_name)

    print("Optimization finished!")
    
def log_plot(log, savename):
    plt.clf()

    for list_name, perf_list in log.items():
        if not 'min' in list_name and not 'avg' in list_name:
            plt.plot([i+1 for i in range(len(perf_list))], perf_list, label=list_name)

    plt.title('All_task_perf_curve')
    plt.xlabel('Steps Axis')
    plt.ylabel('Performance Axis')

    plt.legend()
    plt.savefig(savename)

def train_sequential(ruleset, device, rule_trains, hp=None, max_steps=1e4, seed=2024, save_name='sequential.jpg'):
    '''Train the network sequentially.

    Args:
        rule_trains: a list of list of tasks to train sequentially
        max_steps: int, maximum number of training steps for each list of tasks
        ruleset: the set of rules to train
        seed: int, random seed to be used
    '''
    default_hp = task.get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)
    hp['rule_trains'] = rule_trains
    hp['rules'] = [r for rs in rule_trains for r in rs]

    # Number of training iterations for each rule
    rule_train_iters = [len(r)*max_steps for r in rule_trains]

    model = RNN(hp=hp).to(device)

    def get_current_param_list(model, clone=True):
        v_list = []
        for _, param in model.named_parameters():
            if clone:
                v_list.append(param.clone())
            else:
                v_list.append(param)
        return v_list

    # Using continual learning or not
    # c, ksi = hp['c_intsyn'], hp['ksi_intsyn']
    c, ksi = 1.0, 0.01

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []
    log = defaultdict(list)

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

            # Continual learning with intelligent synapses
            v_prev = v_current

            output = model(input)
            if hp['loss_type'] == 'lsq':
                output = torch.sigmoid(output)

            #TODO: 这里tmpsum会特别靠近0，是为什么
            # tmpsum = output[-1,:,1:].detach().cpu().numpy().sum(axis=-1).mean()
            # print(f'mean_tmpsum: {tmpsum:.4f}')

            # Update cost 
            cost_reg = 0.
            now_v = get_current_param_list(model, clone=False)
            
            for v, w, v_val in zip(now_v, Omega0, v_anc0):
                cost_reg += c * torch.sum(
                    w.detach() * (v - v_val.detach())**2)

            loss = criterion(output, target) + cost_reg
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
            if step % 500 == 0:
                num_steps = step * hp['batch_size_train']
                print(f'Step [{num_steps}/{rule_train_iters[i_rule_train]}], Loss: {sum(loss_list):.4f}')
                log = do_eval(model, log, rule_train=rule_train)
                loss_list = []
        
        log_plot(log, savename=save_name)
        
    print("Optimization finished!")

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = 2024
    lock_random_seed(seed=seed)
    rule_trains = None
    hp = {'activation': 'softplus', 'w_rec_init': 'diag'}
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    train(ruleset='all', device=device, seed=seed, max_steps=3e6, hp=hp, \
           rule_prob_map=rule_prob_map, rule_trains=rule_trains)
    
    hp = dict()
    hp['w_rec_init'] = 'randortho'
    hp['easy_task'] = True
    hp['activation'] = 'softplus'
    # hp['activation'] = 'relu' #目前relu还是会有问题
    hp['c_intsyn'] = 0.0
    hp['ksi_intsyn'] = 0.01
    hp['max_steps'] = 4e5

    rule_trains = [['fdgo'], ['delaygo'], ['dm1', 'dm2'], ['multidm'],
                   ['contextdm1', 'contextdm2']]
    train_sequential(ruleset='all', device=device, rule_trains=rule_trains,\
                      hp=hp, max_steps=hp['max_steps'], save_name='sequential.jpg')
    

    hp['c_intsyn'] = 1.0
    train_sequential(ruleset='all', device=device, rule_trains=rule_trains,\
                      hp=hp, max_steps=hp['max_steps'], save_name='continual_learning.jpg')
