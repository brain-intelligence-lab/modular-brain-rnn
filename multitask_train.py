import torch
import torch.nn as nn
import spatially_embed.multitask as task
from collections import defaultdict
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
        # TODO:原文说这样初始化, 但我反而训不太动
        # scale_factor_q = 0.5
        # self.recurrent_conn.weight.data = torch.eye(hidden_size) * scale_factor_q        
        
        self.readout = nn.Linear(hidden_size, n_output)
        # TODO:这里使用relu训不上去，训到后面会出现nan
        self.rnn_activation = nn.Softplus()
        # self.rnn_activation = nn.ReLU() 

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
        sess: tensorflow session
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hp = model.hp
    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    # print('Trial {:7d}'.format(log['trials'][-1]) +
    #       '  | Time {:0.2f} s'.format(log['times'][-1]) +
    #       '  | Now training '+rule_name_print)
    for rule_test in hp['rule_trains']:
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
        # print('{:15s}'.format(rule_test) +
        #       '  | perf {:0.2f}'.format(np.mean(perf_tmp)))


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

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = 2024
    lock_random_seed(seed=seed)
    # TODO:目前oicdmc(较快), mante(较慢)可以用softplus训起来，all勉强能训，更慢
    ruleset = 'all'
    hp = task.get_default_hp(ruleset=ruleset)
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)
    hp['rule_trains'] = task.rules_dict[ruleset]

    # Assign probabilities for rule_trains.
    rule_prob_map = dict()

    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        if ruleset == 'all':
            # Set default as 1.
            prob_base = 1.0 / (18 * 1.0 + 2 * 5.0)
            rule_prob = np.array(
                    [rule_prob_map.get(r, prob_base) for r in hp['rule_trains']])
            for idx, rule_name in enumerate(hp['rule_trains']):
                if rule_name in task.rules_dict['mante']:
                    rule_prob[idx] = prob_base * 5.0
            hp['rule_probs'] = list(rule_prob)
        else:
            # Set default as 1.
            rule_prob = np.array(
                    [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
            hp['rule_probs'] = list(rule_prob/np.sum(rule_prob))

    model = RNN(hp=hp).to(device)

    step = 0
    max_steps = int(2e4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []
    log = defaultdict(list)

    while step <= max_steps:
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
        if step % 50 == 0:
            print(f'Step [{step}/{max_steps}], Loss: {sum(loss_list):.4f}')
            log = do_eval(model, log, rule_train=hp['rule_trains'])
            # print(log)
            loss_list = []

    print("Optimization finished!")
    