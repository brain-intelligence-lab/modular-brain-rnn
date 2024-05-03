import torch
import numpy as np
import datasets.multitask as task
from collections import defaultdict
import random
import os

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

def do_eval(model, rule_train):
    """Do evaluation.

    Args:
        model: Model class instance
        rule_train: string or list of strings, the rules being trained
    """
    log = defaultdict(list)
    hp = model.hp
    device = next(model.parameters()).device
    model.eval()
    for rule_test in hp['rules']:
        n_rep = 16
        batch_size_test_rep = int(hp['batch_size_test']/n_rep)
        perf_tmp = list()
        with torch.no_grad():
            for i_rep in range(n_rep):
                trial = task.generate_trials(
                    rule_test, hp, 'random', batch_size=batch_size_test_rep)
            
                input = torch.from_numpy(trial.x).to(device)
                y_hat_test = model(input)
                if hp['loss_type'] == 'lsq' and not hp['use_snn']:
                    y_hat_test = torch.sigmoid(y_hat_test).cpu().numpy()
                else:
                    y_hat_test = torch.nn.functional.softmax(y_hat_test, dim=-1)
                    y_hat_test = y_hat_test.cpu().numpy()
                
                # Cost is first summed over time,
                # and averaged across batch and units
                # We did the averaging over time through c_mask
                perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
                perf_tmp.append(perf_test)

        log['perf_'+rule_test].append(np.mean(perf_tmp, dtype=np.float64))
    model.train()

    if hasattr(rule_train, '__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]

    perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_avg'].append(perf_tests_mean)

    perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_min'].append(perf_tests_min)
    print('avg'+'  | perf {:0.3f}'.format(perf_tests_mean))
    print('min'+'  | perf {:0.3f}'.format(perf_tests_min))

    return log