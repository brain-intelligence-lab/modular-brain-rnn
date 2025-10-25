import argparse
import torch
from functions.utils.eval_utils import lock_random_seed
from tensorboardX import SummaryWriter
from multitask_train import train, train_sequential, get_chance_level
from datetime import datetime
import os
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
    parser.add_argument('--reg_factor', default=1.0, type=float)
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--conn_num', type=int, default=-1)
    parser.add_argument('--conn_mode',choices=['fixed', 'grow', 'full'], default='full')
    parser.add_argument('--rnn_type', choices=['RNN', 'GRU', 'LSTM'], default='RNN')
    parser.add_argument('--wiring_rule', choices=['distance', 'random'], default='distance')
    parser.add_argument('--loss_type', choices=['lsq', 'ce'], default='lsq')
    parser.add_argument('--ksi', default=0.1, type=float)
    parser.add_argument('--rule_set', choices=['all', 'mante', 'oicdmc'], default='all')
    parser.add_argument('--non_linearity', choices=['tanh', 'softplus', 'relu', 'leakyrelu'], default='relu')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--continual_learning', action='store_true')
    parser.add_argument('--task_num', default=20, type=int)
    parser.add_argument('--init_mode', choices=['randortho', 'diag', 'one_init'], default='randortho')
    parser.add_argument('--task_list', nargs='+', help='A list of tasks', default=None)
    parser.add_argument('--mask_type', choices=['modular', 'random'], default='random')
    parser.add_argument('--reg_term', action='store_true')
    parser.add_argument('--easy_task', action='store_true')
    parser.add_argument('--read_from_file', action='store_true')
    parser.add_argument('--get_chance_level', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = start_parse()
    lock_random_seed(seed=args.seed)
    assert args.conn_mode =='full' or args.conn_num != -1
    
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
    
    if args.get_chance_level:
        get_chance_level(args, writer=writer)
        writer.close()
        exit(0)
            
    if args.continual_learning:
        if args.task_list is None:
            rule_trains = [['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',],
                           ['dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo',],
                           ['dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',],
                           ['delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm'],
                           ]
        else:
            rule_trains = [[task] for task in args.task_list]

        # rule_trains = [['dm1', 'dm2'], ['delaydm1', 'delaydm2']]
        train_sequential(args, writer=writer, rule_trains=rule_trains)
    else:

        train(args, writer=writer)
    
    writer.close()
        