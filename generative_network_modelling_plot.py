import scipy.io
import numpy as np
import bct
import torch
import torch.nn as nn
from models.recurrent_models import RNN
import functions.generative_network_modelling.generative_network_modelling as gnm
from functions.generative_network_modelling import bct_gpu
from functions.utils.eval_utils import lock_random_seed
import datasets.multitask as task
import matplotlib.pyplot as plt
import pdb

def get_hidden_states(model, rule):
    hp = model.hp
    trial = task.generate_trials(
        rule, hp, 'random',
        batch_size=hp['batch_size_test'])

    input = torch.from_numpy(trial.x).to(device)
    with torch.no_grad():
        _, hidden_states = model(input)

    return hidden_states



def start_parse():
    import argparse
    parser = argparse.ArgumentParser(description='task_induced_modelling')
    parser.add_argument('--use_snn', action='store_true')
    parser.add_argument('--m', default=3000, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--num_of_people', default=10, type=int)
    parser.add_argument('--load_data', choices=['68', '84', '400'], default='84')
    parser.add_argument('--load_model', type=str)

    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = start_parse()
    lock_random_seed(2024)
    device = torch.device(f'cuda:{args.gpu}' if args.gpu>=0 else 'cpu')

    assert args.load_model is not None, "You must specify model_path by using load_model."
    # load trained model:
    # model = RNN(hp=task.get_default_hp('all'), device=device).to(device)
    model = torch.load(args.load_model, map_location=device)
    
    # set how many connections to generate
    m = args.m

    # get empirical data
    if args.load_data=='68':
        demo_data = scipy.io.loadmat('/data_nv/dataset/brain_hcp_data/68/demo_data.mat')
        
        demo_data = demo_data['demo_data']
        Wtgt = demo_data['Wtgt'][0][0]
        A = demo_data['seed'][0][0]
        D = demo_data['D'][0][0]
        # coordinates = demo_data['coordinates'][0][0]
    
    elif args.load_data=='84':
        Wtgt = scipy.io.loadmat('/data_nv/dataset/brain_hcp_data/84/structureM_use.mat')['structureM_use'].astype(np.int16)
        # coor = np.load('/data_nv/dataset/brain_hcp_data/84/mean_coordinate.npy')
        D = np.load('/data_nv/dataset/brain_hcp_data/84/Raw_dis.npy')
        A = np.zeros_like(Wtgt[:,:,0])
                
    elif args.load_data=='400':
        Wtgt = scipy.io.loadmat('/data_nv/dataset/brain_hcp_data/400/SC_matrix.mat')['SC_matrix']
        D = np.load('/data_nv/dataset/brain_hcp_data/400/Raw_dis.npy')
        A = np.zeros_like(Wtgt[:,:,0])

    # set model parameters
    typemodeltype = 'matching'
    modelvar = ['powerlaw', 'powerlaw']

    hidden_states = get_hidden_states(model, 'fdanti').cpu().numpy()

    random_value = np.random.rand(*hidden_states.shape) 

    hidden_states =  hidden_states + random_value * 1e-7

    # hidden_states:[T, Batch_size, hidden_size]

    if len(hidden_states.shape)==3:
        hidden_states_mean = hidden_states.mean(1)

    fc = np.corrcoef(hidden_states_mean, rowvar=False)
    fc[fc<0] = 0

    # permutation = np.random.permutation(len(fc))
    # sorted_indices = np.argsort(permutation)
    # fc = fc[sorted_indices][:, sorted_indices]
    
    Atgt = np.array(Wtgt > 0, dtype=float)

    y_target = [ [] for _ in range(4)]

    for p in range(args.num_of_people):
        y_target[0].append(bct.degrees_und(Atgt[:,:, p]))
        y_target[1].append(bct_gpu.clustering_coef_bu_gpu(Atgt[:,:, p], device=device))
        y_target[2].append(bct_gpu.betweenness_bin_gpu(Atgt[:,:, p], device=device))
        y_target[3].append(D[np.triu(Atgt[:,:, p], k=1) > 0])

    # set an example parameter combination
    eta = -3.2
    gamma = 0.38
    params = [eta, gamma, 1e-5]
    conn_matrices = np.zeros((m, len(A), len(A)))

    all_energy = {}
    all_curve = {}

    # 随机布线
    A = np.zeros_like(A)
    q_value_list = []
    for i in range(m):
        A = gnm.Gen_one_connection(A=A, params=params, modelvar=modelvar, device=device)
        conn_matrices[i] = A
        # ci, q_value = bct.community_louvain(W=A, B='modularity', seed=2024)
        # q_value_list.append(q_value)

    all_curve['ks'], all_energy['random_energy'] = gnm.get_ks_list(conn_matrices, y_target, D=D, device=device)
    # plt.plot([i+1 for i in range(len(q_value_list))], q_value_list, label='random')
    
    print('random_ok!')

    # 只根据距离布线
    A = np.zeros_like(A)
    q_value_list = []
    for i in range(m):
        A = gnm.Gen_one_connection(A=A, params=params, modelvar=modelvar, D=D, device=device)
        conn_matrices[i] = A
        # ci, q_value = bct.community_louvain(W=A, B='modularity', seed=2024)
        # q_value_list.append(q_value)

    all_curve['ks'], all_energy['D_energy'] = gnm.get_ks_list(conn_matrices, y_target, D=D, device=device)
    # plt.plot([i+1 for i in range(len(q_value_list))], q_value_list, label='Dis')

    print('distance_ok!')

    # 只根据matching布线
    A = np.zeros_like(A)
    q_value_list = []
    for i in range(m):
        A = gnm.Gen_one_connection(A=A, params=params, modelvar=modelvar, use_matching=True, device=device)
        conn_matrices[i] = A
        # ci, q_value = bct.community_louvain(W=A, B='modularity', seed=2024)
        # q_value_list.append(q_value)

    all_curve['ks'], all_energy['matching_energy'] = gnm.get_ks_list(conn_matrices, y_target, D=D, device=device)
    # plt.plot([i+1 for i in range(len(q_value_list))], q_value_list, label='matching')

    print('matching_ok!')

    # 只根据fc布线
    A = np.zeros_like(A)
    q_value_list = []
    for i in range(m):
        A = gnm.Gen_one_connection(A, params, modelvar, Fc=fc, device=device)
        conn_matrices[i] = A
        # ci, q_value = bct.community_louvain(W=A, B='modularity', seed=2024)
        # q_value_list.append(q_value)
  
    all_curve['ks'], all_energy['fc_energy'] = gnm.get_ks_list(conn_matrices, y_target, D=D, device=device)
    # plt.plot([i+1 for i in range(len(q_value_list))], q_value_list, label='fc')

    print('fc_ok!')

    #根据距离和matching规则进行布线
    A = np.zeros_like(A)
    q_value_list = []
    for i in range(m):
        A = gnm.Gen_one_connection(A=A, params=params, modelvar=modelvar, D=D, use_matching=True, device=device)
        conn_matrices[i] = A
        # ci, q_value = bct.community_louvain(W=A, B='modularity', seed=2024)
        # q_value_list.append(q_value)

    all_curve['ks'], all_energy['D_matching'] = gnm.get_ks_list(conn_matrices, y_target, D=D, device=device)
    # plt.plot([i+1 for i in range(len(q_value_list))], q_value_list, label='D_matching')

    print('D_matching_ok!')

    # 只根据距离和fc布线
    A = np.zeros_like(A)
    q_value_list = []
    for i in range(m):
        A = gnm.Gen_one_connection(A, params, modelvar, D=D, Fc=fc, device=device)
        conn_matrices[i] = A
        # ci, q_value = bct.community_louvain(W=A, B='modularity', seed=2024)
        # q_value_list.append(q_value)

    all_curve['ks'], all_energy['D_fc'] = gnm.get_ks_list(conn_matrices, y_target, D=D, device=device)
    # plt.plot([i+1 for i in range(len(q_value_list))], q_value_list, label='D_fc')

    print('D_fc_ok!')

    # 只根据距离、matching规则和fc布线
    A = np.zeros_like(A)
    q_value_list = []
    for i in range(m):
        A = gnm.Gen_one_connection(A, params, modelvar, D=D, use_matching=True, Fc=fc, device=device)
        conn_matrices[i] = A
        # ci, q_value = bct.community_louvain(W=A, B='modularity', seed=2024)
        # q_value_list.append(q_value)

    all_curve['ks'], all_energy['D_matching_fc'] = gnm.get_ks_list(conn_matrices, y_target, D=D, device=device)
    # plt.plot([i+1 for i in range(len(q_value_list))], q_value_list, label='D_matching_fc')
    print('D_matching_fc_ok!')

    # plt.title(f'Modularity of different generative ways')
    # plt.xlabel('Connections')
    # plt.legend()
    # plt.savefig(f'Modularity.jpg')

    # KS_k = [all_curve['ks'][i][0] for i in range(len(all_curve['ks']))]
    # KS_c = [all_curve['ks'][i][1] for i in range(len(all_curve['ks']))]
    # KS_b = [all_curve['ks'][i][2] for i in range(len(all_curve['ks']))]
    # KS_e = [all_curve['ks'][i][3] for i in range(len(all_curve['ks']))]

    for list_name, energy_list in all_energy.items():
        # offset_list = [value + (0.05 * (np.random.rand()-0.5) ) for value in energy_list] 
        # plt.plot([i+1 for i in range(len(offset_list))], offset_list, label=list_name, linewidth=2.0)
        plt.plot([i+1 for i in range(len(energy_list))], energy_list, label=list_name)

    # plt.plot([i+1 for i in range(len(KS_k))], KS_k, label='KS_k(degree)')
    # plt.plot([i+1 for i in range(len(KS_c))], KS_c, label='KS_c(clustering)')
    # plt.plot([i+1 for i in range(len(KS_b))], KS_b, label='KS_b(betweenness)')
    # plt.plot([i+1 for i in range(len(KS_e))], KS_e, label='KS_e(edge length)')
        
    plt.title('All_generative_methods')
    plt.xlabel('Connections')
    # plt.ylabel('Energy')

    plt.legend()
    plt.savefig(f'generative_modelling_{args.load_data}_{args.load_model}.jpg')