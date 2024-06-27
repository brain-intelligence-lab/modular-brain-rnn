import scipy.io
import numpy as np
import bct
import torch
import torch.nn as nn
from collections import defaultdict
from generative_network_modelling.generative_network_modelling import Gen_one_connection, myplot, get_ks_list
from generative_network_modelling import bct_gpu
import spatially_embed.multitask as task
from multitask_train import do_eval, lock_random_seed
import matplotlib.pyplot as plt
import pdb

# eta (-3.606, 0.354)
# gamma (0.212, 0.495)

if __name__ == '__main__':

    lock_random_seed(2024)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # set how many connections to generate
    m = 1500
    
    # load
    Wtgt = scipy.io.loadmat('generative_network_modelling/brain_hcp_data/84/structureM_use.mat')['structureM_use'].astype(np.int16)

    # coor = np.load('generative_network_modelling/brain_hcp_data/84/mean_coordinate.npy')

    D = np.load('generative_network_modelling/brain_hcp_data/84/Raw_dis.npy')
    
    A = np.zeros_like(Wtgt[:,:,0])

    # set model parameters
    typemodeltype = 'matching'
    modelvar = ['powerlaw', 'powerlaw']

    hidden_states = np.load('generative_network_modelling/hidden_states_rnn_84.npy', allow_pickle=True)

    # hidden_states = np.load('generative_network_modelling/hidden_states_rsnn.npy', allow_pickle=True)

    # random_value = np.random.rand(*hidden_states.shape) 

    # hidden_states =  hidden_states + random_value * 1e-7

    Atgt = np.array(Wtgt > 0, dtype=float)

    y_target = [ [] for _ in range(4)]

    for p in range(5):
        y_target[0].append(bct.degrees_und(Atgt[:,:, p]))
        y_target[1].append(bct_gpu.clustering_coef_bu_gpu(Atgt[:,:, p], device=device))
        y_target[2].append(bct_gpu.betweenness_bin_gpu(Atgt[:,:, p], device=device))
        y_target[3].append(D[np.triu(Atgt[:,:, p], k=1) > 0])


    # set an example parameter combination

    conn_matrices = np.zeros((m, len(A), len(A)))

    all_energy = {}
    all_curve = {}

    params_energy = {}
    import time

    for eta_int in range(-35, -5, 1):
        for gamma_int in range(1, 25, 1):
            eta = eta_int / 10.0
            gamma = gamma_int / 10.0
            params = [eta, gamma, 1e-5]
    
            # # 随机布线
            # A = np.zeros_like(A)
            # for i in range(m):
            #     A = Gen_one_connection(A=A, params=params, modelvar=modelvar, device=device)
            #     conn_matrices[i] = A
            # all_curve['ks'], all_energy['random_energy'] = get_ks_list(conn_matrices, y_target, D=D, device=device)

            # params_energy[(eta, gamma, 'random_energy')] = np.array(all_energy['random_energy']).min()

            # 只根据距离布线
            A = np.zeros_like(A)
            for i in range(m):
                A = Gen_one_connection(A=A, params=params, modelvar=modelvar, D=D, device=device)
                conn_matrices[i] = A
            all_curve['ks'], all_energy['D_energy'] = get_ks_list(conn_matrices, y_target, D=D, device=device)

            params_energy[(eta, gamma, 'D_energy')] = np.array(all_energy['D_energy']).min()

            # 只根据matching布线
            A = np.zeros_like(A)
            for i in range(m):
                A = Gen_one_connection(A=A, params=params, modelvar=modelvar, use_matching=True, device=device)
                conn_matrices[i] = A
            all_curve['ks'], all_energy['matching_energy'] = get_ks_list(conn_matrices, y_target, D=D, device=device)

            params_energy[(eta, gamma, 'matching_energy')] = np.array(all_energy['matching_energy']).min()


            # 只根据fc布线
            A = np.zeros_like(A)
            for i in range(m):
                hidden_states_mean = hidden_states.mean(1)
                fc = np.corrcoef(hidden_states_mean, rowvar=False)
                # fc = np.abs(fc)
                fc[fc<0] = 0
                # fc = bct_gpu.matching_ind_fc(A, fc)
                # fc = np.abs(fc)
                A = Gen_one_connection(A, params, modelvar, Fc=fc, device=device)
                conn_matrices[i] = A
            all_curve['ks'], all_energy['fc_energy'] = get_ks_list(conn_matrices, y_target, D=D, device=device)

            params_energy[(eta, gamma, 'fc_energy')] = np.array(all_energy['fc_energy']).min()


            #根据距离和matching规则进行布线
            A = np.zeros_like(A)
            for i in range(m):
                A = Gen_one_connection(A=A, params=params, modelvar=modelvar, D=D, use_matching=True, device=device)
                conn_matrices[i] = A
            all_curve['ks'], all_energy['D_matching'] = get_ks_list(conn_matrices, y_target, D=D, device=device)
            params_energy[(eta, gamma, 'D_matching')] = np.array(all_energy['D_matching']).min()

            # 只根据距离和fc布线
            A = np.zeros_like(A)
            for i in range(m):
                hidden_states_mean = hidden_states.mean(1)
                fc = np.corrcoef(hidden_states_mean, rowvar=False)
                # fc = np.abs(fc)
                # fc = bct_gpu.matching_ind_fc(A, fc)
                fc[fc<0] = 0
                A = Gen_one_connection(A, params, modelvar, D=D, Fc=fc, device=device)
                conn_matrices[i] = A
            all_curve['ks'], all_energy['D_fc'] = get_ks_list(conn_matrices, y_target, D=D, device=device)
            params_energy[(eta, gamma, 'D_fc')] = np.array(all_energy['D_fc']).min()

            # 只根据距离、matching规则和fc布线
            A = np.zeros_like(A)
            for i in range(m):
                hidden_states_mean = hidden_states.mean(1)
                fc = np.corrcoef(hidden_states_mean, rowvar=False)
                # fc = np.abs(fc)
                fc[fc<0] = 0
                # fc = bct_gpu.matching_ind_fc(A, fc)
                A = Gen_one_connection(A, params, modelvar, D=D, use_matching=True, Fc=fc, device=device)
                conn_matrices[i] = A
            all_curve['ks'], all_energy['D_matching_fc'] = get_ks_list(conn_matrices, y_target, D=D, device=device)
            params_energy[(eta, gamma, 'D_matching_fc')] = np.array(all_energy['D_matching_fc']).min()

            print(params)
            with open('params_energy.txt', 'w') as file:
                for key, value in params_energy.items():
                    file.write('%s:%s\n' % (key, value))