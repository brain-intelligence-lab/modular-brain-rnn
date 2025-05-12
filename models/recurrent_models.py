import torch
from functions.generative_network_modelling.generative_network_modelling import Gen_one_connection
import scipy.io
import numpy as np
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, hp, device, rec_scale_factor=1.0):
        super(RNN, self).__init__()
        self.hp = hp
        rule_start, n_rule, n_output =  hp['rule_start'], hp['n_rule'], hp['n_output']
        self.alpha, self.sigma = hp['alpha'], hp['sigma_rec']
        hidden_size = hp['n_rnn']
        self.sensory_readin = nn.Linear(rule_start, hidden_size)
        self.rule_readin = nn.Linear(n_rule, hidden_size, bias=False)
        self.hidden_state = 0
        self.recurrent_conn = nn.Linear(hidden_size, hidden_size)
        
        if hp['w_rec_init'] == 'randortho':
            nn.init.orthogonal_(self.recurrent_conn.weight)
        elif hp['w_rec_init'] == 'diag':
            self.recurrent_conn.weight.data = torch.eye(hidden_size, dtype=torch.float32)
        
        self.rec_scale_factor = rec_scale_factor
        self.recurrent_conn.weight.data *= self.rec_scale_factor
        self.readout = nn.Linear(hidden_size, n_output)
        
        self.comm_loss = 0.0         
    
        if hp['activation'] == 'softplus':
            self.rnn_activation = nn.Softplus()
        elif hp['activation'] == 'relu':
            self.rnn_activation = nn.ReLU() 
        elif hp['activation'] == 'tanh':
            self.rnn_activation = nn.Tanh()
        elif hp['activation'] == 'leakyrelu':
            self.rnn_activation = nn.LeakyReLU()
            
        self.device = device

        self.mask = torch.ones_like(self.recurrent_conn.weight.data).to(device)
        self.register_buffer("mymask", self.mask)
    
    def set_mask(self, mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        self.mask = mask.clone().to(self.device)
        
    def forward(self, x):
        # x:(T, B, input_size)
        sensory_input = x[:,:,:self.hp['rule_start']]
        rule_input = x[:,:,self.hp['rule_start']:]
        sensory_input = self.sensory_readin(sensory_input)
        rule_input = self.rule_readin(rule_input)
        rnn_inputs = sensory_input + rule_input

        masked_weights = self.recurrent_conn.weight * self.mask

        self.hidden_state = torch.zeros_like(rnn_inputs[0])
        T = x.size(0)
        hidden_states = []

        self.comm_loss = 0.0  # 重置 comm_loss
        for t in range(T):
            rec_noise = torch.rand_like(rnn_inputs[t]) * self.sigma
            output = self.rnn_activation(rnn_inputs[t] + rec_noise + \
                nn.functional.linear(self.hidden_state, masked_weights, self.recurrent_conn.bias))
            
            if self.hp['reg_term']:
                self.comm_loss += self.hp['reg_strength'] * torch.matmul(self.hidden_state, self.hp['Distance'] * masked_weights.abs() ).sum()
                
            self.hidden_state = self.alpha * output + \
                (1 - self.alpha) * self.hidden_state 
            # 这里完全等效于hidden_states.append(self.hidden_state.clone())
            hidden_states.append(self.hidden_state)
        hidden_states = torch.stack(hidden_states, 0)
        out = self.readout(hidden_states)
        return out

    def gen_conn_matrix(self, wiring_rule='distance'):
        eta = -3.2
        gamma = 0.38
        params = [eta, gamma, 1e-5]
        modelvar = ['powerlaw', 'powerlaw']
        n = self.hp['n_rnn']
        mask = np.zeros((n, n), dtype=np.float32) 

        if wiring_rule == 'distance':
            assert self.hp['n_rnn'] == 84
            if self.hp['n_rnn'] == 84:
                Distance = np.load('/data_smr/dataset/brain_hcp_data/84/Raw_dis.npy')
            Distance = Distance + 1e-5
        else:
            Distance = None
        
        conn_num = self.hp['conn_num']
        mask_list = [mask.copy() for i in range(conn_num + 1)]
        for i in range(1, conn_num + 1):
            mask_list[i] = Gen_one_connection(mask_list[i-1].copy(), params, modelvar, D=Distance, device=self.device, undirected=False)
        
        self.mask_list = mask_list
        self.mask_idx = 0
    
    def gen_modular_conn_matrix(self, module_size_list):
        n = self.hp['n_rnn']
        mask = np.zeros((n, n), dtype=np.float32) 
        assert np.sum(module_size_list) <= n
        
        # 填充子矩阵到对角线
        start_index = 0
        for size in module_size_list:
            if start_index + size <= n:
                mask[start_index:start_index + size, start_index:start_index + size] = np.ones((size, size))
                start_index += size
        
        self.mask_list = [mask]
        self.mask_idx = 0

    def update_conn_num(self, add_conn_num):
        self.mask_idx += add_conn_num # 1-index
        # assert self.mask_idx <= self.hp['conn_num'] 
        self.mask_idx = min(self.mask_idx, self.hp['conn_num'])
    
    # def gen_mask_for_control(self, wiring_rule='random'):
    #     conn_num = self.mask_idx
    #     eta = -2.0
    #     gamma = 0.38
    #     params = [eta, gamma, 1e-5]
    #     modelvar = ['powerlaw', 'powerlaw']

    #     if wiring_rule == 'distance':
    #         assert self.hp['n_rnn'] == 68 or self.hp['n_rnn'] == 84
    #         if self.hp['n_rnn'] == 68:
    #             Distance = scipy.io.loadmat('/home/wyuhang/example_euclidean.mat')['euclidean']
    #         elif self.hp['n_rnn'] == 84:
    #             Distance = np.load('/data_smr/dataset/brain_hcp_data/84/Raw_dis.npy')
    #         Distance = Distance + 1e-5
    #     else:
    #         Distance = None
        
    #     n = self.hp['n_rnn']
    #     mask = 1 - self.mask_list[-1]
    #     for i in range(conn_num):
    #         mask = Gen_one_connection(mask, params, modelvar, D=Distance, device=self.device, undirected=False)
    #     mask = mask - ( 1 - self.mask_list[-1] )
        
    #     assert mask.sum() <= conn_num
    #     mask = torch.tensor(mask).to(self.device)
    #     # 应用掩码矩阵
    #     with torch.no_grad():
    #         self.recurrent_conn.weight.grad *= mask
            
    def grow_connections(self, add_conn_num):
        self.update_conn_num(add_conn_num)
        mask = self.mask_list[self.mask_idx]
        self.set_mask(mask)
    
    def fix_connections(self):
        mask = self.mask_list[-1]
        self.set_mask(mask)
        self.mask_idx = 0
        
    def empty_connections(self):
        mask = torch.zeros_like(self.mask)
        self.set_mask(mask)
