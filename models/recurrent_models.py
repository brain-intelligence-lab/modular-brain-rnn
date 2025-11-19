import torch
from functions.generative_network_modelling.generative_network_modelling import Gen_one_connection
from  functions.utils.math_utils import generate_adj_matrix
import scipy.io
import numpy as np
import torch.nn as nn

class base_recurrent_model(nn.Module):
    def __init__(self):
        super(base_recurrent_model, self).__init__()
    
    def set_activation(self, hp):
        if hp['activation'] == 'softplus':
            self.rnn_activation = nn.Softplus()
        elif hp['activation'] == 'relu':
            self.rnn_activation = nn.ReLU() 
        elif hp['activation'] == 'tanh':
            self.rnn_activation = nn.Tanh()
        elif hp['activation'] == 'leakyrelu':
            self.rnn_activation = nn.LeakyReLU()
        else:
            raise NotImplementedError


    def init_recurrent_layer(self, hp):
        hidden_size = hp['n_rnn']

        if hp['w_rec_init'] == 'randortho':
            nn.init.orthogonal_(self.recurrent_conn.weight)
        elif hp['w_rec_init'] == 'diag':
            self.recurrent_conn.weight.data = torch.eye(hidden_size, dtype=torch.float32)
        elif hp['w_rec_init'] == 'one_init':
            self.recurrent_conn.weight.data = torch.ones_like(self.recurrent_conn.weight.data)
        else:
            raise NotImplementedError

    def set_mask(self, mask):
        pass
    def gen_conn_matrix(self, wiring_rule='distance'):
        pass
    def grow_connections(self, add_conn_num):
        pass
    def fix_connections(self):
        pass
    def empty_connections(self):
        pass
    def update_conn_num(self, add_conn_num):
        pass
    def get_layer_to_analyze(self):
        pass


class RNN(base_recurrent_model):
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

        self.init_recurrent_layer(hp)
        self.rec_scale_factor = rec_scale_factor
        self.recurrent_conn.weight.data *= self.rec_scale_factor
        self.readout = nn.Linear(hidden_size, n_output)
        
        self.comm_loss = 0.0         
        self.set_activation(hp)
            
        self.device = device

        self.mask = torch.ones_like(self.recurrent_conn.weight.data).to(device)
        self.register_buffer("mymask", self.mask)
    
    def set_mask(self, mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        self.mask = mask.clone().to(self.device)
    
    def get_layer_to_analyze(self):
        return self.recurrent_conn.weight.data.detach().cpu().numpy()
        
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
        now_is_val = not self.training
        for t in range(T):
            if now_is_val :
                if x[t].abs().sum() < 1e-3: 
                    rec_noise = 0.0
                else:
                    rec_noise = torch.rand_like(rnn_inputs[t]) * self.sigma
            else:
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

    def gen_conn_matrix(self, wiring_rule='distance', eta=-3.2):
        params = [eta, 0.0, 1e-5]
        modelvar = ['powerlaw', 'powerlaw']
        n = self.hp['n_rnn']
        mask = np.zeros((n, n), dtype=np.float32) 

        if wiring_rule == 'distance':
            assert self.hp['n_rnn'] == 84
            if self.hp['n_rnn'] == 84:
                Distance = np.load('./datasets/brain_hcp_data/84/Raw_dis.npy')
            Distance = Distance + 1e-5
        else:
            Distance = None
        
        conn_num = self.hp['conn_num']
        mask_list = [mask.copy() for i in range(conn_num + 1)]
        for i in range(1, conn_num + 1):
            mask_list[i] = Gen_one_connection(mask_list[i-1].copy(), params, modelvar, D=Distance, device=self.device, undirected=False)
        
        self.mask_list = mask_list
        self.mask_idx = 0
    

    def update_conn_num(self, add_conn_num):
        self.mask_idx += add_conn_num # 1-index
        # assert self.mask_idx <= self.hp['conn_num'] 
        self.mask_idx = min(self.mask_idx, self.hp['conn_num'])


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

class GRU(base_recurrent_model):
    def __init__(self, hp, device, rec_scale_factor=1.0):
        super(GRU, self).__init__()
        self.hp = hp
        rule_start, n_rule, n_output =  hp['rule_start'], hp['n_rule'], hp['n_output']
        self.sigma = hp['sigma_rec']
        hidden_size = hp['n_rnn']
        self.W_xz = nn.Linear(rule_start + n_rule, hidden_size, bias=False)
        self.W_xr = nn.Linear(rule_start + n_rule, hidden_size, bias=False)
        self.W_xh = nn.Linear(rule_start + n_rule, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        
        self.readout = nn.Linear(hidden_size, n_output)

        self.recurrent_conn = self.W_hh
        self.init_recurrent_layer(hp)
        self.rec_scale_factor = rec_scale_factor
        self.W_hh.weight.data *= self.rec_scale_factor
        
        self.set_activation(hp)
            
        self.device = device
    
    def get_layer_to_analyze(self):
        return self.W_hh.weight.data.detach().cpu().numpy()
    
    def forward(self, x):
        # x:(T, B, input_size)
        self.hidden_state = torch.zeros((x.size(1), self.hp['n_rnn']), device=self.device)
        T = x.size(0)
        hidden_states = []

        now_is_val = not self.training
        for t in range(T):

            if now_is_val :
                if x[t].abs().sum() < 1e-3: 
                    r_noise, z_noise, h_noise = 0.0, 0.0, 0.0
                else:
                    r_noise = torch.rand_like(x[t]) * self.sigma
                    z_noise = torch.rand_like(x[t]) * self.sigma
                    h_noise = torch.rand_like(x[t]) * self.sigma
            else:

                r_noise = torch.rand_like(x[t]) * self.sigma
                z_noise = torch.rand_like(x[t]) * self.sigma
                h_noise = torch.rand_like(x[t]) * self.sigma
            
            R = torch.sigmoid(self.W_xr(x[t] + r_noise) + self.W_hr(self.hidden_state))
            Z = torch.sigmoid(self.W_xz(x[t] + z_noise) + self.W_hz(self.hidden_state))
            
            H_hat = self.activation(self.W_xh(x[t] + h_noise) + self.W_hh(R * self.hidden_state))
            
            self.hidden_state = Z * self.hidden_state + (1 - Z) * H_hat
                
            hidden_states.append(self.hidden_state)
        hidden_states = torch.stack(hidden_states, 0)
        out = self.readout(hidden_states)
        return out


class LSTM(base_recurrent_model):
    def __init__(self, hp, device, rec_scale_factor=1.0):
        super(LSTM, self).__init__()
        self.hp = hp
        rule_start, n_rule, n_output =  hp['rule_start'], hp['n_rule'], hp['n_output']
        self.sigma = hp['sigma_rec']
        hidden_size = hp['n_rnn']
        self.W_xi = nn.Linear(rule_start + n_rule, hidden_size, bias=False)
        self.W_xf = nn.Linear(rule_start + n_rule, hidden_size, bias=False)
        self.W_xo = nn.Linear(rule_start + n_rule, hidden_size, bias=False)
        self.W_xc = nn.Linear(rule_start + n_rule, hidden_size, bias=False)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)
        self.W_hc = nn.Linear(hidden_size, hidden_size)
        
        self.readout = nn.Linear(hidden_size, n_output)

        self.recurrent_conn = self.W_hc
        self.init_recurrent_layer(hp)
        self.rec_scale_factor = rec_scale_factor
        self.W_hc.weight.data *= self.rec_scale_factor
        
        self.set_activation(hp)
            
        self.device = device
    
    def get_layer_to_analyze(self):    
        return self.W_hc.weight.data.detach().cpu().numpy()
    
    def forward(self, x):
        # x:(T, B, input_size)
        self.C = torch.zeros((x.size(1), self.hp['n_rnn']), device=self.device)
        self.H = torch.zeros((x.size(1), self.hp['n_rnn']), device=self.device)
        
        T = x.size(0)
        hidden_states = []
        now_is_val = not self.training
        
        for t in range(T):

            if now_is_val :
                if x[t].abs().sum() < 1e-3: 
                    i_noise, f_noise, o_noise, c_noise = 0.0, 0.0, 0.0, 0.0
                else:
                    i_noise = torch.rand_like(x[t]) * self.sigma
                    f_noise = torch.rand_like(x[t]) * self.sigma
                    o_noise = torch.rand_like(x[t]) * self.sigma
                    c_noise = torch.rand_like(x[t]) * self.sigma
            else:
                i_noise = torch.rand_like(x[t]) * self.sigma
                f_noise = torch.rand_like(x[t]) * self.sigma
                o_noise = torch.rand_like(x[t]) * self.sigma
                c_noise = torch.rand_like(x[t]) * self.sigma

            I = torch.sigmoid(self.W_xi(x[t] + i_noise) + self.W_hi(self.H))
            F = torch.sigmoid(self.W_xf(x[t] + f_noise) + self.W_hf(self.H))
            O = torch.sigmoid(self.W_xo(x[t] + o_noise) + self.W_ho(self.H))
            
            C_hat = self.activation(self.W_xc(x[t] + c_noise) + self.W_hc(self.H))
            self.C = F * self.C + I * C_hat
            self.H = O * self.activation(self.C)
            hidden_states.append(self.H)
        hidden_states = torch.stack(hidden_states, 0)
        out = self.readout(hidden_states)
        return out

                