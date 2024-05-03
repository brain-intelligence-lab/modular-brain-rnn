import torch
from functions.generative_network_modelling.generative_network_modelling import Gen_one_connection
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
        for t in range(T):
            rec_noise = torch.rand_like(rnn_inputs[t]) * self.sigma
            output = self.rnn_activation(rnn_inputs[t] + rec_noise + \
                nn.functional.linear(self.hidden_state, masked_weights, self.recurrent_conn.bias))
            
            self.hidden_state = self.alpha * output + \
                (1 - self.alpha) * self.hidden_state 
            # 这里完全等效于hidden_states.append(self.hidden_state.clone())
            hidden_states.append(self.hidden_state)
        hidden_states = torch.stack(hidden_states, 0)
        out = self.readout(hidden_states)
        return out

    def gen_conn_matrix(self, random_prob=0.5):
        eta = -3.2
        gamma = 0.38
        params = [eta, gamma, 1e-5]
        modelvar = ['powerlaw', 'powerlaw']
        n = self.hp['n_rnn']
        mask = np.zeros((n, n), dtype=np.float32) 
        
        Dis = None
        if self.hp['wiring_rule'] in ['dis_rand', 'distance']:
            assert self.hp['n_rnn'] == 84
            Dis = np.load('/data_nv/dataset/brain_hcp_data/84/Raw_dis.npy')
        
        conn_num = self.hp['conn_num']
        mask_list = [mask.copy() for i in range(conn_num + 1)]
        for i in range(1, conn_num + 1):
            if self.hp['wiring_rule'] == 'dis_rand' \
                and np.random.random() < random_prob:
                mask_list[i] = Gen_one_connection(mask_list[i-1].copy(), params, modelvar, D=None, device=self.device, undirected=False)
            else:
                mask_list[i] = Gen_one_connection(mask_list[i-1].copy(), params, modelvar, D=Dis, device=self.device, undirected=False)
        
        self.mask_list = mask_list
        self.mask_idx = 0

    def update_conn_num(self, add_conn_num):
        self.mask_idx += add_conn_num # 1-index
        # assert self.mask_idx <= self.hp['conn_num'] 
        self.mask_idx = min(self.mask_idx, self.hp['conn_num'])
    
    def gen_mask_for_control(self):
        conn_num = self.mask_idx
        eta = -3.2
        gamma = 0.38
        params = [eta, gamma, 1e-5]
        modelvar = ['powerlaw', 'powerlaw']
        
        n = self.hp['n_rnn']
        mask = np.zeros((n, n), dtype=np.float32) 
        for i in range(conn_num):
            mask = Gen_one_connection(mask, params, modelvar, D=None, device=self.device, undirected=False)
        
        assert mask.sum() == conn_num
        mask = torch.tensor(mask).to(self.device)
        # 应用掩码矩阵
        with torch.no_grad():
            self.recurrent_conn.weight.grad *= mask
            
    def grow_connections(self, add_conn_num):
        self.update_conn_num(add_conn_num)
        mask = self.mask_list[self.mask_idx]
        self.set_mask(mask)
    
    def fix_connections(self):
        mask = self.mask_list[-1]
        self.set_mask(mask)
        self.mask_idx = self.hp['conn_num']
        
    def empty_connections(self):
        mask = torch.zeros_like(self.mask)
        self.set_mask(mask)
        
class SpikeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input > 0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None
    

class LIFNeuron(nn.Module):
    def __init__(self, tau, v_th, gamma):
        super(LIFNeuron, self).__init__()
        self.tau = tau
        self.v_th = v_th
        self.gamma = gamma
        self.spikeact = SpikeAct.apply
        self.v = 0
        self.total_neuron_num = 0
        self.fire_neuron_num = 0

    def forward(self, input, reccurrent):
        self.v = self.v * self.tau + input + reccurrent
        out = self.spikeact(self.v, self.gamma)
        self.v = self.v * (1 - out)
        if not self.training:
            self.total_neuron_num += out.numel()
            self.fire_neuron_num += out.sum().item()
        return out

    def reset(self):
        self.v = 0



class RSNN(nn.Module):
    def __init__(self, hp, device, rec_scale_factor=1.0):
        super(RSNN, self).__init__()
        self.hp = hp
        rule_start, n_rule, n_output =  hp['rule_start'], hp['n_rule'], hp['n_output']
        self.alpha, self.sigma = hp['alpha'], hp['sigma_rec']
        hidden_size = hp['n_rnn']
        self.sensory_readin = nn.Linear(rule_start, hidden_size)
        self.rule_readin = nn.Linear(n_rule, hidden_size, bias=False)
        self.hidden_state = 0
        self.recurrent_conn = nn.Linear(hidden_size, hidden_size)
        self.rec_scale_factor = rec_scale_factor
        self.recurrent_conn.weight.data *= self.rec_scale_factor

        self.readout = nn.Linear(hidden_size, n_output)
        self.rsnn_activation = LIFNeuron(tau=0.5, v_th=1.0, gamma=1.0)

        self.device = device

        self.mask = torch.ones_like(self.recurrent_conn.weight.data).to(device)
        self.register_buffer("mymask", self.mask)

    def set_mask(self, mask):
        self.mask = mask.clone().to(self.device)


    def forward(self, x):
        self.rsnn_activation.reset()
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
        for t in range(T):
            rec_noise = torch.rand_like(rnn_inputs[t]) * self.sigma
            output = self.rnn_activation(rnn_inputs[t] + rec_noise + \
                nn.functional.linear(self.hidden_state, masked_weights, self.recurrent_conn.bias))
            
            self.hidden_state = self.alpha * output + \
                (1 - self.alpha) * self.hidden_state 
            # 这里完全等效于hidden_states.append(self.hidden_state.clone())
            hidden_states.append(self.hidden_state)
        hidden_states = torch.stack(hidden_states, 0)
        out = self.readout(hidden_states)
        if self.training:
            return out, hidden_states.detach()
        else:
            return out