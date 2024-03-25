import torch
from functions.utils.eval_utils import gen_ortho_matrix
import torch.nn as nn


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
            self.recurrent_conn.weight.data = torch.tensor(gen_ortho_matrix(hidden_size, hp['rng']), dtype=torch.float32)
        elif hp['w_rec_init'] == 'diag':
            self.recurrent_conn.weight.data = torch.eye(hidden_size, dtype=torch.float32)
        
        self.rec_scale_factor = rec_scale_factor
        self.recurrent_conn.weight.data *= self.rec_scale_factor
        self.readout = nn.Linear(hidden_size, n_output)
                
        # TODO:这里使用relu训不上去，训到后面会出现nan
        if hp['activation'] == 'softplus':
            self.rnn_activation = nn.Softplus()
        elif hp['activation'] == 'relu':
            self.rnn_activation = nn.ReLU() 
        elif hp['activation'] == 'tanh':
            self.rnn_activation = nn.Tanh()
            
        self.device = device

        self.mask = torch.ones_like(self.recurrent_conn.weight.data).to(device)
        self.register_buffer("mymask", self.mask)

    def set_mask(self, mask):
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
        if self.training:
            return out, hidden_states.detach()
        else:
            return out


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