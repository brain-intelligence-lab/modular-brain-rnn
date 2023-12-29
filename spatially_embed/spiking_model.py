import torch
import torch.nn as nn
import torch.nn.functional as F
from spatially_embed.loss_function import Hebbian

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

class ReccurrentLayer(nn.Module):
    def __init__(self, input_size, hidden_size, tau, v_th, gamma):
        super(ReccurrentLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        self.v_th = v_th
        self.gamma = gamma
        self.neuron = LIFNeuron(tau, v_th, gamma)
        self.forward_linear = nn.Linear(input_size, hidden_size)
        self.recurrent_weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # 手动初始化以避免训练前或训练后出现nan值
        nn.init.xavier_normal_(self.recurrent_weight) 
        # 不能初始化成同一个常数，这样会导致pearsonr计算结果为nan
        # self.recurrent_weight.data.fill_(1.0 / hidden_size)

    def forward(self, input, reccurrent):
        input_ = self.forward_linear(input)
        reccurrent = F.linear(reccurrent, self.recurrent_weight)
        out = self.neuron(input_, reccurrent)
        return out


class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num, tau, v_th, gamma, use_hebbian=False, lamb=0.001):
        super(SNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_num = layer_num
        self.tau = tau
        self.v_th = v_th
        self.gamma = gamma
        self.reccurent_layers = nn.ModuleList()
        for _ in range(layer_num):
            self.reccurent_layers.append(ReccurrentLayer(input_size, hidden_size, tau, v_th, gamma))
            input_size = hidden_size
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.hebbian = Hebbian(self.reccurent_layers[1].forward_linear.weight, lamb) if use_hebbian else None
        self.hebbian_reg_term = 0

    def forward(self, input):
        self.reset()
        T = input.size(0)
        batch_size = input.size(1)
        reccurrents = [torch.zeros(batch_size, self.hidden_size, device=input.device) \
            for _ in range(self.layer_num)]
        total_out = torch.zeros(T, batch_size, self.hidden_size, device=input.device)
        
        if self.hebbian and self.training:
            r_in = torch.zeros(T, batch_size, self.hidden_size, device=input.device)
            r_out = torch.zeros(T, batch_size, self.hidden_size, device=input.device)
        
        for t in range(T):
            input_ = input[t, ...]
            for i in range(self.layer_num):
                if self.hebbian and self.training and i == 1:
                    r_in[t, ...] = input_.detach()
                    
                out = self.reccurent_layers[i](input_, reccurrents[i])
                input_ = out
                reccurrents[i] = out
                
                if self.hebbian and self.training and i == 1:
                    r_out[t, ...] = out.detach()
                
            total_out[t, ...] = out
        
        out = self.output_layer(total_out)
        
        if self.hebbian and self.training:
            r_in = r_in.mean(0).mean(0)
            r_out = r_out.mean(0).mean(0)
            self.hebbian_reg_term = self.hebbian(r_in, r_out).sum()
            
        return out

    def reset(self):
        self.hebbian_reg_term = 0
        for i in range(self.layer_num):
            self.reccurent_layers[i].neuron.reset()

if __name__ == '__main__':
    tau = 0.5
    v_th = 1.0
    gamma = 1.0
    input_size = 10
    hidden_size = 20
    output_size = 2
    layer_num = 2
    T = 100
    batch_size = 32
    snn = SNN(input_size, hidden_size, output_size, layer_num, tau, v_th, gamma)
    x = torch.randn(T, batch_size, input_size)
    out = snn(x)
    print(out.shape)


