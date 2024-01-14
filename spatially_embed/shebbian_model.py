import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, input, reccurrent):
        self.v = self.v * self.tau + input + reccurrent
        out = self.spikeact(self.v, self.gamma)
        self.v = self.v * (1 - out)
        return out

    def reset(self):
        self.v = 0

class HebbReccurrentLayer(nn.Module):
    def __init__(self, input_size, hidden_size, tau, v_th, gamma, structural_weight):
        super(HebbReccurrentLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        self.v_th = v_th
        self.gamma = gamma
        self.neuron = LIFNeuron(tau, v_th, gamma)
        self.forward_linear = nn.Linear(input_size, hidden_size)
        self.recurrent_weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.recurrent_weight.data.normal_(0, 1.0 / hidden_size)
        # register structural weight
        self.register_buffer('structural_weight', structural_weight)
        # self.structural_weight = nn.Parameter(structural_weight)
        # self.structural_weight.requires_grad = False
        self.running_input_freq = 0
        self.running_output_freq = 0
        self.T_count = 0

    def hebbian(self):
        '''
        delta_w = input_freq * output_freq - alpha * w * output_freq * output_freq
        :param running_input_freq: input spike frequency [batch_size, input_size]
        :param running_output_freq: output spike frequency [batch_size, out_size]
        :return:
        '''
        self.running_input_freq /= self.T_count
        self.running_output_freq /= self.T_count
        # 设置脉冲频率阈值为0.25
        delta_w = (self.running_input_freq.t() - 0.25)@ (self.running_output_freq - 0.25)
        # delta_w_s = F.softmax(delta_w, dim=0)
        # delta_w = delta_w * (delta_w_s > 0.8).float()
        self.structural_weight -= delta_w * 0.0001
        self.structural_weight = torch.clamp(self.structural_weight, min=0.0, max=1.0)
        self.running_input_freq = 0
        self.running_output_freq = 0
        self.T_count = 0

    def forward(self, input, reccurrent):
        input_ = self.forward_linear(input)
        # reccurrent_func = F.linear(reccurrent, self.recurrent_weight)
        # # reccurrent_struct = F.linear(reccurrent, self.structural_weight).detach()
        # reccurrent_ = reccurrent_func * reccurrent_struct
        reccurrent_ = F.linear(reccurrent, self.structural_weight.detach() * self.recurrent_weight)
        out = self.neuron(input_, reccurrent_)
        if self.training:
            self.T_count += 1
            self.running_input_freq += reccurrent
            self.running_output_freq += out
        return out

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num, tau, v_th, gamma, structural_weight):
        super(SNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_num = layer_num
        self.tau = tau
        self.v_th = v_th
        self.gamma = gamma
        self.hebbian = True
        self.reccurent_layers = nn.ModuleList()
        for _ in range(layer_num):
            self.reccurent_layers.append(HebbReccurrentLayer(input_size, hidden_size, tau, v_th, gamma, structural_weight))
            input_size = hidden_size
        self.output_layer = nn.Linear(hidden_size, output_size)

        # xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0.0)

    def forward(self, input):
        self.reset()
        T = input.size(0)
        batch_size = input.size(1)
        reccurrent = [torch.zeros(batch_size, self.hidden_size, device=input.device)] * self.layer_num
        r_out = torch.zeros(T, batch_size, self.hidden_size, device=input.device)
        for t in range(T):
            input_ = input[t, ...]
            for i in range(self.layer_num):
                out = self.reccurent_layers[i](input_, reccurrent[i])
                input_ = out
                reccurrent[i] = out
            r_out[t, ...] = out
        out_final = self.output_layer(r_out)
        return out_final

    def reset(self):
        for i in range(self.layer_num):
            self.reccurent_layers[i].neuron.reset()

    def hebb_update(self):
        for i in range(self.layer_num):
            self.reccurent_layers[i].hebbian()


if __name__ == '__main__':
    tau = 0.5
    v_th = 1.0
    gamma = 1.0
    input_size = 10
    hidden_size = 20
    output_size = 2
    layer_num = 2
    T = 10
    batch_size = 32
    structural_weight = torch.rand(hidden_size, hidden_size)
    snn = SNN(input_size, hidden_size, output_size, layer_num, tau, v_th, gamma, structural_weight)
    x = torch.randn(T, batch_size, input_size)
    optimizer = torch.optim.Adam(snn.parameters(), lr=0.001)
    for t in range(5):
        optimizer.zero_grad()
        out = snn(x)
        out.sum().backward()
        optimizer.step()
        snn.hebb_update()

    # for name, param in snn.named_parameters():
    #     print(name)
