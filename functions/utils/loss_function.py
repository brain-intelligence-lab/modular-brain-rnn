import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatiallyLoss(nn.Module):
    def __init__(self, base_criterion, model, names, distances, reg_type='l1', lamb=0.01):
        '''
        :param base_criterion: 原始的任务损失函数 (nn.CrossEntropyLoss())
        :param model: 模型 (nn.Module)
        :param names: 模型中需要正则化的参数名称 (list)
        :param distances: 模型中需要正则化的参数对应的距离矩阵 (dict)
        :param reg_type: 正则化类型 (str)
        :param lamb: 正则化系数 (float)
        '''
        super(SpatiallyLoss, self).__init__()
        self.base_criterion = base_criterion
        self.model = model
        self.names = names
        self.distances = distances
        self.reg_type = reg_type
        self.lamb = lamb
        # reg_type: l1, l2, Euc_space, Com_space
        if self.reg_type == 'l1':
            self.reg = self.l1_reg
        elif self.reg_type == 'l2':
            self.reg = self.l2_reg
        elif self.reg_type == 'Euc_space':
            self.reg = self.Euc_space_reg
        elif self.reg_type == 'Com_space':
            self.reg = self.Com_space_reg
        else:
            raise NotImplementedError

    def l1_reg(self):
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.names:
                reg_loss += torch.sum(torch.abs(param))
        return reg_loss

    def l2_reg(self):
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.names:
                reg_loss += torch.sum(torch.pow(param, 2))
        return reg_loss

    def Euc_space_reg(self):
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.names:
                dis_mat = self.distances[name].to(param.device)
                weight_mat = torch.abs(param)
                reg_loss += torch.sum(dis_mat * weight_mat)
        return reg_loss

    def Com_space_reg(self):
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.names:
                dis_mat = self.distances[name].to(param.device)
                weight_mat = torch.abs(param)
                weight_node_strength = torch.sum(weight_mat, dim=1)
                weight_node_strength = torch.pow(weight_node_strength, -0.5)
                S = torch.diag(weight_node_strength)
                S_ = S @ weight_mat @ S 
                C = torch.linalg.matrix_exp(S_).fill_diagonal_(0)
                reg_loss += torch.sum(weight_mat * dis_mat * C)
        return reg_loss
        
    def forward(self, inputs, target):
        base_loss = self.base_criterion(inputs, target)
        reg_loss = self.reg()
        return base_loss + self.lamb * reg_loss


# https://julien-vitay.net/lecturenotes-neurocomputing/4-neurocomputing/5-Hebbian.html#

class Hebbian(nn.Module):
    def __init__(self, weight, alpha = 0.9, lamb = -1):
        '''
        :param weight: 某一层的权重矩阵
        :param lamb: hebbian项的缩放系数(-1: learnable)
        '''
        super(Hebbian, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.lamb = lamb
        # -1: learnable
        if lamb == -1:
            self.lamb = torch.rand_like(self.weight.data)
            self.lamb = nn.Parameter(self.lamb)
        
        self.theta_in = 0
        self.theta_out = 0
        
    def forward(self, r_in, r_out):
        '''
        :param r_in: in_feature (input对T、batch_size取mean之后得到的平均发放率)
        :param r_out: out_feature (output对T、batch_size取mean之后得到的平均发放率)
        '''
        r_in_ = (r_in - self.theta_in).unsqueeze(1)
        r_out_ = (r_out - self.theta_out).unsqueeze(0)
        
        # TODO:这里是应该先更新theta_in(theta_out)还是先计算r_in_(r_out_)？
        # 对batch_size取mean得到batch内的平均值
        self.theta_in = self.alpha * self.theta_in + (1 - self.alpha) * r_in
        self.theta_out = self.alpha * self.theta_out + (1 - self.alpha) * r_out
        
        delta_w = torch.matmul(r_in_, r_out_).detach()
        Hebbian_reg_term = self.lamb * delta_w * self.weight
        return Hebbian_reg_term
    
        