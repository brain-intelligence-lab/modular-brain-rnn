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
                diag_weight_node_strength = torch.diag(weight_node_strength)
                S = diag_weight_node_strength
                # S = torch.pow(diag_weight_node_strength, -0.5)
                # S = S @ weight_mat @ S # S = D^(-1/2) * W * D^(-1/2) 会产生nan值
                # S = S  @ S
                # if S has nan print nan these print ok
                # C = e^S
                # C = torch.linalg.matrix_exp(S).fill_diagonal_(0)
                C = S
                # normalize C
                reg_loss += torch.sum(weight_mat * dis_mat * C)
        return reg_loss

    def forward(self, inputs, target):
        base_loss = self.base_criterion(inputs, target)
        reg_loss = self.reg()
        return base_loss + self.lamb * reg_loss





