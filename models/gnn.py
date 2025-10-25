import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

    def apply_scale_factor(self, scale_factor, layers_to_scale, one_init=False):
        with torch.no_grad():
            for layer in layers_to_scale:
                target_layer = dict(self.named_modules())[layer] 
                if one_init:
                    target_layer.lin.weight.data = torch.ones_like(target_layer.lin.weight.data)
                sum_before_scaling = target_layer.lin.weight.data.abs().sum().item()
                target_layer.lin.weight.data *= scale_factor
                sum_after_scaling = dict(self.named_modules())[layer].lin.weight.data.abs().sum().item()
                assert np.isclose(sum_after_scaling, sum_before_scaling * scale_factor), "缩放后的权重和不符合预期！"
                print(f"已将层 '{layer}' 的权重乘以缩放)")