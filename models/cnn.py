import torch
import numpy as np
import torch.nn as nn

class SiameseNet(nn.Module):
    def __init__(self, input_channels=3, channels_1=32, channels_2=32, output_channels=64):
        super(SiameseNet, self).__init__()
        # Backbone network for feature extraction
        self.conv1 = nn.Conv2d(input_channels, channels_1, kernel_size=3, padding=1) # hidden_channels x 32 x 32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # hidden_channels x 16 x 16
        
        self.conv2 = nn.Conv2d(channels_1, channels_2, kernel_size=3, padding=1) # hidden_channels x 16 x 16
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # hidden_channels x 8 x 8
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels_2 * 8 * 8, 256)
        self.relu3 = nn.ReLU()   
        self.fc2 = nn.Linear(256, output_channels)
        

        
    def apply_scale_factor(self, scale_factor, layers_to_scale, one_init=False):
        with torch.no_grad():
            for layer in layers_to_scale:
                target_layer = dict(self.named_modules())[layer] 
                if one_init:
                    target_layer.weight.data = torch.ones_like(target_layer.weight.data)
                sum_before_scaling = target_layer.weight.data.abs().sum().item()
                target_layer.weight.data *= scale_factor
                sum_after_scaling = dict(self.named_modules())[layer].weight.data.abs().sum().item()
                assert np.isclose(sum_after_scaling, sum_before_scaling * scale_factor), "Scaled weight sum does not meet expectations!"
                print(f"Multiplied weights of layer '{layer}' by scale factor")
        

    def forward_one(self, x):
        """Process a single image through backbone network"""
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        out = self.fc2(x)
        
        return out

    def forward(self, x1, x2):
        """Process a pair of images"""
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2