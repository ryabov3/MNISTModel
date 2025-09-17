import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self, in_channels, out):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3), # 26x26
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3), # 24x24
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(32*24*24, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        out = self.linear(x)
        return out