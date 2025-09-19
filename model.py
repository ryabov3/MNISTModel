import torch.nn as nn


class MNISTModel(nn.Module):
    def __init__(self, in_channels, out):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16), # 26x26
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3), # 24x24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # 12x12
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(32*12*12, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        out = self.linear(x)
        return out
