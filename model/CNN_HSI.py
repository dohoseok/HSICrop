import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

'''Implementation for CONVOLUTIONAL NEURAL NETWORKS FOR HYPERSPECTRAL IMAGE CLASSIFICATION'''

class CNN_HSI(nn.Module):
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 5e-2)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            init.normal_(m.weight, 0, 5e-2)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def __init__(self, in_channel, nc):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 128, 1),
            nn.ReLU(),
            nn.LocalResponseNorm(3),
            nn.Dropout(0.6)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.LocalResponseNorm(3),
            nn.Dropout(0.6)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, nc, 1),
            nn.ReLU(),
            nn.AvgPool2d(5, 1)
        )

        self.apply(self.weight_init)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out3 = out3.squeeze(-1).squeeze(-1)
        return out3
