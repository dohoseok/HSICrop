import torch
from torch import nn
import torch.nn.functional as F

class SeBranch(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 20, 16)
        self.bn1 = nn.BatchNorm1d(20)
        self.maxPooling = nn.MaxPool1d(5, 5)
        self.conv2 = nn.Conv1d(20, 20, 16)
        self.bn2 = nn.BatchNorm1d(20)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x_padding = F.pad(input, [8, 7])
        x_conv1 = self.relu(self.bn1(self.conv1(x_padding)))
        x_unsample = self.maxPooling(x_conv1)
        x_unsample_padding = F.pad(x_unsample, [8, 7])
        feature = self.relu(self.bn2(self.conv2(x_unsample_padding)))
        return feature

class SaBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 30, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(30)
        self.maxPooling = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 30, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(30)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        # input size: 21*21*1
        x_conv1 = self.relu(self.bn1(self.conv1(input)))
        x_unsample = self.maxPooling(x_conv1)
        feature = self.relu(self.bn2(self.conv2(x_unsample)))
        return feature

class TwoCNN(nn.Module):
    def __init__(self, bands, nc):
        super().__init__()
        self.seBranch = SeBranch(bands)
        self.saBranch = SaBranch()
        fusion_size = (bands // 5) * 20 + (21 // 2) ** 2 *30
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, nc)
        )

    def forward(self, x):
        spectra = x[...,11,11]
        neighbor_region = torch.mean(x, dim=-3)
        # print("x", x.shape, "spectra", spectra.shape, "neighbor_region", neighbor_region.shape)
        # spectra, neighbor_region):
        se = self.seBranch(spectra)
        sa = self.saBranch(neighbor_region)
        batchsz = se.shape[0]
        se = se.view(batchsz, -1)
        sa = sa.reshape(batchsz, -1)
        fusion = torch.cat([se, sa], dim=-1)
        logits = self.classifier(fusion)
        return logits