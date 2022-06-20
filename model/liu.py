import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init


class LIU(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9):
        super(LIU, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        self.conv1 = nn.Conv2d(input_channels, 80, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv1_bn = nn.BatchNorm2d(80)

        self.features_sizes = self._get_sizes()

        self.fc_enc = nn.Linear(self.features_sizes[2], n_classes)

        # Decoder
        self.fc1_dec = nn.Linear(self.features_sizes[2], self.features_sizes[2])
        self.fc1_dec_bn = nn.BatchNorm1d(self.features_sizes[2])
        self.fc2_dec = nn.Linear(self.features_sizes[2], self.features_sizes[1])
        self.fc2_dec_bn = nn.BatchNorm1d(self.features_sizes[1])
        self.fc3_dec = nn.Linear(self.features_sizes[1], self.features_sizes[0])
        self.fc3_dec_bn = nn.BatchNorm1d(self.features_sizes[0])
        self.fc4_dec = nn.Linear(self.features_sizes[0], input_channels)

        self.apply(self.weight_init)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        _, c, w, h = x.size()
        size0 = c * w * h

        x = self.pool1(x)
        _, c, w, h = x.size()
        size1 = c * w * h

        x = self.conv1_bn(x)
        _, c, w, h = x.size()
        size2 = c * w * h

        return size0, size1, size2

    def forward(self, x):
        x = x.squeeze()
        x_conv1 = self.conv1_bn(self.conv1(x))
        x = x_conv1
        x_pool1 = self.pool1(x)
        x = x_pool1
        x_enc = F.relu(x).view(-1, self.features_sizes[2])
        x = x_enc

        x_classif = self.fc_enc(x)
        return x_classif

        # # x = F.relu(self.fc1_dec_bn(self.fc1_dec(x) + x_enc))
        # x = F.relu(self.fc1_dec(x))
        # x = F.relu(
        #     self.fc2_dec_bn(self.fc2_dec(x) + x_pool1.view(-1, self.features_sizes[1]))
        # )
        # x = F.relu(
        #     self.fc3_dec_bn(self.fc3_dec(x) + x_conv1.view(-1, self.features_sizes[0]))
        # )
        # x = self.fc4_dec(x)
        # return x_classif, x