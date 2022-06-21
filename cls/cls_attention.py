import torch
from torch import nn
import copy


class SE(nn.Module):
    """
    https://arxiv.org/abs/1709.01507
    """
    def __init__(self, r, C):
        super().__init__()
        self.r = r
        self.C = C
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(self.C, self.C // self.r)
        self.dense2 = nn.Linear(self.C // self.r, self.C)

    def forward(self, x):
        batch_size = x.size(0)
        x_c = self.globalavgpool(x)
        x_c = x_c.view((batch_size, -1))
        x_c = self.dense1(x_c)
        x_c = nn.ReLU(inplace=True)(x_c)
        x_c = self.dense2(x_c)
        x_c = nn.Sigmoid()(x_c)
        x_c = x_c.view((batch_size, self.C, 1, 1))
        x = x + x * x_c

        return x


class BAM(nn.Module):
    """
    https://arxiv.org/abs/1807.06514
    """
    def __init__(self, r, C):
        super(BAM, self).__init__()
        self.r = r
        self.C = C
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(self.C, self.C // self.r)
        self.bn1 = nn.BatchNorm1d(self.C // self.r)
        self.dense2 = nn.Linear(self.C // self.r, self.C)

        self.conv1 = nn.Conv2d(
            self.C, self.C // self.r, kernel_size=1,
            stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(self.C // self.r)
        self.conv2 = nn.Conv2d(
            self.C // self.r, self.C // self.r, kernel_size=3,
            stride=1, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm2d(self.C // self.r)
        self.conv3 = nn.Conv2d(
            self.C // self.r, self.C // self.r, kernel_size=3,
            stride=1, padding=4, dilation=4)
        self.bn4 = nn.BatchNorm2d(self.C // self.r)
        self.conv4 = nn.Conv2d(self.C // self.r, 1, kernel_size=1, stride=1)

    def forward(self, x):
        batch_size = x.size(0)
        x_c = self.globalavgpool(x)
        x_c = x_c.view((batch_size, -1))
        x_c = self.dense1(x_c)
        x_c = self.bn1(x_c)
        x_c = nn.ReLU(inplace=True)(x_c)
        x_c = self.dense2(x_c)
        x_c = x_c.view((batch_size, self.C, 1, 1))

        x_s = self.conv1(x)
        x_s = self.bn2(x_s)
        x_s = nn.ReLU(inplace=True)(x_s)
        x_s = self.conv2(x_s)
        x_s = self.bn3(x_s)
        x_s = nn.ReLU(inplace=True)(x_s)
        x_s = self.conv3(x_s)
        x_s = self.bn4(x_s)
        x_s = nn.ReLU(inplace=True)(x_s)
        x_s = self.conv4(x_s)

        x_cs = nn.Sigmoid()(x_c + x_s)
        x_cs = x * x_cs
        x = x + x_cs

        return x


class CBAM(nn.Module):
    """
    https://arxiv.org/abs/1807.06521v2
    """
    def __init__(self, r, C):
        super(CBAM, self).__init__()
        self.r = r
        self.C = C
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.globalmaxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.dense1 = nn.Linear(self.C, self.C // self.r)
        self.dense2 = nn.Linear(self.C // self.r, self.C)
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size=7,
            stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_c_avg = self.globalavgpool(x)
        x_c_avg = x_c_avg.view((batch_size, -1))
        x_c_avg = self.dense1(x_c_avg)
        x_c_avg = nn.ReLU(inplace=True)(x_c_avg)
        x_c_avg = self.dense2(x_c_avg)
        x_c_max = self.globalmaxpool(x)
        x_c_max = x_c_max.view((batch_size, -1))
        x_c_max = self.dense1(x_c_max)
        x_c_max = nn.ReLU(inplace=True)(x_c_max)
        x_c_max = self.dense2(x_c_max)
        x_c = x_c_avg + x_c_max
        x_c = nn.Sigmoid()(x_c)
        x_c = x_c.view((batch_size, self.C, 1, 1))
        x = x * x_c

        x_s_avg = x.mean(axis=1, keepdim=True)
        x_s_max = x.max(axis=1, keepdim=True)[0]
        x_s = torch.cat((x_s_avg, x_s_max), axis=1)
        x_s = self.conv1(x_s)
        x_s = self.bn1(x_s)
        x_s = nn.Sigmoid()(x_s)
        x_s = x * x_s
        x = x + x_s

        return x_s
