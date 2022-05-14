import torch
from torch import nn
from torchvision import models


class PipeNet(nn.Module):
    def __init__(self, num_class):
        super(PipeNet, self). __init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*2*2, num_class if num_class > 2 else 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.gap(x)
        emb = nn.Flatten(1)(x)
        logits = self.fc(emb)

        # Normalize emb
        emb_mag = torch.norm(emb, p=2, dim=1, keepdim=True).detach()
        emb_norm = emb.div(emb_mag.expand_as(emb))

        return emb_norm, logits


class ResNet18(nn.Module):
    def __init__(self, num_class):
        super(ResNet18, self).__init__()
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(
            512, num_class if num_class > 2 else 1)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        emb = torch.flatten(x, 1)
        logits = self.fc(emb)

        # Normalize emb
        emb_mag = torch.norm(emb, p=2, dim=1, keepdim=True).detach()
        emb_norm = emb.div(emb_mag.expand_as(emb))
        return emb_norm, logits
