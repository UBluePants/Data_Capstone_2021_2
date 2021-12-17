import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List

class Resnet34FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet34 = models.resnet34(pretrained=True)
        layers : List[nn.Module] = []
        layers = layers + [resnet34.conv1]
        layers = layers + [resnet34.bn1]
        layers = layers + [resnet34.relu]
        layers = layers + [resnet34.layer1]
        layers = layers + [resnet34.layer2[0]]
        self.enc_1 = nn.Sequential(*layers)
        layers : List[nn.Module] = []
        layers = layers + [resnet34.layer2[1:]]
        layers = layers + [resnet34.layer3[:2]]
        self.enc_2 = nn.Sequential(*layers)
        layers : List[nn.Module] = []
        layers = layers + [resnet34.layer3[2:]]
        layers = layers + [resnet34.layer4]
        self.enc_3 = nn.Sequential(*layers)

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
