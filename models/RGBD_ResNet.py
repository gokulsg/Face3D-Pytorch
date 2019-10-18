from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, in_planes, out_planes, pretrained=False):
        super(ResNet50, self).__init__()
        if pretrained is True:
            self.model = models.resnet50(pretrained=True)
            print("Pretrained model is loaded")
        else:
            self.model = models.resnet50(pretrained=False)
        if in_planes == 4:
            self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')
        # Parameters of newly constructed modules have requires_grad=True by default
        self.model.fc = nn.Linear(self.model.fc.in_features, out_planes)

    def forward(self, x):
        return self.model(x)

