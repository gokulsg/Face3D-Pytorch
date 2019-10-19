from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from torchvision import models


def ResNet50(in_planes, out_planes, pretrained=False):
    if pretrained is True:
        model = models.resnet50(pretrained=True)
        print("Pretrained model is loaded")
    else:
        model = models.resnet50(pretrained=False)
    if in_planes == 4:
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(model.fc.in_features, out_planes)
    return model