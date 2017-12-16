# -*- coding: utf-8 -*-
from torch import nn
import torchvision
from torch.nn import init
from torch.nn import functional as F
from torch import autograd
import torch


class ResNet(nn.Module):
    __factor = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False, num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()
        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        if depth not in ResNet.__factor:
            raise KeyError('Unsupported depth network')

        self.base = ResNet.__factor[depth](pretrained=pretrained)

        if not cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_plans = self.base.fc.in_features # size
            print('s',out_plans)

            if self.has_embedding: # embedding space
                self.feat = nn.Linear(out_plans, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan.out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat.bn.bias, 0)
            else:
                self.num_features = out_plans

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)

            if self.num_classes:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)   # reshape

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)

        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)

        if self.dropout > 0:
            x = self.drop(x)

        if self.num_classes > 0:
            x = self.classifier(x)

        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


if __name__ == '__main__':
    a = resnet50()
    b = ResNet(50)
    x = autograd.Variable(torch.randn(100, 3, 256, 256))
    b.forward(x)