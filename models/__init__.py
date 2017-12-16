# -*- coding: utf-8 -*-

from .resnet import *

__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152
}


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError('Unknown model')
    __factory[name](*args, **kwargs)