#!/usr/bin/env python

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V

"""
Implementation of various normalization techniques. Also only works on instances
where batch size = 1.

"""


class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size, learnable=True, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = T(1, input_size).fill_(0)
        self.beta = T(1, input_size).fill_(0)
        self.epsilon = epsilon
        # Wrap as parameters if necessary
        if learnable:
            W = P
        else:
            W = V
        self.alpha = W(self.alpha)
        self.beta = W(self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - th.mean(x, 1).expand_as(x)) / th.sqrt(th.var(x, 1).expand_as(x) + self.epsilon)
        if self.learnable:
            x = self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)


class BradburyLayerNorm(nn.Module):

    """
    Layer Norm, according to:
    https://github.com/pytorch/pytorch/issues/1959#issuecomment-312364139
    """

    def __init__(self, features, eps=1e-6):
        super(BradburyLayerNorm, self).__init__()
        self.gamma = nn.Parameter(th.ones(features))
        self.beta = nn.Parameter(th.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class BaLayerNorm(nn.Module):

    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf

    This implementation mimicks the original torch implementation at:
    https://github.com/ryankiros/layer-norm/blob/master/torch_modules/LayerNormalization.lua
    """
    def __init__(self, input_size, learnable=True, epsilon=1e-5):
        super(BaLayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.epsilon = epsilon
        self.alpha = T(1, input_size).fill_(0)
        self.beta = T(1, input_size).fill_(0)
        # Wrap as parameters if necessary
        if learnable:
            W = P
        else:
            W = V
        self.alpha = W(self.alpha)
        self.beta = W(self.beta)

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), -1)
        mean = th.mean(x, 1).expand_as(x)
        center = x - mean
        std = th.sqrt(th.mean(th.square(center), 1)).expand_as(x)
        output = center / (std + self.epsilon)
        if self.learnable:
            output = self.alpha * output + self.beta
        return output.view(size)
