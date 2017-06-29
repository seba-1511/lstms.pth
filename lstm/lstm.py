#!/usr/bin/env python

import math
import torch as th
import torch.nn as nn
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.input_weights = None
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.input_weights = P(th.rand(4 * hidden_size, input_size))
        self.hidden_weights = P(th.rand(4 * hidden_size, hidden_size))
        self.input_bias = P(th.rand(4 * hidden_size))
        self.hidden_bias = P(th.rand(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, c = hidden
        out = th.mm(self.input_weights, x) + th.mm(self.hidden_weights, h)
        return self.lstm(x, hidden)


class GalLSTM(nn.Module):

    """
    Implementation of Gal & Ghahramami:
    'A Theoretically Grounded Application of Dropout in Recurrent Neural Networks'
    http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
    """

    def __init__(self, input_size, hidden_size, dropout=0.5, bias=True):
        super(GalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.lstm = nn.LSTM(input_size, hidden_size, bias)
        self.dropout = dropout
        self.reset()

    def reset(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def forward(self, x, hidden):
        if hidden[0].size(0) > 1:
            hidden[0].data.set_(th.mul(hidden[0], self.mask.repeat(hidden[0].size(0), 1)).data)
        else:
            hidden[0].data.set_(th.mul(hidden[0], self.mask).data)
        out, hidden = self.lstm(x, hidden)
        return out, hidden


class MoonLSTM(nn.Module):
    pass


class SemenuitaLSTM(nn.Module):
    pass
