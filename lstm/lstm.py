#!/usr/bin/env python

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V

"""
Implementation of LSTM variants.

For now, they only support a batch-size of 1, and are ideal for RL use-cases. 
Besides that, they should be compatible with the other PyTorch RNN layers.
"""

"""
TODO:
    * Benchmark which works best.
"""



class SlowLSTM(nn.Module):

    """
    A pedagogic implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0):
        super(SlowLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        # input to hidden weights
        self.w_xi = P(T(hidden_size, input_size))
        self.w_xf = P(T(hidden_size, input_size))
        self.w_xo = P(T(hidden_size, input_size))
        self.w_xc = P(T(hidden_size, input_size))
        # hidden to hidden weights
        self.w_hi = P(T(hidden_size, hidden_size))
        self.w_hf = P(T(hidden_size, hidden_size))
        self.w_ho = P(T(hidden_size, hidden_size))
        self.w_hc = P(T(hidden_size, hidden_size))
        # bias terms
        self.b_i = T(hidden_size).fill_(0)
        self.b_f = T(hidden_size).fill_(0)
        self.b_o = T(hidden_size).fill_(0)
        self.b_c = T(hidden_size).fill_(0)

        # Wrap biases as parameters if desired, else as variables without gradients
        if bias:
            W = P
        else:
            W = V
        self.b_i = W(self.b_i)
        self.b_f = W(self.b_f)
        self.b_o = W(self.b_o)
        self.b_c = W(self.b_c)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(h.size(0), -1)
        x = x.view(x.size(0), -1)
        # Linear mappings
        i_t = th.mm(x, self.w_xi) + th.mm(h, self.w_hi) + self.b_i
        f_t = th.mm(x, self.w_xf) + th.mm(h, self.w_hf) + self.b_f
        o_t = th.mm(x, self.w_xo) + th.mm(h, self.w_ho) + self.b_o
        # activations
        i_t.sigmoid_()
        f_t.sigmoid_()
        o_t.sigmoid_()
        # cell computations
        c_t = th.mm(x, self.w_xc) + th.mm(h, self.w_hc) + self.b_c
        c_t.tanh_()
        c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
        h_t = th.mul(o_t, th.tanh(c_t))
        # Reshape for compatibility
        h_t = h_t.view(h_t.size(0), 1, -1)
        c_t = c_t.view(c_t.size(0), 1, -1)
        if self.dropout > 0.0:
            F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
        return h_t, (h_t, c_t)

    def sample_mask(self):
        pass


class LSTM(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(hidden_size, 4*input_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4*input_size, bias=bias)
        self.reset_parameters()

    def sample_mask(self):
        pass

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(c.size(0), -1)
        x = x.view(x.size(0), -1)
        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)
        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size] 
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]
        # cell computations
        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)
        h_t = th.mul(o_t, c_t.tanh())
        # Reshape for compatibility
        h_t = h_t.view(h_t.size(0), 1, -1)
        c_t = c_t.view(c_t.size(0), 1, -1)
        if self.dropout > 0.0:
            F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
        return h_t, (h_t, c_t)


class GalLSTM(nn.Module):

    """
    Implementation of Gal & Ghahramami:
    'A Theoretically Grounded Application of Dropout in Recurrent Neural Networks'
    http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, lstm=None):
        super(GalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if lstm is None:
            lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bias=bias)
        self.lstm = lstm
        self.dropout = dropout
        self.sample_mask()

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def forward(self, x, hidden):
        if self.dropout > 0.0:
            if self.training:
                hidden[0].data.set_(th.mul(hidden[0], self.mask).data)
                hidden[0].data *= 1.0/(1.0 - self.dropout)
        out, hidden = self.lstm.forward(x, hidden)
        return out, hidden


class MoonLSTM(LSTM):

    """
    Implementation of Moon & al.:
    'RNNDrop: A Novel Dropout for RNNs in ASR'
    https://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf
    """
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0):
        super(MoonLSTM, self).__init__(input_size, hidden_size, bias, dropout)
        self.sample_mask()

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(h.size(0), -1)
        x = x.view(x.size(0), -1)
        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)
        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        c_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size] 
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]
        # cell computations
        c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
        if self.dropout > 0.0:
            if self.training:
                c_t.data.set_(th.mul(c_t, self.mask).data)
                c_t.data *= 1.0/(1.0 - self.dropout)
        h_t = th.mul(o_t, c_t.tanh())
        # Reshape for compatibility
        h_t = h_t.view(h_t.size(0), 1, -1)
        c_t = c_t.view(c_t.size(0), 1, -1)
        return h_t, (h_t, c_t)


class SemeniutaLSTM(LSTM):
    """
    Implementation of Semeniuta & al.:
    'Recurrent Dropout without Memory Loss'
    https://arxiv.org/pdf/1603.05118.pdf
    """
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.5):
        super(SemeniutaLSTM, self).__init__(input_size, hidden_size, bias, dropout)
        self.sample_mask()

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(h.size(0), -1)
        x = x.view(x.size(0), -1)
        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)
        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        c_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size] 
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]
        # cell computations
        if self.dropout > 0.0:
            c_t = F.dropout(c_t, p=self.dropout, training=self.training)
        c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
        h_t = th.mul(o_t, c_t.tanh())
        # Reshape for compatibility
        h_t = h_t.view(h_t.size(0), 1, -1)
        c_t = c_t.view(c_t.size(0), 1, -1)
        return h_t, (h_t, c_t)
