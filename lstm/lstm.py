#!/usr/bin/env python

import math
import torch as th
import torch.nn as nn
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V

"""
Implementation of LSTM variants.

For now, they only support a batch-size of 1, and are ideal for RL use-cases. 
Besides that, they should be compatible with the other PyTorch RNN layers.
"""


class LSTM(nn.Module):

    """
    A pedagogic implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0):
        super(LSTM, self).__init__()
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
        if bias:
            self.b_i = P(self.b_i)
            self.b_f = P(self.b_f)
            self.b_o = P(self.b_o)
            self.b_c = P(self.b_c)

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
        o_t = o_t.view(o_t.size(0), 1, -1)
        h_t = h_t.view(h_t.size(0), 1, -1)
        c_t = c_t.view(c_t.size(0), 1, -1)
        return o_t, (h_t, c_t)

    def reset(self):
        pass


class GalLSTM(nn.Module):

    """
    Implementation of Gal & Ghahramami:
    'A Theoretically Grounded Application of Dropout in Recurrent Neural Networks'
    http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.5, lstm=None):
        super(GalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if lstm is None:
            lstm = nn.LSTM(input_size, hidden_size, bias)
        self.lstm = lstm
        self.dropout = dropout
        self.reset()

    def reset(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def forward(self, x, hidden):
        hidden[0].data.set_(th.mul(hidden[0], self.mask).data)
        out, hidden = self.lstm(x, hidden)
        return out, hidden


class MoonLSTM(LSTM):

    """
    Implementation of Moon & al.:
    'RNNDrop: A Novel Dropout for RNNs in ASR'
    https://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf
    """
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.5):
        super(MoonLSTM, self).__init__(input_size, hidden_size, bias, dropout)
        self.reset()

    def reset(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(c.size(0), -1)
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
        c_t = self.mask * c_t
        h_t = th.mul(o_t, th.tanh(c_t))
        # Reshape for compatibility
        o_t = o_t.view(o_t.size(0), 1, -1)
        h_t = h_t.view(h_t.size(0), 1, -1)
        c_t = c_t.view(c_t.size(0), 1, -1)
        return o_t, (h_t, c_t)


class SemeniutaLSTM(LSTM):
    """
    Implementation of Semeniuta & al.:
    'Recurrent Dropout without Memory Loss'
    https://arxiv.org/pdf/1603.05118.pdf
    """
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.5):
        super(SemeniutaLSTM, self).__init__(input_size, hidden_size, bias, dropout)
        self.reset()

    def reset(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(c.size(0), -1)
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
        c_t = self.mask * c_t
        c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
        h_t = th.mul(o_t, th.tanh(c_t))
        # Reshape for compatibility
        o_t = o_t.view(o_t.size(0), 1, -1)
        h_t = h_t.view(h_t.size(0), 1, -1)
        c_t = c_t.view(c_t.size(0), 1, -1)
        return o_t, (h_t, c_t)




class NormalizedLSTM(LSTM):
    """
    Implementation of LSTM with Tensor Normalization.
    """
    pass
