#!/usr/bin/env python
"""
A helper class to contruct multi-layered LSTMs.
"""
import math

import torch as th
import torch.nn as nn
from torch.autograd import Variable


class MultiLayerLSTM(nn.Module):

    """
    MultiLayer LSTM of any type.

    Note: Dropout is deactivated on the last layer.
    """

    def __init__(self, input_size, layer_type, layer_sizes=(64, 64), *args, **kwargs):
        super(MultiLayerLSTM, self).__init__()
        rnn = layer_type
        layers = []
        prev_size = input_size
        for size in layer_sizes[:-1]:
            layer = rnn(input_size=prev_size, hidden_size=size, *args, **kwargs)
            layers.append(layer)
            prev_size = size
        if "dropout" in kwargs:
            del kwargs["dropout"]
        layer = rnn(input_size=prev_size, hidden_size=layer_sizes[-1], dropout=0.0, *args, **kwargs)
        layers.append(layer)
        self.layers = layers
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.params = nn.ModuleList(layers)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def create_hiddens(self, batch_size=1):
        # Uses Xavier init here.
        hiddens = []
        for layer in self.layers:
            std = math.sqrt(2.0 / (layer.input_size + layer.hidden_size))
            hiddens.append(
                [
                    Variable(th.empty(1, batch_size, layer.hidden_size).normal_(0, std)),
                    Variable(th.empty(1, batch_size, layer.hidden_size).normal_(0, std)),
                ]
            )
        return hiddens

    def sample_mask(self):
        for layer in self.layers:
            layer.sample_mask()

    def forward(self, x, hiddens):
        new_hiddens = []
        for layer, h in zip(self.layers, hiddens):
            x, new_h = layer(x, h)
            new_hiddens.append(new_h)
        return x, new_hiddens
