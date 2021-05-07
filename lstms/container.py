#!/usr/bin/env python
"""
A helper class to contruct multi-layered LSTMs.
"""
import math
from typing import List, Tuple, Type

import torch as th
import torch.nn as nn

from .lstm import LSTM


class MultiLayerLSTM(nn.Module):

    """
    MultiLayer LSTM of any type.

    Note: Dropout is deactivated on the last layer.
    """

    def __init__(self, input_size: int, layer_type: Type[LSTM], layer_sizes: List[int] = (64, 64), *args, **kwargs):
        super(MultiLayerLSTM, self).__init__()
        rnn = layer_type
        self.layers: List[LSTM] = []
        prev_size = input_size
        for size in layer_sizes[:-1]:
            layer = rnn(input_size=prev_size, hidden_size=size, *args, **kwargs)
            self.layers.append(layer)
            prev_size = size
        if "dropout" in kwargs:
            del kwargs["dropout"]
        if len(layer_sizes) > 0:
            layer = rnn(input_size=prev_size, hidden_size=layer_sizes[-1], dropout=0.0, *args, **kwargs)
            self.layers.append(layer)
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.params = nn.ModuleList(self.layers)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def create_hiddens(self, batch_size: int = 1) -> List[Tuple[th.Tensor, th.Tensor]]:
        # Uses Xavier init here.
        hiddens: List[Tuple[th.Tensor, th.Tensor]] = []
        for layer in self.layers:
            std = math.sqrt(2.0 / (layer.input_size + layer.hidden_size))
            hiddens.append(
                (
                    th.empty(1, batch_size, layer.hidden_size).normal_(0, std),
                    th.empty(1, batch_size, layer.hidden_size).normal_(0, std),
                )
            )
        return hiddens

    def sample_mask(self):
        for layer in self.layers:
            layer.sample_mask()

    def forward(
        self, x: th.Tensor, hiddens: Tuple[th.Tensor, th.Tensor]
    ) -> Tuple[th.Tensor, List[Tuple[th.Tensor, th.Tensor]]]:
        new_hiddens: List[Tuple[th.Tensor, th.Tensor]] = []
        for layer, h in zip(self.layers, hiddens):
            x, new_h = layer(x, h)
            new_hiddens.append(new_h)
        return x, new_hiddens
