#!/usr/bin/env python

import torch as th
import torch.nn as nn
from torch.autograd import Variable as V

from time import time

from lstms import (SlowLSTM,
                  LSTM,
                  GalLSTM,
                  MoonLSTM,
                  SemeniutaLSTM,
                  LayerNormLSTM,
                  LayerNormGalLSTM,
                  LayerNormMoonLSTM,
                  LayerNormSemeniutaLSTM,
                  MultiLayerLSTM,
                  )


SIZE = 16
if __name__ == '__main__':
    lstm = MultiLayerLSTM(SIZE, LSTM)
    l = LSTM(16, 32)
    h = (V(th.rand(1, 1, 32)), V(th.rand(1, 1, 32)))

    x = V(th.rand(1, 1, SIZE))
    hiddens = lstm.create_hiddens()

    out, h = x, hiddens
    for i in range(10):
        out, h = lstm(x, h)
