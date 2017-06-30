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
    lstm = MultiLayerLSTM(2*SIZE, LSTM)
    l = LayerNormLSTM(2*SIZE, SIZE)
    h = (V(th.rand(1, SIZE, SIZE)), V(th.rand(1, SIZE, SIZE)))
    # h = (V(th.rand(1, 1, SIZE)), V(th.rand(1, 1, SIZE)))

    # x = V(th.rand(1, 1, 2*SIZE))
    x = V(th.rand(1, SIZE, 2*SIZE))
    hiddens = lstm.create_hiddens(bsz=SIZE)

    out, h = x, hiddens
    # out, h = x, h
    for i in range(10):
        # out, h = l.forward(x, h)
        print(i)
        out, h = lstm(x, h)
