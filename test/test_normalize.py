#!/usr/bin/env python

import torch as th
from torch import Tensor as T
from torch.autograd import Variable as V
from lstms import LayerNorm, LSTM, LayerNormLSTM

if __name__ == '__main__':
    th.manual_seed(1234)
    # vec = T(1, 1, 5).fill_(1.0)
    vec = V(th.rand(1, 1, 5))
    ln = LayerNorm(5, learnable=False)
    print(ln(vec))

    ln = LayerNorm(5, learnable=True)
    print(ln(vec))
    for p in ln.parameters():
        print(p)

    lstm = LSTM(5, 5)
    asdf, (a, e) = lstm(ln(vec), (vec, vec))
    print(asdf)



