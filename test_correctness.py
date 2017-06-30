#!/usr/bin/env python

import torch as th
import torch.nn as nn
from torch.autograd import Variable as V

from lstm import LSTM, GalLSTM, MoonLSTM, SemeniutaLSTM


if __name__ == '__main__':
    x = V(th.rand(1, 1, 256))
    hiddens = (V(th.rand(1, 1, 256)), V(th.rand(1, 1, 256)))
    th.manual_seed(1234)
    ref = nn.LSTM(256, 256)
    th.manual_seed(1234)
    cus = GalLSTM(256, 256, dropout=0.0)

    mask = cus.mask
    for i in range(10):
        label, g = ref(x, hiddens)
        pred, h = cus(x, hiddens)
        assert(th.equal(pred.data, label.data))
        assert(th.equal(g[0].data, h[0].data))
        assert(th.equal(g[1].data, h[1].data))
        assert(th.equal(mask.data, cus.mask.data))
        hiddens = g
        x = pred
        mask = cus.mask


    x = V(th.rand(1, 1, 256))
    hiddens = (V(th.rand(1, 1, 256)), V(th.rand(1, 1, 256)))
    th.manual_seed(1234)
    ref = LSTM(256, 256)
    th.manual_seed(1234)
    # cus = GalLSTM(256, 256, dropout=0.0)
    # cus = MoonLSTM(256, 256, dropout=0.0)
    cus = SemeniutaLSTM(256, 256, dropout=0.0)

    # mask = cus.mask
    for i in range(10):
        label, g = ref(x, hiddens)
        pred, h = cus(x, hiddens)
        assert(th.equal(pred.data, label.data))
        assert(th.equal(g[0].data, h[0].data))
        assert(th.equal(g[1].data, h[1].data))
        hiddens = g
        x = pred
        # mask = cus.mask

    print('Test passed')
