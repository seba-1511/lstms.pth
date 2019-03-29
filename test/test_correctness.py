#!/usr/bin/env python

import torch as th
import torch.nn as nn
from torch.autograd import Variable as V

from lstms import SlowLSTM, LSTM, GalLSTM, MoonLSTM, SemeniutaLSTM


if __name__ == '__main__':
    lstms = [
        (SlowLSTM, 'SlowLSTM'),
        (LSTM, 'LSTM'),
    ]
    for lstm, name in lstms:
        th.manual_seed(1234)
        x = V(th.rand(1, 1, 256))
        hiddens = (V(th.rand(1, 1, 256)), V(th.rand(1, 1, 256)))
        ref = nn.LSTM(256, 256, bias=False, dropout=0.0)
        cus = lstm(256, 256, bias=False, dropout=0.0)

        # Make sure they have the same parameters:
        val = th.rand(1)[0]
        for c in cus.parameters():
            c.data.fill_(val)
        for r in ref.parameters():
            r.data.fill_(val)

        objective = V(th.zeros(1, 256))

        i, j = x.clone(), [h.clone() for h in hiddens]
        g, h = x.clone(), [h.clone() for h in hiddens]
        for _ in range(10):
            i, j = ref(i, j)
            g, h = cus(g, h)
            assert(th.equal(g.data, i.data))
            assert(th.equal(j[0].data, h[0].data))
            assert(th.equal(j[1].data, h[1].data))
            ref_loss = th.sum((i - objective)**2)
            cus_loss = th.sum((g - objective)**2)
            ref_loss.backward(retain_graph=True)
            cus_loss.backward(retain_graph=True)
        print('Correct: ', name)
    print('Test passed')
