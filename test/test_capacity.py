#!/usr/bin/env python

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.optim import Adam, SGD

from lstms import (SlowLSTM,
                   LSTM,
                   GalLSTM,
                   MoonLSTM,
                   SemeniutaLSTM,
                   LayerNormLSTM,
                   LayerNormGalLSTM,
                   LayerNormMoonLSTM,
                   LayerNormSemeniutaLSTM,
                   )

"""
An artificial memory benchmark, not necessarily representative of each method's capacity.
"""

DS_SIZE = 100
SEQ_LEN = 10
NUM_EPOCHS = 10
DROPOUT = 0.9
LR = 0.01
MOMENTUM = 0.9
BSZ = 10
SIZE = 256


if __name__ == '__main__':
    hiddens = (V(th.rand(1, 1, SIZE)), V(th.rand(1, 1, SIZE)))
    lstms = [
        (nn.LSTM, 'nn.LSTM'),
        (SlowLSTM, 'SlowLSTM'),
        (LSTM, 'LSTM'),
        (GalLSTM, 'GalLSTM'),
        (MoonLSTM, 'MoonLSTM'),
        (SemeniutaLSTM, 'SemeniutaLSTM'),
        (LayerNormGalLSTM, 'GalLSTM'),
        (LayerNormMoonLSTM, 'MoonLSTM'),
        (LayerNormSemeniutaLSTM, 'SemeniutaLSTM'),
    ]
    results = []
    for lstm, name in lstms:
        print('Benching: ', name)
        th.manual_seed(1234)
        lstm = lstm(SIZE, SIZE, dropout=DROPOUT)
        opt = SGD(lstm.parameters(), lr=LR, momentum=MOMENTUM)
        loss = F.smooth_l1_loss

        inputs = [[th.rand(1, 1, SIZE) for l in range(SEQ_LEN)] for _ in range(DS_SIZE)]
        labels = [sum(s) for s in inputs]

        for epoch in range(NUM_EPOCHS):
            print(' ')
            print('*'*20, ' Epoch ', epoch, ' ', '*'*20)
            total_error = 0.0
            error = 0.0
            for idx, (seq, y) in enumerate(zip(inputs, labels)):
                if hasattr(lstm, 'sample_mask'):
                    lstm.sample_mask()
                y = V(y)
                h = hiddens
                out = 0.0
                for x in seq:
                    x = V(x)
                    out, h = lstm(x, h)
                error += loss(out, y)
                if (1 + idx) % BSZ == 0:
                    # print('    Batch Error: ', error.data[0] / BSZ)
                    total_error += error.item()
                    opt.zero_grad()
                    error.backward()
                    opt.step()
                    error = 0.0
            print('Average Error: ', total_error / DS_SIZE)
            print(' ')

        total_error = 0.0
        for seq, y in zip(inputs, labels):
                y = V(y, volatile=True)
                h = hiddens
                for x in seq:
                    x = V(x, volatile=True)
                    out, h = lstm(x, h)
                total_error += loss(y, out).item()
        print('Error: ', total_error / DS_SIZE)
        results.append([name, total_error / DS_SIZE])

    print(' ')
    print('## Summary: ')
    print('Note: nn.LSTM does not have dropout in these experiments, as we are dealing with a single LSTM layer.')
    print('Info: dropout = ', DROPOUT, ', SEQ_LEN = ', SEQ_LEN, ', dataset size = ', DS_SIZE, ' layer size = ', SIZE)
    print(' ')
    print('model          |         error')
    print('---------------|--------------')
    for name, score in results:
        print(name + ' '*(14-len(name)), '| %.3f' % (score, ))

