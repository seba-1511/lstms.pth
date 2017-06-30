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
                  )


if __name__ == '__main__':
    N_ITER = 1000
    SIZES = [128, 256, 512, 1024, 2048]
    DROPOUT = 0.5
    lstms = [
        (SlowLSTM, 'SlowLSTM'),
        (LSTM, 'LSTM'),
        (GalLSTM, 'GalLSTM'),
        (MoonLSTM, 'MoonLSTM'),
        (SemeniutaLSTM, 'SemeniutaLSTM'),
        (LayerNormLSTM, 'LayerNormLSTM'),
        (LayerNormGalLSTM, 'LayerNormGalLSTM'),
        (LayerNormMoonLSTM, 'LayerNormMoonLSTM'),
        (LayerNormSemeniutaLSTM, 'LayerNormSemeniutaLSTM'),
    ]

    for lstm, name in lstms:
        ref_res = []
        cus_res = []
        for size in SIZES:
            x = V(th.rand(1, 1, size))
            hiddens = (V(th.rand(1, 1, size)), V(th.rand(1, 1, size)))
            th.manual_seed(1234)
            ref = nn.LSTM(size, size, dropout=DROPOUT)
            th.manual_seed(1234)
            cus = lstm(size, size, dropout=DROPOUT)

            out, h = x, hiddens
            ref_start = time()
            for i in range(N_ITER):
                out, h = ref(out, h)
            ref_time = time() - ref_start
            ref_res.append(ref_time)

            out, h = x, hiddens
            cus_start = time()
            for i in range(N_ITER):
                out, h = cus(out, h)
            cus_time = time() - cus_start
            cus_res.append(cus_time)

        print('## ', name, ' Benchmark ')
        print(' ')
        print('Inference timings on a single sequence of length', N_ITER, ' with `dropout = ', DROPOUT, '`.')
        print(' ')
        print('size   | nn.LSTM   | ', name, ' | Speedup')
        print('-------|-----------|-' + '-' * len(name) + '---|--------')
        for size, ref, cus in zip(SIZES, ref_res, cus_res):
            print(size, ('   | %.3f     | %.3f' + ' ' * (len(name) - 4) + '  | %.3f') % (ref, cus, ref / cus))
        print(' ')
