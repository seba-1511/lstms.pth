# lstm.pth

Implementation of LSTM variants, in PyTorch. 

For now, they only support a sequence size of 1, and are ideal for RL use-cases. 
Besides that, they are a stripped-down version of PyTorch's RNN layers. 
(no bidirectional, no num_layers, no batch_first)

Base Models:

* SlowLSTM: a (mostly useless) pedagogic example.
* LayerNorm: Layer Normalization as in [Ba & al.](https://arxiv.org/pdf/1607.06450.pdf): *Layer Normalization*.

Dropout Models:

* LSTM: the original.
* GalLSTM: using dropout as in [Gal & Ghahramami](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf): *A Theoretically Grounded Application of Dropout in RNNs*.
* MoonLSTM: using dropout as in [Moon & al](https://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf): *RNNDrop: A Novel Dropout for RNNs in ASR*.
* SemeniutaLSTM: using dropout as in [Semeniuta & al](https://arxiv.org/pdf/1603.05118.pdf): *Recurrent Dropout without Memory Loss*.

Normalization + Dropout Models:

* LayerNormLSTM: Dropout + Layer Normalization.
* LayerNormGalLSTM: Gal Dropout + Layer Normalization.
* LayerNormMoonLSTM: Moon Dropout + Layer Normalization.
* LayerNormSemeniutaLSTM: Semeniuta Dropout + Layer Normalization.

**Convention:** If applicable, the activations are computed first, and **then** the nodes are droped. (dropout on the output, not the input, just like PyTorch)

## Install

`pip install -e .`

## Usage

You can find a good example of how to use the layers in [test/test_speed.py](./test/test_speed.py).

All Dropout models share the same signature:

```python
    LSTM(self, input_size, hidden_size, bias=True, dropout=0.0, dropout_method='pytorch')
```

All Normalization + Dropout models share the same signature:

```python

    LayerNormLSTM(self, input_size, hidden_size, bias=True, dropout=0.0, 
                 dropout_method='pytorch', ln_preact=True, learnable=True):
```

And all models use the same `.forward(x, hidden)`signature as the official PyTorch LSTM layers.

**Note:** `LayerNorm` is not an LSTM layer, and thus uses `.forward(x)`.

## Capacity Results

Available by running `make capacity`

Note: nn.LSTM does not have dropout in these experiments, as we are dealing with a single LSTM layer.

Info: dropout =  0.9, SEQ_LEN =  10, dataset size =  100, layer size =  256.

Warning: This is an artificial memory benchmark, not necessarily representative of each method's capacity.
 
model          |         error
---------------|--------------
nn.LSTM        | 3.515
SlowLSTM       | 4.171
LSTM           | 4.158
GalLSTM        | 3.517
MoonLSTM       | 4.443
SemeniutaLSTM  | 3.773

## Speed Results

Available by running `make speed`.

Warning: Inference timings only, and on a single sequence of length 1000  with `dropout =  0.5 `.

###  SlowLSTM  Benchmark 
 
 
size   | nn.LSTM   |  SlowLSTM  | Speedup
-------|-----------|------------|--------
128    | 0.757     | 0.745      | 1.017
256    | 0.583     | 0.800      | 0.729
512    | 0.872     | 1.099      | 0.793
1024    | 3.658     | 3.969      | 0.922
2048    | 10.077     | 11.628      | 0.867
 
###  LSTM  Benchmark 
 
size   | nn.LSTM   |  LSTM  | Speedup
-------|-----------|--------|--------
128    | 0.725     | 0.379  | 1.913
256    | 0.615     | 0.511  | 1.202
512    | 0.835     | 0.610  | 1.369
1024    | 3.485     | 3.547  | 0.982
2048    | 11.066     | 7.762  | 1.426
 
###  GalLSTM  Benchmark 
 
size   | nn.LSTM   |  GalLSTM  | Speedup
-------|-----------|-----------|--------
128    | 0.669     | 0.696     | 0.961
256    | 0.579     | 0.652     | 0.887
512    | 0.966     | 0.885     | 1.091
1024    | 3.630     | 3.441     | 1.055
2048    | 9.598     | 9.509     | 1.009
 
###  MoonLSTM  Benchmark 
 
size   | nn.LSTM   |  MoonLSTM  | Speedup
-------|-----------|------------|--------
128    | 0.656     | 0.551      | 1.190
256    | 0.652     | 0.589      | 1.108
512    | 0.949     | 0.614      | 1.546
1024    | 3.646     | 3.020      | 1.207
2048    | 10.202     | 8.539      | 1.195
 
###  SemeniutaLSTM  Benchmark 
 
size   | nn.LSTM   |  SemeniutaLSTM  | Speedup
-------|-----------|-----------------|--------
128    | 0.580     | 0.375           | 1.548
256    | 0.718     | 0.843           | 0.852
512    | 0.859     | 0.574           | 1.496
1024    | 4.041     | 3.372           | 1.198
2048    | 9.587     | 8.668           | 1.106

