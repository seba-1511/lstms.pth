# lstms.pth

Implementation of LSTM variants, in PyTorch. 

For now, they only support a sequence size of 1, and meant for RL use-cases. 
Besides that, they are a stripped-down version of PyTorch's RNN layers. 
(no bidirectional, no num_layers, no batch_first)

Base Modules:

* SlowLSTM: a (mostly useless) pedagogic example.
* LayerNorm: Layer Normalization as in [Ba & al.](https://arxiv.org/pdf/1607.06450.pdf): *Layer Normalization*.

Dropout Modules:

* LSTM: the original.
* GalLSTM: using dropout as in [Gal & Ghahramami](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf): *A Theoretically Grounded Application of Dropout in RNNs*.
* MoonLSTM: using dropout as in [Moon & al](https://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf): *RNNDrop: A Novel Dropout for RNNs in ASR*.
* SemeniutaLSTM: using dropout as in [Semeniuta & al](https://arxiv.org/pdf/1603.05118.pdf): *Recurrent Dropout without Memory Loss*.

Normalization + Dropout Modules:

* LayerNormLSTM: Dropout + Layer Normalization.
* LayerNormGalLSTM: Gal Dropout + Layer Normalization.
* LayerNormMoonLSTM: Moon Dropout + Layer Normalization.
* LayerNormSemeniutaLSTM: Semeniuta Dropout + Layer Normalization.

Container Modules:

* MultiLayerLSTM: helper class to build multiple layers LSTMs.

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

And all models use the same `out, hidden = model.forward(x, hidden)`signature as the official PyTorch LSTM layers. They also all provide a `model.sample_mask()` method, which needs to be called in order to sample a new Dropout mask. (e.g, when processing a new sequence)

**Note:** `LayerNorm` is not an LSTM layer, and thus uses `out = model.forward(x)`.

### Containers

This package provides a helper class, `MultiLayerLSTM`, which can be use to stack multiple LSTMs together.

```python
lstm = MultiLayerLSTM(input_size=256, layer_type=LayerNormSemeniutaLSTM,
                      layer_sizes=(64, 64, 16), dropout=0.7, ln_preact=False)
hiddens = lstm.create_hiddens(bsz=batch_size)
x = Variable(th.rand(1, 1, 256))
for _ in range(10):
    out, hiddens = lstm(x, hiddens)
```

Note that `hiddens` doesn't match the PyTorch specification. It is the list of `(h_i, c_i)` for each LSTM layer. Instead, the `LSTM` layers in PyTorch return a single tuple of `(h_n, c_n)`, where `h_n` and `c_n` have sizes (num_layers * num_directions, batch, hidden_size).

## Capacity Benchmarks

Warning: This is an artificial memory benchmark, not necessarily representative of each method's capacity.

Note: nn.LSTM and SlowLSTM do not have dropout in these experiments.

Info: dropout =  0.9 , SEQ_LEN =  10 , dataset size =  100  layer size =  256
 
 model          |         error
 ---------------|--------------
 nn.LSTM        | 3.515
 SlowLSTM       | 4.171
 LSTM           | 4.160
 GalLSTM        | 4.456
 MoonLSTM       | 4.442
 SemeniutaLSTM  | 3.762
 GalLSTM        | 4.456
 MoonLSTM       | 4.442
 SemeniutaLSTM  | 3.762


## Speed Benchmarks

Available by running `make speed`.

Warning: Inference timings only, and on a single sequence of length 1000  with `dropout =  0.5 `.

##  SlowLSTM  Benchmark 

size   | nn.LSTM   |  SlowLSTM  | Speedup
-------|-----------|------------|--------
128    | 0.628     | 0.666      | 0.943
256    | 0.676     | 0.759      | 0.890
512    | 0.709     | 1.026      | 0.690
1024    | 2.364     | 2.867      | 0.824
2048    | 6.161     | 8.261      | 0.746

##  LSTM  Benchmark 

size   | nn.LSTM   |  LSTM  | Speedup
-------|-----------|--------|--------
128    | 0.568     | 0.387  | 1.466
256    | 0.668     | 0.419  | 1.594
512    | 0.803     | 0.769  | 1.045
1024    | 2.966     | 2.002  | 1.482
2048    | 6.291     | 6.393  | 0.984

##  GalLSTM  Benchmark 

size   | nn.LSTM   |  GalLSTM  | Speedup
-------|-----------|-----------|--------
128    | 0.557     | 0.488     | 1.142
256    | 0.683     | 0.446     | 1.530
512    | 0.966     | 0.548     | 1.763
1024    | 2.524     | 2.587     | 0.975
2048    | 6.618     | 6.099     | 1.085

##  MoonLSTM  Benchmark 

size   | nn.LSTM   |  MoonLSTM  | Speedup
-------|-----------|------------|--------
128    | 0.667     | 0.445      | 1.499
256    | 0.818     | 0.535      | 1.530
512    | 0.908     | 0.695      | 1.306
1024    | 2.517     | 2.553      | 0.986
2048    | 6.475     | 6.779      | 0.955

##  SemeniutaLSTM  Benchmark 

size   | nn.LSTM   |  SemeniutaLSTM  | Speedup
-------|-----------|-----------------|--------
128    | 0.692     | 0.513           | 1.348
256    | 0.685     | 0.697           | 0.983
512    | 0.717     | 0.701           | 1.022
1024    | 2.639     | 2.751           | 0.959
2048    | 7.294     | 6.122           | 1.191

##  LayerNormLSTM  Benchmark 

size   | nn.LSTM   |  LayerNormLSTM  | Speedup
-------|-----------|-----------------|--------
128    | 0.646     | 1.656           | 0.390
256    | 0.583     | 1.800           | 0.324
512    | 0.770     | 1.989           | 0.387
1024    | 2.623     | 3.844           | 0.682
2048    | 6.573     | 9.592           | 0.685

##  LayerNormGalLSTM  Benchmark 

size   | nn.LSTM   |  LayerNormGalLSTM  | Speedup
-------|-----------|--------------------|--------
128    | 0.566     | 0.486              | 1.163
256    | 0.592     | 0.350              | 1.693
512    | 0.920     | 0.606              | 1.517
1024    | 2.508     | 2.427              | 1.034
2048    | 7.356     | 10.268              | 0.716

##  LayerNormMoonLSTM  Benchmark 

size   | nn.LSTM   |  LayerNormMoonLSTM  | Speedup
-------|-----------|---------------------|--------
128    | 0.507     | 0.389               | 1.305
256    | 0.685     | 0.511               | 1.342
512    | 0.762     | 0.685               | 1.111
1024    | 2.661     | 2.261               | 1.177
2048    | 8.904     | 9.710               | 0.917

##  LayerNormSemeniutaLSTM  Benchmark 

size   | nn.LSTM   |  LayerNormSemeniutaLSTM  | Speedup
-------|-----------|--------------------------|--------
128    | 0.492     | 0.388                    | 1.267
256    | 0.583     | 0.360                    | 1.616
512    | 0.760     | 0.578                    | 1.316
1024    | 2.586     | 2.328                    | 1.111
2048    | 6.970     | 10.725                    | 0.650
