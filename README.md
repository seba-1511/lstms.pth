# lstm.pth

Implementation of LSTM variants, in PyTorch. 

For now, they only support a batch-size of 1, and are ideal for RL use-cases. 
Besides that, they should be compatible with the other PyTorch RNN layers.

Models implemented:

* SlowLSTM: a pedagogic example.
* LSTM: the original.
* GalLSTM: using dropout as in [Gal & Ghahramami](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf): *A Theoretically Grounded Application of Dropout in Recurrent Neural Networks*.
* MoonLSTM: using dropout as in [Moon & al](https://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf): *RNNDrop: A Novel Dropout for RNNs in ASR*.
* SemeniutaLSTM: using dropout as in [Semeniuta & al](https://arxiv.org/pdf/1603.05118.pdf): *Recurrent Dropout without Memory Loss*.

**Convention:** when using dropout, the activations are first computed, and **then** the nodes are droped. (dropout on the output, not the input)

## Install

`pip install -e .`

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

Available by running `make speed`

###  SlowLSTM  Benchmark 
 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
 
size   | nn.LSTM   |  SlowLSTM  | Speedup
-------|-----------|------------|--------
128    | 0.757     | 0.745      | 1.017
256    | 0.583     | 0.800      | 0.729
512    | 0.872     | 1.099      | 0.793
1024    | 3.658     | 3.969      | 0.922
2048    | 10.077     | 11.628      | 0.867
 
###  LSTM  Benchmark 
 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
 
size   | nn.LSTM   |  LSTM  | Speedup
-------|-----------|--------|--------
128    | 0.725     | 0.379  | 1.913
256    | 0.615     | 0.511  | 1.202
512    | 0.835     | 0.610  | 1.369
1024    | 3.485     | 3.547  | 0.982
2048    | 11.066     | 7.762  | 1.426
 
###  GalLSTM  Benchmark 
 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
 
size   | nn.LSTM   |  GalLSTM  | Speedup
-------|-----------|-----------|--------
128    | 0.669     | 0.696     | 0.961
256    | 0.579     | 0.652     | 0.887
512    | 0.966     | 0.885     | 1.091
1024    | 3.630     | 3.441     | 1.055
2048    | 9.598     | 9.509     | 1.009
 
###  MoonLSTM  Benchmark 
 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
 
size   | nn.LSTM   |  MoonLSTM  | Speedup
-------|-----------|------------|--------
128    | 0.656     | 0.551      | 1.190
256    | 0.652     | 0.589      | 1.108
512    | 0.949     | 0.614      | 1.546
1024    | 3.646     | 3.020      | 1.207
2048    | 10.202     | 8.539      | 1.195
 
###  SemeniutaLSTM  Benchmark 
 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
 
size   | nn.LSTM   |  SemeniutaLSTM  | Speedup
-------|-----------|-----------------|--------
128    | 0.580     | 0.375           | 1.548
256    | 0.718     | 0.843           | 0.852
512    | 0.859     | 0.574           | 1.496
1024    | 4.041     | 3.372           | 1.198
2048    | 9.587     | 8.668           | 1.106

