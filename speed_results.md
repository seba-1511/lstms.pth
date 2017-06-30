##  SlowLSTM  Benchmark 
 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
 
size   | nn.LSTM   |  SlowLSTM  | Speedup
-------|-----------|------------|--------
128    | 0.732     | 0.664      | 1.102
256    | 0.596     | 0.775      | 0.769
512    | 0.774     | 1.004      | 0.771
1024    | 2.574     | 3.236      | 0.795
2048    | 7.797     | 9.435      | 0.826
 
##  LSTM  Benchmark 
 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
 
size   | nn.LSTM   |  LSTM  | Speedup
-------|-----------|--------|--------
128    | 0.570     | 0.493  | 1.158
256    | 0.646     | 0.467  | 1.382
512    | 0.790     | 0.637  | 1.241
1024    | 2.962     | 2.514  | 1.178
2048    | 8.143     | 6.777  | 1.201
 
##  GalLSTM  Benchmark 
 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
 
size   | nn.LSTM   |  GalLSTM  | Speedup
-------|-----------|-----------|--------
128    | 0.513     | 0.634     | 0.810
256    | 0.604     | 0.747     | 0.809
512    | 0.867     | 0.871     | 0.996
1024    | 2.846     | 2.651     | 1.074
2048    | 7.310     | 7.163     | 1.020
 
##  MoonLSTM  Benchmark 
 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
 
size   | nn.LSTM   |  MoonLSTM  | Speedup
-------|-----------|------------|--------
128    | 0.679     | 0.391      | 1.737
256    | 0.640     | 0.488      | 1.311
512    | 0.785     | 0.637      | 1.233
1024    | 2.834     | 2.435      | 1.164
2048    | 7.533     | 7.174      | 1.050
 
##  SemeniutaLSTM  Benchmark 
 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
 
size   | nn.LSTM   |  SemeniutaLSTM  | Speedup
-------|-----------|-----------------|--------
128    | 0.544     | 0.549           | 0.992
256    | 0.631     | 0.469           | 1.347
512    | 0.848     | 0.674           | 1.259
1024    | 3.015     | 2.614           | 1.153
2048    | 7.469     | 7.459           | 1.001
 
