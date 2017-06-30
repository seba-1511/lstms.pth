##  SlowLSTM  Benchmark 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
size   | nn.LSTM   |  SlowLSTM  | Speedup
-------|-----------|------------|--------
128    | 0.625     | 0.647      | 0.966
256    | 0.605     | 0.732      | 0.826
512    | 0.875     | 1.088      | 0.804
1024    | 2.625     | 2.952      | 0.889
2048    | 6.952     | 8.768      | 0.793
 
##  LSTM  Benchmark 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
size   | nn.LSTM   |  LSTM  | Speedup
-------|-----------|--------|--------
128    | 0.690     | 0.470  | 1.469
256    | 0.578     | 0.492  | 1.176
512    | 0.849     | 0.688  | 1.234
1024    | 2.653     | 2.388  | 1.111
2048    | 6.927     | 6.736  | 1.028
 
##  GalLSTM  Benchmark 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
size   | nn.LSTM   |  GalLSTM  | Speedup
-------|-----------|-----------|--------
128    | 0.497     | 0.624     | 0.797
256    | 0.622     | 0.632     | 0.984
512    | 0.847     | 0.684     | 1.239
1024    | 2.489     | 2.810     | 0.886
2048    | 7.033     | 6.889     | 1.021
 
##  MoonLSTM  Benchmark 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
size   | nn.LSTM   |  MoonLSTM  | Speedup
-------|-----------|------------|--------
128    | 0.598     | 0.393      | 1.522
256    | 0.591     | 0.475      | 1.244
512    | 0.820     | 0.580      | 1.414
1024    | 2.592     | 2.279      | 1.137
2048    | 7.074     | 6.552      | 1.080
 
##  SemeniutaLSTM  Benchmark 
Inference timings on a single sequence of length 1000  with `dropout =  0.5 `.
size   | nn.LSTM   |  SemeniutaLSTM  | Speedup
-------|-----------|-----------------|--------
128    | 0.705     | 0.410           | 1.718
256    | 0.578     | 0.472           | 1.224
512    | 0.803     | 0.544           | 1.476
1024    | 2.562     | 2.391           | 1.071
2048    | 7.452     | 6.937           | 1.074
 
