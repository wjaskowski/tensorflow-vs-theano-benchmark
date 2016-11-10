# Theano vs. Tensorflow benchmark

## Runnning
```
ipython benchmark.ipy
```

## Experiment

### Software
- Ubuntu 16.04
- Python 3.5
- Theano 0.8.2, Lasagne 0.2dev
- TensorFlow 0.11
- OpenBlas
- CUDA 8.0, CuDNN 5.1

### Hardware
- i7-5960X CPU @ 3.00GHz (8 physical cores)
- NVidia Titan X

### Method
- 1000 times repeat: forward pass and back propagation
- various neural network architectures

### Architectures
- input shapes: 40x30, 128x128, 256x256
- 2-4 conv layers
- 8-128 filters
- output neurons: 128, 512

### Findings
- GPU: *TensorFlow is 1.3-7 times slower than Theano*
- CPU: TensorFlow is usually much faster than Theano (up to 4.8 times)

## Results
The last two columns show *how many times* Theano is faster than Tensorflow

### GPU
device|input_shape|arch|th_fwd_time|tf_fwd_time|th_bprop_time|tf_bprop_time|tf/th_fwd_time|tf/th_bprop_time
---|---|---|---|---|---|---|---|---
gpu|40x30|[8,8], 128|0.000192|0.00138|0.000549|0.002861|7.187499999999999|5.211293260473588
gpu|40x30|[16,16], 128|0.000211|0.001152|0.000592|0.002839|5.459715639810427|4.795608108108108
gpu|40x30|[32,16], 128|0.000222|0.00138|0.000712|0.002657|6.216216216216216|3.731741573033708
gpu|40x30|[64,32], 512|0.000333|0.001345|0.001052|0.002734|4.039039039039039|2.5988593155893533
gpu|128x128|[64,32], 128|0.001776|0.00368|0.005551|0.009213|2.0720720720720722|1.659700954782922
gpu|128x128|[64,64], 128|0.002012|0.00381|0.006631|0.010018|1.8936381709741552|1.5107826873774697
gpu|128x128|[128,64], 128|0.003091|0.005284|0.010887|0.014455|1.7094791329666774|1.3277303205658124
gpu|128x128|[32,32,32], 128|0.001255|0.003473|0.003459|0.00724|2.7673306772908366|2.093090488580515
gpu|128x128|[32,32,32,32], 128|0.00128|0.003389|0.003604|0.007533|2.64765625|2.0901775804661487
gpu|256x256|[32,32,32], 128|0.004198|0.01065|0.011839|0.024008|2.53692234397332|2.0278739758425544

### CPU
device|input_shape|arch|th_fwd_time|tf_fwd_time|th_bprop_time|tf_bprop_time|tf/th_fwd_time|tf/th_bprop_time
---|---|---|---|---|---|---|---|---
cpu|40x30|[8,8], 128|0.000949|0.001875|0.003039|0.004152|1.975763962065332|1.3662388943731492
cpu|40x30|[16,16], 128|0.001768|0.000867|0.00546|0.002536|0.4903846153846154|0.46446886446886454
cpu|40x30|[32,16], 128|0.00306|0.000946|0.009645|0.004337|0.30915032679738563|0.44966303784344214
cpu|40x30|[64,32], 512|0.007011|0.001584|0.021999|0.005402|0.22593068035943517|0.2455566162098277
cpu|128x128|[64,32], 128|0.054292|0.015037|0.185496|0.05085|0.276965298754881|0.27412990037521023
cpu|128x128|[64,64], 128|0.045763|0.018699|0.171529|0.061975|0.4086052050783384|0.3613091663800291
