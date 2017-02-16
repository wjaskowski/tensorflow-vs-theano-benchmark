# Theano vs. Tensorflow benchmark

## Runnning
```
ipython benchmark.ipy
```

## Experiment

### Findings
- GPU: For small architectures and input images TensorFlow is up to 5 times slower than Theano
- CPU: TensorFlow is usually much faster than Theano (up to 4.8 times)

## Results
See: [GPU](results_gpu.csv), [CPU](results_cpu.csv).

Note: The last two columns show *how many times* Theano is faster than Tensorflow.

### Software
- Ubuntu 16.04
- Python 3.5
- Theano 0.8.2, Lasagne 0.2dev
- TensorFlow 1.0
- OpenBlas
- CUDA 8.0, CuDNN 5.1

### Hardware
- i7-5960X CPU @ 3.00GHz (8 physical cores)
- NVidia Titan X

### Method
- random data
- 1000 times repeat: forward pass and backpropagation

### Architectures
- input shapes: 40x30, 128x128, 256x256
- 2-4 conv layers
- 8-128 filters
- output neurons: 128, 512
