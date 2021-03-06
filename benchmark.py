#!/usr/bin/env python3
import numpy as np
import re
from time import time

learning_rate = 0.001

def create_th(image_shape, output_dim, layers_conf):
    from lasagne.init import GlorotUniform, Constant
    from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, get_output, \
        get_all_params, set_all_param_values
    from lasagne.nonlinearities import rectify
    from lasagne.objectives import squared_error
    from lasagne.updates import rmsprop

    x = th.tensor.tensor4("input")
    t = th.tensor.matrix("target")

    net = InputLayer(shape=[None, 1, image_shape[0], image_shape[1]], input_var=x)
    for num_filters,kernel_size,stride in layers_conf[:-1]:
        net = Conv2DLayer(net, num_filters=num_filters, filter_size=[kernel_size, kernel_size],
                        nonlinearity=rectify, W=GlorotUniform(),
                        b=Constant(.1), stride=stride)
    net = DenseLayer(net, num_units=layers_conf[-1], nonlinearity=rectify, W=GlorotUniform(),
                     b=Constant(.1))
    net = DenseLayer(net, num_units=output_dim, nonlinearity=None)

    q = get_output(net)
    loss = squared_error(q, t).mean()

    params = get_all_params(net, trainable=True)
    updates = rmsprop(loss, params, learning_rate)

    backprop = th.function([x, t], loss, updates=updates, name="bprop")
    fwd_pass = th.function([x], q, name="fwd")
    return fwd_pass, backprop


def create_tf(image_shape, output_dim, layers_conf):
    from tensorflow.contrib import layers

    x = tf.placeholder(tf.float32, shape=[None, 1, image_shape[0], image_shape[1]], name="input")
#    x = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], 1], name="input")
    t = tf.placeholder(tf.float32, shape=[None, output_dim], name="target")

    net = x
    for num_filters,kernel_size,stride in layers_conf[:-1]:
        net = layers.conv2d(net, num_outputs=num_filters, kernel_size=kernel_size, stride=stride, activation_fn=tf.nn.relu,
                weights_initializer=layers.xavier_initializer_conv2d(), biases_initializer=tf.constant_initializer(0.1), data_format="NCHW")

    net = layers.flatten(net)
    net = layers.fully_connected(net, num_outputs=layers_conf[-1], activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1))
    y = layers.fully_connected(net, num_outputs=output_dim, activation_fn=None, weights_initializer=layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1))
    mean_square_error = 0.5*tf.reduce_mean((y - t)**2)
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss=mean_square_error)

    def backprop(x_batch, y_batch):
        optimizer.run({x: x_batch, t: y_batch})

    def fwd_pass(x_batch):
        return y.eval({x: x_batch})
        import timeline

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run([y], feed_dict={x: x_batch}, options=run_options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        trace_file = tf.gfile.Open(name='timeline', mode='w')
        trace_file.write(tl.generate_chrome_trace_format(show_memory=True))

    return fwd_pass, backprop


def time_batch(data_x, data_y, function, repeats=1000):
    data_points = data_x.shape[0]
    def bench(n):
        for i in range(n):
            starti = (i*batchsize) % data_points
            endi = starti + batchsize
            batch_x = data_x[starti:endi]
            batch_y = data_y[starti:endi]
            function(batch_x, batch_y)
    bench(repeats // 20) # warmup

    start = time()
    bench(repeats)
    return (time() - start)/repeats


if __name__ == '__main__':
# Random data
    import sys
    backend = sys.argv[1]
    assert backend in ['tf', 'th']
    device = sys.argv[2]
    assert device in ['cpu', 'gpu']
    image_shape = [int(x) for x in sys.argv[3:5]]
    layers = [x for x in sys.argv[5:]]
    for i in range(len(layers[:-1])):
        m = re.search('(\d+),k=(\d+),s=(\d+)', layers[i])
        layers[i] = [int(m.group(j+1)) for j in range(3)]
    layers[-1] = int(layers[-1])

    # Prepare random data
    np.random.seed(123)
    batchsize = 128
    data_points = 1024
    output_dim = 4
    data_x = np.random.rand(data_points, *image_shape, 1).astype(dtype=np.float32)
    data_y = np.random.rand(data_points, output_dim).astype(dtype=np.float32)

    if backend == 'tf':
        data_x = np.reshape(data_x, [data_points, 1, image_shape[0], image_shape[1]])
        import tensorflow as tf
        if device == 'gpu':
            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            sess = tf.Session(config=config)
        else:
            sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
        with sess.as_default():
            tf.set_random_seed(np.random.rand())
            fwd_tf, bprop_tf = create_tf(image_shape, output_dim, layers)
            tf.global_variables_initializer().run()
            print("%.6f %.6f" % (time_batch(data_x, data_y, lambda x, y: fwd_tf(x)), 
            time_batch(data_x, data_y, lambda x, y: bprop_tf(x, y))))
    elif backend == 'th':
        import os
        os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=%s,floatX=float32,allow_gc=False,lib.cnmem=0.6" % device
        import theano as th

        data_x = np.reshape(data_x, [data_points, 1, image_shape[0], image_shape[1]])
        comp_time = time()
        fwd_th, bprop_th = create_th(image_shape, output_dim, layers)
        print('compilation time: %.1f' % (time() - comp_time))
        print("%.6f %.6f" % (time_batch(data_x, data_y, lambda x, y: fwd_th(x)), 
                time_batch(data_x, data_y, lambda x, y: bprop_th(x, y))))
