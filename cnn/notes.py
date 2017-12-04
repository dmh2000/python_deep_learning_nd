import tensorflow as tf
import numpy as np

# output depth
k_output = 64

# image dimensions
image_width = 10
image_height = 10
color_channels = 3

# convolution filter
filter_size_width = 5
filter_size_height = 5

# input/image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])

# weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# apply convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# apply activation function
conv_layer = tf.nn.relu(conv_layer)



"""
Setup the strides, padding and filter weight/bias such that
the output shape is (1, 2, 2, 3).
"""
"""
Setup the strides, padding and filter weight/bias such that
the output shape is (1, 2, 2, 3).
"""
import tensorflow as tf
import numpy as np
import math

# `tf.nn.conv2d` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)


def conv2d(input):
    # Filter (weights and bias)
    # The shape of the filter weight is (height, width, input_depth, output_depth)
    # The shape of the filter bias is (output_depth,)
    # TODO: Define the filter weights `F_W` and filter bias `F_b`.
    # NOTE: Remember to wrap them in `tf.Variable`, they are trainable parameters after all.
    output_channels = 3
    input_height = 4
    input_width = 4
    input_channels = 1
    filter_height = 2
    filter_width = 2
    filter_depth = 1
    F_W = tf.Variable(tf.truncated_normal([filter_height, filter_width, input_channels, output_channels]))
    F_b = tf.Variable(tf.zeros(output_channels))
    # TODO: Set the stride for each dimension (batch_size, height, width, depth)
    batch_size = 1
    strides = [batch_size, 2, 2, 1]
    # TODO: set the padding, either 'VALID' or 'SAME'.
    padding = "SAME"

    out_height = math.ceil(float(input_height - filter_height + 1) / float(strides[1]))
    out_width = math.ceil(float(input_width - filter_width + 1) / float(strides[2]))
    print(out_height, out_width)
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv2d
    # `tf.nn.conv2d` does not include the bias computation so we have to add it ourselves after.
    return tf.nn.conv2d(input, F_W, strides, padding) + F_b


out = conv2d(X)




