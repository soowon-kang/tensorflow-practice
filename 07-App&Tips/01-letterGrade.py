import tensorflow as tf
import numpy as np

# http://docsscipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

# tf Graph input
X = tf.placeholder("float32", [None, 3])  # 1 (for bias), x1, x2
Y = tf.placeholder("float32", [None, 3])  # A, B, C => 3 classes

# Set model weights
W = tf.Variable(tf.zeros([3, 3]))

# Construct model
# https://www.tensorflow.org/version/r0.7/tutorials/mnist/beginners/index.html
# First, we multiply x by W with the expression tf.matmul(x, W)
# This is flipped from when we multiplied them in our equation,
# where we has Wx, as a small trick to deal with x being
# a 2D tensor with multiple inputs
# We then add b, and finally apply tf.nn.softmax
h = tf.matmul(X, W)             # to fit dimensions
hypothesis = tf.nn.softmax(h)   # softmax

# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

# Gradient descent
rate = tf.constant(0.1)     # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

# Before starting, initialize the variables.
# We will 'run' this first
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print(("%4d | " % step) +
              ("cost: %.16f, " % (sess.run(cost,
                                           feed_dict={X: x_data, Y: y_data}))) +
              ("W: %s" % sess.run(W)))

