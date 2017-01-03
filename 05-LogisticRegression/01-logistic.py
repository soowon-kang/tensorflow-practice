import tensorflow as tf
import numpy as np

# http://docsscipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Try to find values for W and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_uniform([1, len(x_data)], -5.0, 5.0))

# Our hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

# Logistic cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# Minimize
rate = tf.constant(0.1)     # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

# Before starting, initialize the variables.
# We will 'run' this first
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(("%4d | " % step) +
              ("cost: %.16f, " % (sess.run(cost,
                                           feed_dict={X: x_data, Y: y_data}))) +
              ("W: %s" % sess.run(W)))

# Learns best fit is W: [0 1 1]

