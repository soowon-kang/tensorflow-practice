import tensorflow as tf
import numpy as np

# http://docsscipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('xor.txt', unpack=True, dtype='int')
x_data = xy[0:-1]
y_data = xy[-1]

# tf Graph input
X = tf.placeholder("float32")
Y = tf.placeholder("float32")

# Set model weights
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# Construct model
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

# logistic cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# Gradient descent
rate = tf.constant(0.01)     # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

# Before starting, initialize the variables.
# We will 'run' this first
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

for step in range(1001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print(("%4d | " % step) +
              ("cost: %.16f, " % (sess.run(cost,
                                           feed_dict={X: x_data, Y: y_data}))) +
              ("W: %s" % sess.run(W)))

print("-" * 30)

# Test model
correct_prediction = tf.equal(Y, tf.floor(0.5 + hypothesis))

# Calculate accuracy
acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run([hypothesis, tf.floor(0.5+hypothesis), correct_prediction, acc],
               feed_dict={X: x_data, Y: y_data}))
print("Accuracy: ", acc.eval({X: x_data, Y: y_data}, session=sess))


