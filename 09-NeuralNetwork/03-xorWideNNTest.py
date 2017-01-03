import tensorflow as tf
import numpy as np

# http://docsscipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('xor.txt', unpack=True, dtype='int')
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

# tf Graph input
X = tf.placeholder("float32")
Y = tf.placeholder("float32")

# Set model weights
W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0))

# Set model biases
b1 = tf.Variable(tf.zeros([10]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Construct model
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# logistic cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

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

for step in range(10001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 2000 == 0:
        print("%5d | " % step,
              ("cost: %.16f, " % (sess.run(cost,
                                           feed_dict={X: x_data, Y: y_data}))),
              "W1: %s" % sess.run(W1),
              "W2: %s" % sess.run(W2))

print("-" * 30)

# Test model
correct_prediction = tf.equal(Y, tf.floor(0.5 + hypothesis))

# Calculate accuracy
acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run([hypothesis, tf.floor(0.5+hypothesis), correct_prediction, acc],
               feed_dict={X: x_data, Y: y_data}))
print("Accuracy: ", acc.eval({X: x_data, Y: y_data}, session=sess))


