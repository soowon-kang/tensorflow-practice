import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(
    "/Users/CubePenguin/PycharmProjects/tensorflow-practice/00-data/",
    one_hot=True)

training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph input
X = tf.placeholder("float", [None, 784])    # 28 * 28 == 784
Y = tf.placeholder("float", [None, 10])     # 0-9 digits => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
# https://www.tensorflow.org/version/r0.7/tutorials/mnist/beginners/index.html
# First, we multiply x by W with the expression tf.matmul(x, W)
# This is flipped from when we multiplied them in our equation,
# where we has Wx, as a small trick to deal with x being
# a 2D tensor with multiple inputs
# We then add b, and finally apply tf.nn.softmax
h = tf.matmul(X, W)             # to fit dimensions
hypothesis = tf.nn.softmax(h + b)   # softmax

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


# Training
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)

    # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Fit training using batch data
        sess.run(train, feed_dict={X: batch_x, Y: batch_y})

        # Compute average cost
        avg_cost += sess.run(cost,
                             feed_dict={X: batch_x, Y: batch_y}) / total_batch

    # Display log per each epoch step
    if epoch % display_step == 0:
        print("Epoch: %2d," % (epoch+1), "cost: %.9f" % avg_cost)

print("Training Finished!")

# Get one and predict
r = np.random.randint(0, mnist.test.num_examples - 1)
print("Label: %s" % sess.run(tf.arg_max(mnist.test.labels[r:r+1], 1)))
print("Prediction: %s" % sess.run(tf.arg_max(hypothesis, 1),
                                  feed_dict={X: mnist.test.images[r:r+1]}))

# Show the Image
plt.imshow(mnist.test.images[r:r+1].reshape(28, 28),
           cmap='Greys',
           interpolation='nearest')
plt.show()
