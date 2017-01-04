import input_data
import numpy as np
import tensorflow as tf

mnist = input_data.read_data_sets(
    "/Users/CubePenguin/PycharmProjects/tensorflow-practice/00-data/",
    one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 16
batch_size = 100
display_step = 1

# tf Graph input
X = tf.placeholder("float", [None, 784])  # 28 * 28 == 784
Y = tf.placeholder("float", [None, 10])  # 0-9 digits => 10 classes


# Xavier initialization
def xavier_init(n_inputs, n_outputs, uniform=True):
    """Set the parameter initialization using the method described.
    This method is designed to keep the scale of the gradients roughly the same
    in all layers.
    Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
    Args:
        n_inputs: The number of input nodes into each output.
        n_outputs: The number of output nodes for each input.
        uniform: If true use a uniform distribution, otherwise use a normal.
    Returns:
        An initializer.
    """
    if uniform:
        # 6 was used in the paper.
        init_range = np.math.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = np.math.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


# Set model weight & bias
W1 = tf.get_variable("W1", shape=[784, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[256, 10],
                     initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([10]))

# Construct model
in1 = tf.nn.relu(tf.matmul(X, W1) + b1)
hid1 = tf.nn.relu(tf.matmul(in1, W2) + b2)  # Hidden layer with ReLU activation

hypothesis = tf.matmul(hid1, W3) + b3  # No need to use softmax here

# Softmax loss
# tf.nn.softmax_cross_entropy_with_logits means
# calculating cost function with softmax classifier where
# the input of the classifier is real number
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))

# Adam optimizer
# The best optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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
    total_batch = int(mnist.train.num_examples / batch_size)

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
        print("Epoch: %2d," % (epoch + 1), "cost: %.9f" % avg_cost)

print("Training Finished!")

# Test model
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

# Calculate accuracy
acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy: %.4f" % acc.eval({X: mnist.test.images, Y: mnist.test.labels},
                                  session=sess))

# NN + Xavier initialization: 97.8%

