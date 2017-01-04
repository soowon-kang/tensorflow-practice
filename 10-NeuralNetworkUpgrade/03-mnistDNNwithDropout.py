import input_data
import numpy as np
import tensorflow as tf

mnist = input_data.read_data_sets(
    "/Users/CubePenguin/PycharmProjects/tensorflow-practice/00-data/",
    one_hot=True)

# Parameters
learning_rate = 0.001
# training_epochs = 15
batch_size = 100
display_step = 1

# tf Graph input
X = tf.placeholder("float", [None, 784])  # 28 * 28 == 784
Y = tf.placeholder("float", [None, 10])  # 0-9 digits => 10 classes

# Set model weight & bias
W1 = tf.get_variable("W1", shape=[784, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable("W5", shape=[256, 10],
                     initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([256]))
b4 = tf.Variable(tf.random_normal([256]))
b5 = tf.Variable(tf.random_normal([10]))

# Construct model
dropout_rate = tf.placeholder("float")
in1 = tf.nn.relu(tf.matmul(X, W1) + b1)
d1 = tf.nn.dropout(in1, dropout_rate)
hid1 = tf.nn.relu(tf.matmul(d1, W2) + b2)   # Hidden layer with ReLU activation
d2 = tf.nn.dropout(hid1, dropout_rate)
hid2 = tf.nn.relu(tf.matmul(d2, W3) + b3)   # Hidden layer with ReLU activation
d3 = tf.nn.dropout(hid2, dropout_rate)
hid3 = tf.nn.relu(tf.matmul(d3, W4) + b4)   # Hidden layer with ReLU activation
d4 = tf.nn.dropout(hid3, dropout_rate)

hypothesis = tf.matmul(d4, W5) + b5         # No need to use softmax here

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
bef_cost = 100.
aft_cost = 99.
epoch = 0
while bef_cost - aft_cost > 0.001:
    epoch += 1
    bef_cost = aft_cost
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)

    # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Fit training using batch data
        sess.run(train, feed_dict={X: batch_x, Y: batch_y, dropout_rate: 0.7})

        # Compute average cost
        avg_cost += sess.run(cost, feed_dict={X: batch_x,
                                              Y: batch_y,
                                              dropout_rate: 1.0}) / total_batch
    aft_cost = avg_cost

    # Display log per each epoch step
    if epoch % display_step == 0:
        print("Epoch: %2d," % epoch, "cost: %.9f" % avg_cost)

print("Training Finished!")

# Test model
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

# Calculate accuracy
acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy: %.4f" % acc.eval({X: mnist.test.images,
                                   Y: mnist.test.labels,
                                   dropout_rate: 1.},
                                  session=sess))

# Deep Neural Networks and Dropout: 98%
