import input_data
import numpy as np
import tensorflow as tf

mnist = input_data.read_data_sets(
    "/Users/CubePenguin/PycharmProjects/tensorflow-practice/00-data/",
    one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# tf Graph input
X = tf.placeholder("float", [None, 784])    # 28 * 28 == 784
Y = tf.placeholder("float", [None, 10])     # 0-9 digits => 10 classes

# Set model weight & bias
W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 256]))
W3 = tf.Variable(tf.random_normal([256, 10]))

b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([10]))

# Construct model
in1 = tf.nn.relu(tf.matmul(X, W1) + b1)
hid1 = tf.nn.relu(tf.matmul(in1, W2) + b2)  # Hidden layer with ReLU activation

hypothesis = tf.matmul(hid1, W3) + b3       # No need to use softmax here

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

# Test model
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

# Calculate accuracy
acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy: %.4f" % acc.eval({X: mnist.test.images, Y: mnist.test.labels},
                                  session=sess))

# Neural Networks: 94.4%

