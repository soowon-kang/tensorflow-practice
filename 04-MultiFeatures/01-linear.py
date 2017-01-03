import tensorflow as tf

# tf Graph input
x1_data = [1., 0., 3., 0., 5.]
x2_data = [0., 2., 0., 4., 0.]
y_data = [1., 2., 3., 4., 5.]

# Try to find values for W and b that compute y_data = W * x_data + b
W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Our hypothesis
hypothesis = W1 * x1_data + W2 * x2_data + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

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
    sess.run(train)
    if step % 20 == 0:
        print("%4d | cost: %.16f, W1: %s, W2: %s, b: %s" % (step,
                                                            sess.run(cost),
                                                            sess.run(W1),
                                                            sess.run(W2),
                                                            sess.run(b)))


