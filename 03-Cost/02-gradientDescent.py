import tensorflow as tf

# tf Graph input
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 1 and b 0, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis
hypothesis = W * X

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
rate = tf.constant(0.1)     # Learning rate, alpha
descent = W - rate * tf.reduce_mean((W * X - Y) * X)
update = W.assign(descent)

# Before starting, initialize the variables.
# We will 'run' this first
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(20):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(("%2d | " % step) +
          ("cost: %.16f, " % (sess.run(cost,
                                       feed_dict={X: x_data, Y: y_data}))) +
          ("W: %s" % (sess.run(W))))

