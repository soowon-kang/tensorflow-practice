import tensorflow as tf

# tf Graph input
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 1 and b 0, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis
hypothesis = W * X + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

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
              ("W: %s, b: %s" % (sess.run(W), sess.run(b))))

# Learns best fit is W: [1.0], b: [0.0]
print(sess.run(hypothesis, feed_dict={X: 5.}))      # Y = [5.0]
print(sess.run(hypothesis, feed_dict={X: 2.5}))     # Y = [2.5]

