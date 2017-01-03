import tensorflow as tf

# Basic operations with variable as graph input
# The value returned by the constructor represents
# the output of the Constant op. (define as input when running session)

# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
# variable types should be matched
add = tf.add(a, b)
mul = tf.mul(a, b)

# launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2,
                                                                         b: 3}))
