import tensorflow as tf

# Simple hello world using TensorFlow

# Create Constant op
# The op is added as a node to the default graph.

# The value returned by the constructor represents the output of the Constant op
hello = tf.constant("Hello, TensorFlow!")

# Start TensorFlow session
sess = tf.Session()

print(sess.run(hello))

