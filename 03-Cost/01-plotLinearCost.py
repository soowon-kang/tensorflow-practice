import tensorflow as tf
import matplotlib.pyplot as plt

# tf Graph input
X = [1., 2., 3.]
Y = [1., 2., 3.]

n = len(X)

# Set model weights
W = tf.placeholder(tf.float32)

# Construct a linear model
hypothesis = tf.mul(X, W)

# Cost function
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / n

# Initializing the variables
init = tf.global_variables_initializer()

# For graphs
W_val = []
cost_val = []

# Launch the graph.
sess = tf.Session()
sess.run(init)
for i in range(-30, 50):
    x = i*0.1
    y = sess.run(cost, feed_dict={W: x})
    W_val.append(x)
    cost_val.append(y)
    print(x, y)

# Graphic display
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()