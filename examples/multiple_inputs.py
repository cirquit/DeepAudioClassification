""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.01
num_steps = 500
batch_size = 128
display_step = 100
save_step    = 100
model_dir    = "models/"


# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
dummy_input = 1  # dummy input 
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
X_dummy = tf.placeholder("float", [None, dummy_input]) 
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'dummy_h2': tf.Variable(tf.random_normal([dummy_input, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'dummy_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x_1, x_2):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x_1, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    dummy_layer = tf.add(tf.matmul(x_2, weights['dummy_h2']), biases['dummy_b2'])

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out'] + tf.matmul(dummy_layer, weights['out'])
    return out_layer

# Construct model
logits = neural_net(X, X_dummy)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Initialize saver
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

   #  dummy_var = np.array([[x] for x in range(batch_size)], dtype='float32')

   #  for step in range(1, num_steps+1):
   #      batch_x, batch_y = mnist.train.next_batch(batch_size)
   #      # Run optimization op (backprop)
   #      sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, X_dummy: dummy_var})
   #      if step % display_step == 0 or step == 1:
   #          # Calculate batch loss and accuracy
   #          loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
   #                                                               Y: batch_y,
   #      							 X_dummy: dummy_var})
   #          print("Step " + str(step) + ", Minibatch Loss= " + \
   #                "{:.4f}".format(loss) + ", Training Accuracy= " + \
   #                "{:.3f}".format(acc))

   #      if step % save_step == 0:
   #      	modelname = "mnist-example-step-{}-.chkp".format(step)
   #      	saver.save(sess, model_dir + modelname)

   #  print("Optimization Finished!")

    dummy_test_var = np.array([[x] for x in range(10000)], dtype='float32')

    saver.restore(sess, "models/mnist-example-step-500-.chkp")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels,
				      X_dummy: dummy_test_var}))
