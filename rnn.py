from __future__ import print_function
import shutil

import sys

import numpy as np
import tensorflow as tf

num_units = 1
input_size = 1
size_data = 10
num_epochs = 1000
print_interval = num_epochs // 10
log_dir = 'summaries'
starting_numbers = np.random.choice(num_epochs, num_epochs)


def data(i):
    return range(i, i + size_data)


# inputs = tf.constant(list(data), shape=[size_data], dtype=tf.float32)

init = tf.random_uniform_initializer()
with tf.Session() as sess, tf.variable_scope("", initializer=init):
    # GRU
    inputs = tf.placeholder(tf.float32, shape=[size_data])
    cell = tf.nn.rnn_cell.GRUCell(input_size, num_units)
    unpack = [tf.reshape(x, [1, input_size]) for x in tf.unpack(inputs)]
    lstm_output, _ = tf.nn.rnn(cell, unpack, dtype=tf.float32)

    # Transform outputs
    w = tf.get_variable("w", shape=(size_data, size_data))
    outputs = tf.matmul(tf.concat(1, lstm_output), w)

    # Train loss
    targets = tf.concat(1, inputs)
    loss = tf.nn.l2_loss(outputs - targets)
    train_op = tf.train.AdadeltaOptimizer().minimize(loss)

    # values to track
    summary_op = tf.scalar_summary('loss', loss)
    shutil.rmtree(log_dir)
    writer = tf.train.SummaryWriter(log_dir, sess.graph_def)

    tf.initialize_all_variables().run()
    for i in xrange(num_epochs):
        start = 25
        feed = {inputs: data(start)}
        _, summary, train_outputs, loss_value = sess.run(
            [train_op, summary_op, outputs, loss], feed_dict=feed)

        if i % print_interval == 0:
            # save summary
            writer.add_summary(summary)

            print('loss: ' + str(loss_value))

        if loss_value < .1:
            print('loss: ' + str(loss_value))
            break

    test_outputs = sess.run(outputs, feed_dict={inputs: data(start)})

print('outputs')
for output in train_outputs[0, :]:
    print("{:1.1f}, ".format(output), end='')
