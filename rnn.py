import tensorflow as tf
import numpy as np

num_units = 1
input_size = 1
num_epochs = 70
size_data = 10
log_dir = '../summaries'
starting_numbers = np.random.choice(num_epochs, num_epochs)


def data(i):
    return range(i, i + size_data)


# inputs = tf.constant(list(data), shape=[size_data], dtype=tf.float32)

init = tf.constant_initializer(1)
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
    train_op = tf.train.MomentumOptimizer(.01, .1).minimize(loss)

    tf.initialize_all_variables().run()
    for i in xrange(num_epochs):
        start = starting_numbers[i]
        _, train_outputs, loss_value = sess.run([train_op, outputs, loss],
                                                feed_dict={inputs: data(start)})

        tf.scalar_summary('loss', loss_value)
        tf.merge_all_summaries()
        tf.train.SummaryWriter(log_dir)


        if i % 10 == 0:
            print('loss: ' + str(loss_value))

    test_outputs = sess.run(outputs, feed_dict={inputs: data(7)})

print('outputs')
for output in train_outputs[0, :]:
    print("{:1.1f}".format(output))
