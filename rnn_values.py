from __future__ import print_function

import random
import shutil

import math

import numpy as np
import tensorflow as tf

optimizers = {
    1: tf.train.GradientDescentOptimizer(1.),
    2: tf.train.AdadeltaOptimizer(),
    3: tf.train.AdagradOptimizer(.1),
    4: tf.train.MomentumOptimizer(1., 1.),
    5: tf.train.AdamOptimizer(.1),
    6: tf.train.FtrlOptimizer(.1),
    7: tf.train.RMSPropOptimizer(.1)
}


class Config:
    def __init__(self):
        self.opt_choice = 3
        self.num_units = 1
        self.input_size = 1
        self.size_data = 8
        self.num_epochs = 10000

    def __str__(self):
        return str(self.__dict__)


config = Config()
print(config)
print_interval = 100

log_dir = 'summaries'
vocabulary_size = 22
starting_numbers = list(range(vocabulary_size))
embedding_size = int(math.ceil(math.log(vocabulary_size, 2)))
test = random.choice(starting_numbers)
starting_numbers.pop(test)
random.shuffle(starting_numbers)


def data(i):
    return list(range(i, i + config.size_data))


def target(i):
    return list(range(i, i + config.size_data))


init = tf.random_uniform_initializer()
with tf.Session() as sess, tf.variable_scope("", initializer=init):

    # embeddings
    inputs = tf.placeholder(tf.float32, shape=[config.size_data], name='inputs')
    inputs_list = [tf.reshape(x, [1, 1]) for x in tf.unpack(inputs)]

    # GRU
    cell = tf.nn.rnn_cell.GRUCell(config.num_units, config.input_size)
    lstm_output, state = tf.nn.rnn(cell, inputs_list, dtype=tf.float32)

    # Transform outputs
    w = tf.get_variable("w", [config.num_units])
    concat = tf.concat(1, lstm_output, name='concat')
    outputs = tf.mul(w, concat, name='outputs')

    # Train loss
    targets = tf.placeholder(tf.float32, shape=[config.size_data], name='targets')
    loss = tf.nn.l2_loss(targets - outputs)
    train_op = optimizers[config.opt_choice].minimize(loss)

    # Tensorboard
    shutil.rmtree(log_dir)
    writer = tf.train.SummaryWriter(log_dir, sess.graph)

    tf.initialize_all_variables().run()


    def feed(i):
        return {inputs: data(i),
                targets: target(i)}


    epoch = 0.0
    prev_cost = 0
    avg_speed = 0
    while True:
        epoch += 1.0
        try:
            cost = 0
            for i in range(len(starting_numbers)):
                start = 0  # starting_numbers[i]
                _, loss_value, train_outputs = sess.run(
                    [train_op, loss, outputs], feed_dict=feed(start))
                cost += loss_value

            speed = 0 if epoch == 1 else prev_cost - cost
            avg_speed = avg_speed * ((epoch - 1) / epoch) + speed / epoch
            prev_cost = cost

            print('\repoch: {:5.0f} | cost: {:6.4f} | avg speed: {:6.4f} | speed {:6.4f}'
                  .format(epoch, cost, avg_speed, speed), end='')
            if epoch % print_interval == 0:
                test_outputs = sess.run(outputs, feed_dict=feed(start))

                print()
                print()
                print("inputs", data(start))
                print("outputs", test_outputs[0, :])
                print("outputs", map(round, test_outputs[0, :]))
                print("targets", target(start))

                rand_choice = random.choice(starting_numbers)
                test_outputs = sess.run(outputs, feed_dict=feed(rand_choice))

                # save summary for Tensorboard
                # writer.add_summary(summary)

        except KeyboardInterrupt:
            break

    test = 30
    test_data = data(test)
    test_outputs = sess.run(outputs, feed_dict=feed(test))
    print()
    print("inputs", test_data)
    print("outputs", test_outputs)
    print("targets", target(test))
    # print_output(test_data, test_outputs)
