from __future__ import print_function

import random
import shutil

import math

import numpy as np
import tensorflow as tf

optimizers = {
    1: tf.train.GradientDescentOptimizer(.1),
    2: tf.train.AdadeltaOptimizer(),
    3: tf.train.AdagradOptimizer(.1),
    4: tf.train.MomentumOptimizer(.1, 1.),
    5: tf.train.AdamOptimizer(.1),
    6: tf.train.FtrlOptimizer(.1),
    7: tf.train.RMSPropOptimizer(.1)
}


class Config:
    def __init__(self):
        self.opt_choice = 6
        self.num_units = 5
        self.input_size = 1
        self.size_data = 1
        self.num_epochs = 10000

    def __str__(self):
        return str(self.__dict__)


config = Config()
print(config)
print_interval = 50

log_dir = 'summaries'
vocabulary_size = 22
starting_numbers = list(range(vocabulary_size))
embedding_size = int(math.ceil(math.log(vocabulary_size, 2)))
test = random.choice(starting_numbers)
# starting_numbers.pop(test)
# random.shuffle(starting_numbers)


def data(i):
    return [i] * config.size_data
    # range(i, i + config.size_data)


def target(i):
    return [map(int, (bin(i)[2:].rjust(config.num_units, '0')))]

init = tf.random_uniform_initializer()
with tf.Session() as sess, tf.variable_scope("", initializer=init):

    # embeddings
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    inputs = tf.placeholder(tf.int32, shape=[config.size_data], name='inputs')
    lookups = tf.nn.embedding_lookup(embeddings, inputs)
    inputs_list = tf.unpack(tf.expand_dims(lookups, 1))

    # GRU
    cell = tf.nn.rnn_cell.GRUCell(config.num_units, config.input_size)
    lstm_output, state = tf.nn.rnn(cell, inputs_list, dtype=tf.float32)

    # Transform outputs
    outputs = tf.nn.sigmoid(state)

    # Train loss
    targets = tf.placeholder(tf.float32, shape=[1, config.num_units], name='targets')
    losses = tf.nn.softmax_cross_entropy_with_logits(outputs, targets)
    loss = tf.reduce_sum(losses, name='loss')
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
                start = starting_numbers[i]
                _, loss_value, train_outputs = sess.run(
                    [train_op, loss, outputs], feed_dict=feed(start))
                cost += loss_value

            speed = 0 if epoch == 1 else prev_cost - cost
            avg_speed = avg_speed * ((epoch - 1) / epoch) + speed / epoch
            prev_cost = cost

            print('\repoch: {:5.0f} | cost: {:6.1f} | avg speed: {:6.4f} | speed {:6.4f}'
                  .format(epoch, cost, avg_speed, speed), end='')
            if epoch % print_interval == 0:
                # test_outputs = sess.run(outputs, feed_dict=feed(test))
                #
                # print()
                # print()
                # print("inputs", data(test))
                # print("outputs", test_outputs)
                # print("rounded outputs", map(round, test_outputs[0]))
                # print("choice", np.argmax(test_outputs))
                # print("targets", target(test))

                rand_choice = random.choice(starting_numbers)
                test_outputs = sess.run(outputs, feed_dict=feed(rand_choice))

                print()
                print("inputs", data(rand_choice))
                print("{:15s} {}".format("outputs", test_outputs))
                print("{:15s} {}".format("rounded outputs", [map(int, map(round, test_outputs[0]))]))
                print("{:15s} {}".format("targets", target(rand_choice)))
                print()

                # save summary for Tensorboard
                # writer.add_summary(summary)

        except KeyboardInterrupt:
            break

    test_data = data(test)
    test_outputs = sess.run(outputs, feed_dict=feed(test))
    print()
    print("inputs", test_data)
    print("outputs", test_outputs)
    print("choice", np.argmax(test_outputs))
    print("targets", test_data[0])
    # print_output(test_data, test_outputs)
