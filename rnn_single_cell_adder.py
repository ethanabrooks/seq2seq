from __future__ import print_function

import random
import shutil

import math

import itertools

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell

optimizers = {
    1: tf.train.GradientDescentOptimizer(.1),
    2: tf.train.AdadeltaOptimizer(),
    3: tf.train.AdagradOptimizer(.01),
    4: tf.train.MomentumOptimizer(.1, 1.),
    5: tf.train.AdamOptimizer(.1),
    6: tf.train.FtrlOptimizer(.1),
    7: tf.train.RMSPropOptimizer(.1)
}


class Config:
    def __init__(self):
        self.opt_choice = 3
        self.num_terms = 5
        self.distinct_nums = 5
        self.input_size = 1
        self.size_train = 1024
        self.size_test = 256
        self.num_instances = self.size_train + self.size_test
        self.num_cells = 21

    def __str__(self):
        return str(self.__dict__)


args = Config()
print(args)
print_interval = 100

log_dir = 'summaries'
vocabulary_size = 30
starting_numbers = list(range(vocabulary_size))
embedding_size = int(math.ceil(math.log(vocabulary_size, 2)))
test = random.choice(starting_numbers)
starting_numbers.pop(test)
random.shuffle(starting_numbers)

distinct_nums = vocabulary_size // args.num_terms - 1
product = np.array(list(itertools.product(*(range(distinct_nums)
                                            for _ in range(args.num_terms))))).transpose()


def data():
    # return np.array(list(itertools.product(*(range(args.distinct_nums)
    #                                          for _ in range(args.num_terms))))).transpose()
    return np.random.randint(distinct_nums,
                             size=[args.num_terms, args.num_instances])


def target(data):
    return np.sum(data, 0)


init = tf.random_uniform_initializer()
with tf.Session() as sess, tf.variable_scope("", initializer=init):
    # embeddings

    with tf.device('/cpu:0'):
        inputs = tf.placeholder(tf.int32,
                                shape=[args.num_terms, args.num_instances],
                                name='inputs')
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        lookups = tf.nn.embedding_lookup(embeddings, inputs)
    inputs_list = tf.unpack(lookups)

    # GRU
    cell = tf.nn.rnn_cell.GRUCell(args.num_cells)
    # cells = rnn_cell.MultiRNNCell([cell] * 2)
    lstm_output, outputs = tf.nn.rnn(cell, inputs_list, dtype=tf.float32)

    # Train loss
    targets = tf.placeholder(tf.int64, shape=args.num_instances, name='targets')
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, targets)
    loss = tf.reduce_sum(losses, name='loss')
    train_op = optimizers[args.opt_choice].minimize(loss)

    # Tensorboard
    # shutil.rmtree(log_dir)
    writer = tf.train.SummaryWriter(log_dir, sess.graph)

    tf.initialize_all_variables().run()


    def feed(train=True, print_args=False):
        input_values = data()
        if print_args:
            print("data")
            print(input_values)
            print("target")
            print(target(input_values))
        return {inputs: input_values,
                targets: target(input_values)}


    epoch = 0.0
    prev_cost = 0
    avg_speed = 0
    while True:
        epoch += 1.0
        try:
            cost = 0
            feed_dict = feed()
            _, loss_value, train_outputs = sess.run(
                [train_op, loss, outputs], feed_dict=feed())
            cost += loss_value

            speed = 0 if epoch == 1 else prev_cost - cost
            avg_speed = avg_speed * ((epoch - 1) / epoch) + speed / epoch
            prev_cost = cost

            print('\repoch: {:5.0f} | cost: {:6.1f} | avg speed: {:6.4f} | speed {:6.4f}'
                  .format(epoch, cost, avg_speed, speed), end='')
            if epoch % print_interval == 0:
                test_outputs = sess.run(outputs, feed_dict=feed_dict)

                print()
                print("inputs\n", feed_dict[inputs][:, :10])
                print("{:10}".format("choice"), np.argmax(test_outputs, axis=1)[:10])
                print("{:10}".format("targets"), feed_dict[targets][:10])
                print()

                feed_dict = feed(train=False)
                test_outputs = sess.run(outputs, feed_dict=feed_dict)

                print()
                print("inputs\n", feed_dict[inputs][:, :10])
                print("{:10}".format("choice"), np.argmax(test_outputs, axis=1)[:10])
                print("{:10}".format("targets"), feed_dict[targets][:10])
                print()
                # save summary for Tensorboard
                # writer.add_summary(summary)

        except KeyboardInterrupt:
            break
