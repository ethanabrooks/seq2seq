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


class Args:
    def __init__(self):
        self.opt_choice = 3
        self.num_terms = 5
        self.distinct_nums = 5
        self.vocabulary_size = self.distinct_nums * self.num_terms
        self.num_instances = self.distinct_nums ** self.num_terms / 2
        self.num_cells = self.vocabulary_size
        self.fold = 3
        self.batch_size = self.num_instances // self.fold

    def __str__(self):
        return str(self.__dict__)


args = Args()
print(args)
print_interval = 100

log_dir = 'summaries'
starting_numbers = list(range(args.vocabulary_size))
embedding_size = int(math.ceil(math.log(args.vocabulary_size, 2)))
test = random.choice(starting_numbers)
starting_numbers.pop(test)
random.shuffle(starting_numbers)

product_array = np.array(list(itertools.product(*(range(args.distinct_nums)
                                                  for _ in range(args.num_terms)))))
np.random.shuffle(product_array)


def data():
    # return np.random.randint(args.distinct_nums,
    #                          size=[args.distinct_nums, 1562])
    return product_array.transpose()


def target(data):
    return np.sum(data, 0)


init = tf.random_uniform_initializer()
with tf.Session() as sess, tf.variable_scope("", initializer=init):
    # embeddings
    inputs = tf.placeholder(tf.int32,
                            shape=[args.num_terms, args.batch_size],
                            name='inputs')
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([args.vocabulary_size, embedding_size], -1.0, 1.0),
                                 name='embeddings')
        lookups = tf.nn.embedding_lookup(embeddings, inputs, name='lookups')
    inputs_list = tf.unpack(lookups)

    # GRU
    cell = tf.nn.rnn_cell.GRUCell(args.num_cells)
    lstm_output, outputs = tf.nn.rnn(cell, inputs_list, dtype=tf.float32)

    # TODO: add matrix mult at the end so that lstm can learn sparse repr

    # Train loss
    targets = tf.placeholder(tf.int64, shape=args.batch_size, name='targets')
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, targets)
    loss = tf.reduce_sum(losses, name='loss')
    train_op = optimizers[args.opt_choice].minimize(loss)

    # Tensorboard
    # shutil.rmtree(log_dir)
    writer = tf.train.SummaryWriter(log_dir, sess.graph)

    tf.initialize_all_variables().run()


    def feed(batch, print_args=False):
        input_values = data()
        start = args.batch_size * batch
        end = start + args.batch_size
        input_values = input_values[:, start:end]
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
            for batch in range(2):
                _, loss_value, train_outputs = sess.run(
                    [train_op, loss, outputs], feed_dict=feed(batch))
                cost += loss_value

            speed = 0 if epoch == 1 else prev_cost - cost
            avg_speed = avg_speed * ((epoch - 1) / epoch) + speed / epoch
            prev_cost = cost

            print('\repoch: {:5.0f} | cost: {:6.1f} | avg speed: {:6.4f} | speed {:6.4f}'
                  .format(epoch, cost, avg_speed, speed), end='')
            if epoch % print_interval == 0:
                feed_dict = feed(batch=0)
                test_outputs = sess.run(outputs, feed_dict=feed_dict)

                print()
                print("TRAIN")
                print("inputs")
                print(feed_dict[inputs][:, :10])
                print("{:10}".format("choice"), np.argmax(test_outputs, axis=1)[:10])
                print("{:10}".format("targets"), feed_dict[targets][:10])
                print()

                print()
                print("TEST")
                feed_dict = feed(batch=2)
                test_outputs = sess.run(outputs, feed_dict=feed_dict)
                print("inputs")
                print(feed_dict[inputs][:, :10])
                choices = np.argmax(test_outputs, axis=1).round(0)
                print("{:10}".format("choice"), choices[:10])
                print("{:10}".format("targets"), feed_dict[targets][:10])
                accuracy = (choices == feed_dict[targets]).sum() / float(choices.size)
                print("\n >>> accuracy: {} <<< \n".format(accuracy))
                # save summary for Tensorboard
                # writer.add_summary(summary)

        except KeyboardInterrupt:
            break
