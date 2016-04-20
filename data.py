import random
import os
import tensorflow as tf
from enum import Enum

tf.app.flags.DEFINE_integer('num_examples', 1,
                            'number of instances in all datasets')
tf.app.flags.DEFINE_string('directory', 'data',
                           'directory to write training data')
tf.app.flags.DEFINE_string('babi_directory', '../bAbI-tasks',
                           'directory in which to run babi-task generation script')

FLAGS = tf.app.flags.FLAGS


class Dataset(Enum):
    train = .7
    test = .2
    valid = .1


writers = {dataset:
               tf.python_io.TFRecordWriter(
                   os.path.join(FLAGS.directory, dataset.name + '.tfrecords')
               ) for dataset in Dataset}


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


for _ in range(FLAGS.num_examples):
    choice_threshold = random.random()
    for dataset in Dataset:
        if choice_threshold < dataset.value:
            writer = writers[dataset]
        else:
            choice_threshold -= dataset.value
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'question': bytes_feature('how are you?'),
                'answer': bytes_feature('great'),
                'label': bytes_feature('I am great.')
            }))
    writer.write(example.SerializeToString())
