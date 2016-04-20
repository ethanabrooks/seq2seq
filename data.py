from __future__ import print_function

import random
import subprocess

import os
import tensorflow as tf
import progress
from enum import Enum

tf.app.flags.DEFINE_integer('num_examples', 200,
                            'number of instances in all datasets')
tf.app.flags.DEFINE_string('directory', 'data',
                           'directory to write training data')
tf.app.flags.DEFINE_string('babi_directory', '../bAbI-tasks',
                           'directory in which to run babi-task generation script')

FLAGS = tf.app.flags.FLAGS

"""
problem tasks:
2, 3, 7, 11, 13, 14, 15, 16, 17, 18 : multiple sentences
19: single letter answers
20: weird answers
ok tasks:
1, 4, 5, 6, 8, 9, 10, 12
"""

tasks = [1, 4, 5, 6, 8, 9, 10, 12]


class Dataset(Enum):
    train = .7
    test = .2
    valid = .1


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


writers = {dataset:
               tf.python_io.TFRecordWriter(
                   os.path.join(FLAGS.directory, dataset.name + '.tfrecords')
               ) for dataset in Dataset}

for _ in progress.Bar('Generating dataset', max=FLAGS.num_examples):
    task = random.choice(tasks)
    babi_string, _ = subprocess.Popen(
        ["/Users/Ethan/torch/install/bin/babi-tasks",
         str(task), "--path-length=1", "--decoys=0"],
        stdout=subprocess.PIPE).communicate()
    babi_output = babi_string.split('\n')
    for line in babi_output:
        entities = line.split('\t')
        if entities[0][-1] == '?':
            question, answer, sentenceNum = entities
            try:
                sentence = babi_output[int(sentenceNum) - 1].split('\t')[0]
            except ValueError:
                print("task: ", task)
            question, sentence = (x[2:] for x in (question, sentence))
            break

    choice_threshold = random.random()
    for dataset in Dataset:
        if choice_threshold < dataset.value:
            writer = writers[dataset]
        else:
            choice_threshold -= dataset.value

    q, a, l = map(bytes_feature, (question, answer, sentence))
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'question': q,
                'answer': a,
                'label': l
            }))
    writer.write(example.SerializeToString())
    # print('question:\t', question)
    # print('answer:\t\t', answer)
    # print('label:\t\t', sentence)
