from __future__ import print_function

import csv
import random
import subprocess

import os
import tensorflow as tf
from progressbar import ProgressBar
from enum import Enum

tf.app.flags.DEFINE_integer('num_examples', 2200,
                            'number of instances in all datasets')
tf.app.flags.DEFINE_string('directory', 'data',
                           'directory to write training data')
tf.app.flags.DEFINE_string('babi_directory', '../bAbI-tasks',
                           'directory in which to run babi-task generation script')

FLAGS = tf.app.flags.FLAGS

TASKS = [1, 4, 5, 6, 9, 12]


class Dataset(Enum):
    valid = .1
    test = .2
    train = .7


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

instances = set()

handles = []

try:
    writers = {}

    for dataset in Dataset:
        path = os.path.join(FLAGS.directory, dataset.name + '.tfrecords')
        # writers[dataset] = tf.python_io.TFRecordWriter(path)
        f = open(path, 'w')
        handles.append(f)
        writers[dataset] = csv.writer(f)

    num_duplicates = 0
    for _ in ProgressBar()(range(FLAGS.num_examples)):
        task = random.choice(TASKS)
        babi_string, _ = subprocess.Popen(
            ["/Users/Ethan/torch/install/bin/babi-tasks",
             str(task), "--path-length=1", "--decoys=0"],
            stdout=subprocess.PIPE).communicate()
        babi_output = babi_string.split('\n')
        for line in babi_output:
            entities = line.split('\t')
            try:
                if entities[0][-1] == '?':
                    question, answer, sentenceNum = entities
                    i = int(sentenceNum)
                    sentence = babi_output[i - 1].split('\t')[0]
                    question, sentence = (x[2:] for x in (question, sentence))
                    break
            except IndexError:
                pass

        choice_threshold = random.random()
        for ds in Dataset:
            if choice_threshold < ds.value:
                dataset = ds
                break
            else:
                choice_threshold -= dataset.value

        key = question + answer + sentence
        if key not in instances:
            instances.add(key)
            writers[dataset].writerow((question, answer, sentence))
            # q, a, l = map(bytes_feature, (question, answer, sentence))
            # example = tf.train.Example(
            #     features=tf.train.Features(
            #         feature={
            #             'question': q,
            #             'answer': a,
            #             'label': l
            #         }))
            # writer.write(example.SerializeToString())
        else:
            num_duplicates += 1
finally:
    for handle in handles:
        handle.close()

print("number of duplicates: ", num_duplicates)
