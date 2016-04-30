import math

import itertools

import tensorflow as tf
import numpy as np
import uuid

x = np.array([1, 2, 3])
product = random_matrix = np.array(list(itertools.product(*(range(4)
                                                            for _ in range(6)))))
print(product)
print(np.sum(product, 1))
