from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse

import tensorflow as tf

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINing_FILE = 'adult.data'
TRAIN_URL = '%s%s' % (DATA_URL, TRAINing_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s%s' % (DATA_URL, EVAL_FILE)


