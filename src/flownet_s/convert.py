# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
import sys
import click
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.summary import FileWriter
from tensorflow.python.framework import graph_util

from .flownet_s import FlowNetS
from ..net import Mode
from ..training_schedules import LONG_SCHEDULE

parser = argparse.ArgumentParser()
parser.add_argument('--restore_path', type=str, required=True)
args = parser.parse_args()


def main():
    dir_name, file_name = os.path.split(args.restore_path)

    graph = tf.Graph()
    with graph.as_default():
        net = FlowNetS(mode=Mode.TEST)
        images_placeholder, _ = net.placeholders()
        image_a, image_b = tf.split(
            images_placeholder, num_or_size_splits=2, axis=3)
        inputs = {
            'input_a': image_a,
            'input_b': image_b,
        }
        predict_flow = net.model(inputs, LONG_SCHEDULE, trainable=False)
        labels = tf.identity(predict_flow["flow"], name="output")
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=50)

    session_config = tf.ConfigProto()
    sess = tf.Session(graph=graph, config=session_config)
    sess.run(init_op)

    saver.restore(sess, args.restore_path)
    FileWriter("__tb", sess.graph)
    graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(add_shapes=True), ['output'])
    tf.train.write_graph(
        graph, dir_name, 'flownet_s.pb', as_text=False)


if __name__ == '__main__':
    main()
