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
import time
import math
import click
import socket
import warnings
import argparse
import threading
import collections
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

from io import BytesIO

from .flownet_s import FlowNetS
from ..net import Mode
from ..training_schedules import LONG_SCHEDULE
from ..flow_to_image import flow_to_image, color_function

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default=socket.gethostname())
parser.add_argument('--port', type=int, default=8888)
parser.add_argument('--restore_path', type=str, required=True)
parser.add_argument('--return_image', action="store_true")
parser.add_argument('--threshold', type=float, default=10.0)
args = parser.parse_args()


class Inference(object):
    def __init__(self, restore_path):
        graph = tf.Graph()
        with graph.as_default():
            self.model = FlowNetS(mode=Mode.TEST)
            self.images_placeholder, _ = self.model.placeholders()
            self.input_op = {
                'input_a': self.images_placeholder[..., :3],
                'input_b': self.images_placeholder[..., 3:],
            }
            predictions = self.model.model(self.input_op, LONG_SCHEDULE)
            self.output_op = predictions['flow']
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=session_config)
        self.sess.run(init_op)
        saver.restore(self.sess, restore_path)

    def __call__(self, input_data):
        feed_dict = {self.images_placeholder: input_data * (1 / 255.0)}
        # begin = time.time()
        # print("\033[Ktime: {:.4f}".format(time.time() - begin))
        t_begin = time.time()
        output = self.sess.run(self.output_op, feed_dict=feed_dict)
        calc_time = time.time() - t_begin
        if args.return_image:
            output = flow_to_image(
                -output[0][..., [1, 0]], threshold=args.threshold)
        return output, calc_time


def receive_and_send(connection, process_func):
    with connection as c:
        data_buffer = b""
        while True:
            received_buffer = c.recv(8192)
            if not received_buffer:
                break
            data_buffer += received_buffer
            if data_buffer[-7:] == b"__end__":
                break
        try:
            input_data = np.load(BytesIO(data_buffer))['input']
            output_data, calc_time = process_func(input_data)
            f = BytesIO()
            np.savez_compressed(f, output=output_data, calc_time=calc_time)
            f.seek(0)
            c.sendall(f.read())
        except ValueError:
            pass


def run_server(server_info, restore_path):
    inference_model = Inference(restore_path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(server_info)
        s.listen(32)
        print("boot: {}:{}".format(*server_info))
        while True:
            client_conn, client_addr = s.accept()
            print("\033[Kfrom: {}:{}".format(*client_addr), end="\r")
            try:
                th = threading.Thread(
                    target=receive_and_send,
                    args=(client_conn, inference_model), daemon=True)
                th.start()
                # th.join()
                # receive_and_send(client_conn, inference_model)
            except BrokenPipeError:
                print("Send data aborted!")
                pass


if __name__ == '__main__':
    run_server((args.host, args.port), args.restore_path)
