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
import cv2
import time
import math
import click
import imghdr
import imageio
import argparse
import threading
import collections
import numpy as np
import tensorflow as tf
from tensorflow.summary import FileWriter
from tensorflow.python.framework import graph_util

from .flownet_s import FlowNetS
from ..net import Mode
from ..training_schedules import LONG_SCHEDULE
from ..flow_to_image import flow_to_image, color_function
# from ..flowlib import flow_to_image

parser = argparse.ArgumentParser()
parser.add_argument('--restore_path', type=str, required=True)
args = parser.parse_args()


def init_camera(camera_height, camera_width, device_id=0):
    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return cap


def rescale_frame(frame, ratio):
    height = int(frame.shape[0] * ratio)
    width = int(frame.shape[1] * ratio)
    shape = (width, height)
    return cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # cmap = np.tile(np.linspace(0, 2 * np.pi, 1001), (1001, 1))
    # plt.imshow(color_function(cmap).astype(np.uint8))
    # plt.show()

    # initializing camera
    cap = init_camera(480, 640, 0)

    # initializing worker and variables
    diff_step = 10
    frame_list = collections.deque(maxlen=300)
    image_size = (384, 512, 3)
    input_image = np.zeros(
        (1, *image_size[:2], image_size[-1] * 2)).astype(np.uint8)
    output_flow = np.zeros((1, *image_size[:2], 2))
    output_image = np.zeros(image_size).astype(np.uint8)
    for _ in range(diff_step):
        frame_list.append(np.zeros(image_size).astype(np.uint8))

    def _get_frame():
        while True:
            begin = time.time()
            res, frame = cap.read()
            assert res, "Something wrong occurs with camera!"
            frame_list.append(rescale_frame(frame[:, ::-1, :], 0.8))
            time.sleep(max(0.0, 1.0 / 30 - (time.time() - begin)))

    # Create a new network
    graph = tf.Graph()
    with graph.as_default():
        model = FlowNetS(mode=Mode.TEST)
        training_schedule = LONG_SCHEDULE
        images_placeholder, _ = model.placeholders()
        input_op = {
            'input_a': images_placeholder[..., :3],
            'input_b': images_placeholder[..., 3:],
        }
        predictions = model.model(input_op, training_schedule)
        output_op = predictions['flow']
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=session_config)
    sess.run(init_op)
    saver.restore(sess, args.restore_path)

    def _inference():
        time_list = collections.deque(maxlen=10)
        while True:
            begin = time.time()
            input_image[0, ..., :3] = frame_list[-diff_step]
            input_image[0, ..., 3:] = frame_list[-1]
            feed_dict = {images_placeholder: input_image / 255.0}
            output_flow[:] = sess.run(output_op, feed_dict=feed_dict)
            output_image[:] = flow_to_image(-output_flow[0][..., [1, 0]])
            time_list.append(time.time() - begin)
            print("\033[KFPS: {:.3f}".format(
                np.mean(1 / np.array(time_list))), end="\r")

    t1 = threading.Thread(target=_get_frame)
    t1.setDaemon(True)
    t2 = threading.Thread(target=_inference)
    t2.setDaemon(True)
    t1.start()
    t2.start()

    while True:
        # cv2.imshow("raw", input_image[0, ..., :3])
        cv2.imshow("comp", np.mean([
            input_image[0, ..., :3],
            input_image[0, ..., 3:]], axis=0).astype(np.uint8))
        _pre = frame_list[-diff_step].astype(np.float)
        _post = frame_list[-1].astype(np.float)
        diff = np.mean([_pre, _post], axis=0).astype(np.uint8)
        # diff = np.mean(
        #     list(frame_list)[::-diff_step][:10], axis=0).astype(np.uint8)
        # output_image_overwrap = np.mean(
        #     [input_image[0, ..., 3:], output_image], axis=0).astype(np.uint8)
        cv2.imshow("diff", diff)
        cv2.imshow("output", output_image)
        key = cv2.waitKey(2)
        if key == 27:
            # imageio.imsave("../test.png", output_image)
            break
