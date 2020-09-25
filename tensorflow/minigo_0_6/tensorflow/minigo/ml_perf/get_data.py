# Copyright (c) 2020 Graphcore Ltd.
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file has been modified by Graphcore Ltd.
# It has been modified to run the application on IPU hardware.

"""Copies & compiles the data required to start the reinforcement loop."""

import sys
sys.path.insert(0, '.')  # nopep8

import asyncio
import glob
import os

from absl import app, flags
from ml_perf import utils

N = os.environ.get('BOARD_SIZE', '9')

flags.DEFINE_string('src_dir', 'gs://minigo-pub/ml_perf/0.6/',
                    'Directory on GCS to copy source data from. Files will be '
                    'copied from subdirectories of src_dir corresponding to '
                    'the BOARD_SIZE environment variable (defaults to 9).')

flags.DEFINE_string('dst_dir', 'ml_perf/',
                    'Local directory to write to. Files will be written to '
                    'subdirectories of dst_dir corresponding to the BOARD_SIZE '
                    'environment variable (defaults to 9).')

flags.DEFINE_boolean('use_tpu', False,
                     'Set to true to generate models that can run on Cloud TPU')

FLAGS = flags.FLAGS


def freeze_graph(path):
  utils.wait(utils.checked_run(
      'python', 'freeze_graph.py',
      '--model_path={}'.format(path), '--use_tpu={}'.format(FLAGS.use_tpu)))


def main(unused_argv):
  try:
    for d in ['checkpoint', 'target']:
      # Pull the required training checkpoints and models from GCS.
      dst = os.path.join(FLAGS.dst_dir, d)
      src = os.path.join(FLAGS.src_dir, d)
      utils.ensure_dir_exists(dst)
      utils.wait(utils.checked_run('gsutil', '-m', 'cp', '-r', src, dst))

    # Freeze the target model.
    freeze_graph(os.path.join(FLAGS.dst_dir, 'target', 'target', 'target'))

    # Freeze the training checkpoint models.
    pattern = os.path.join(FLAGS.dst_dir, 'checkpoint', 'checkpoint', 'work_dir', 'work_dir', '*.index')
    patterns = glob.glob(pattern)
    if not patterns:
        raise FileNotFoundError("No files found in {}.".format(pattern))
    for path in patterns:
      # Avoid refreezing previously created checkpoint file.
      if "raw.ckpt" not in path:
        freeze_graph(os.path.splitext(path)[0])

  finally:
    asyncio.get_event_loop().close()


if __name__ == '__main__':
  app.run(main)

