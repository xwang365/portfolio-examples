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

"""Runs a reinforcement learning loop to train a Go playing model."""

import sys
sys.path.insert(0, '.')  # nopep8

import asyncio
import glob
import logging
import numpy as np
import os
import random
import re
import shutil
import subprocess
import tensorflow as tf
import time
from ml_perf.utils import *

from absl import app, flags
from rl_loop import example_buffer, fsdb
from tensorflow import gfile

# Train while seflplay is running
PARALLEL_TRAIN = True
# Start multiple separate selfplay processes instead of one distributed one.
# The hope is that it reduces blockage by memory access.
MULTI_SP = True

# Distribute selfplay in an even ratio (6:6) onto two sockets or skewed (4:8)
DISTRIBUTION_STRATEGY = 'even'
# DISTRIBUTION_STRATEGY = 'skewed'


N = int(os.environ.get('BOARD_SIZE', 9))

flags.DEFINE_string('checkpoint_dir',
                    'ml_perf/checkpoint/checkpoint',
                    'The checkpoint directory specify a start model and a set '
                    'of golden chunks used to start training.  If not '
                    'specified, will start from scratch.')

flags.DEFINE_string('target_path', 'ml_perf/target/target/target.pb'.format(N),
                    'Path to the target model to beat.')

flags.DEFINE_integer('iterations', 100, 'Number of iterations of the RL loop.')

flags.DEFINE_float('gating_win_rate', 0.55,
                   'Win-rate against the current best required to promote a '
                   'model to new best.')

flags.DEFINE_string('flags_dir', None,
                    'Directory in which to find the flag files for each stage '
                    'of the RL loop. The directory must contain the following '
                    'files: bootstrap.flags, selfplay.flags, eval.flags, '
                    'train.flags.')

flags.DEFINE_integer('window_size', 10,
                     'Maximum number of recent selfplay rounds to train on.')

flags.DEFINE_boolean('parallel_post_train', False,
                     'If true, run the post-training stages (eval, validation '
                     '& selfplay) in parallel.')

flags.DEFINE_string('engine', 'tf', 'The engine to use for selfplay.')

flags.DEFINE_integer('num_ipus_selfplay', 12,
                     'Number of IPUs to use for selfplay.')


FLAGS = flags.FLAGS


class State:
  """State data used in each iteration of the RL loop.

  Models are named with the current reinforcement learning loop iteration number
  and the model generation (how many models have passed gating). For example, a
  model named "000015-000007" was trained on the 15th iteration of the loop and
  is the 7th models that passed gating.
  Note that we rely on the iteration number being the first part of the model
  name so that the training chunks sort correctly.
  """

  def __init__(self):
    self.start_time = time.time()

    self.iter_num = 0
    self.gen_num = 0

    self.best_model_name = None

  @property
  def output_model_name(self):
    return '%06d-%06d' % (self.iter_num, self.gen_num)

  @property
  def train_model_name(self):
    return '%06d-%06d' % (self.iter_num, self.gen_num + 1)

  @property
  def best_model_path(self):
    if self.best_model_name is None:
      # We don't have a good model yet, use a random fake model implementation.
      return 'random:0,0.4:0.4'
    else:
      return '{},{}.pb'.format(
         FLAGS.engine, os.path.join(fsdb.models_dir(), self.best_model_name))

  @property
  def train_model_path(self):
    return '{},{}.pb'.format(
         FLAGS.engine, os.path.join(fsdb.models_dir(), self.train_model_name))

  @property
  def seed(self):
    return self.iter_num + 1


class ColorWinStats:
  """Win-rate stats for a single model & color."""

  def __init__(self, total, both_passed, opponent_resigned, move_limit_reached):
    self.total = total
    self.both_passed = both_passed
    self.opponent_resigned = opponent_resigned
    self.move_limit_reached = move_limit_reached
    # Verify that the total is correct
    assert total == both_passed + opponent_resigned + move_limit_reached


class WinStats:
  """Win-rate stats for a single model."""

  def __init__(self, line):
    pattern = '\s*(\S+)' + '\s+(\d+)' * 8
    match = re.search(pattern, line)
    if match is None:
      raise ValueError('Can\t parse line "{}"'.format(line))
    self.model_name = match.group(1)
    raw_stats = [float(x) for x in match.groups()[1:]]
    self.black_wins = ColorWinStats(*raw_stats[:4])
    self.white_wins = ColorWinStats(*raw_stats[4:])
    self.total_wins = self.black_wins.total + self.white_wins.total


def initialize_from_checkpoint(state):
  """Initialize the reinforcement learning loop from a checkpoint."""
  # The checkpoint's work_dir should contain the most recently trained model.
  model_paths = glob.glob(os.path.join(FLAGS.checkpoint_dir,
                                       'work_dir/model.ckpt-*.pb'))
  print(os.path.join(FLAGS.checkpoint_dir, 'work_dir/model.ckpt-*.pb'))
  print(os.getcwd())
  if len(model_paths) != 1:
    raise RuntimeError(
      'Expected exactly one model in the checkpoint work_dir'
      '({}), got [{}]'.format(
        os.path.join(FLAGS.checkpoint_dir, 'work_dir'), ', '.join(model_paths)))
  start_model_path = model_paths[0]

  # Copy the latest trained model into the models directory and use it on the
  # first round of selfplay.
  state.best_model_name = 'checkpoint'

  shutil.copy(start_model_path,
              os.path.join(fsdb.models_dir(), state.best_model_name + '.pb'))

  start_model_files = glob.glob(os.path.join(
    FLAGS.checkpoint_dir, 'work_dir/model.ckpt-9383_raw.ckpt*'))

  for file_name in start_model_files:
    shutil.copy(file_name,
        os.path.join(fsdb.models_dir(),
                     state.best_model_name +
                     os.path.basename(file_name)[len("model.ckpt-9383"):]))

  # Copy the training chunks.
  golden_chunks_dir = os.path.join(FLAGS.checkpoint_dir, "..", 'golden_chunks')
  for basename in os.listdir(golden_chunks_dir):
    path = os.path.join(golden_chunks_dir, basename)
    shutil.copy(path, fsdb.golden_chunk_dir())

  # Copy the training files.
  work_dir = os.path.join(FLAGS.checkpoint_dir, 'work_dir')
  for basename in os.listdir(work_dir):
    path = os.path.join(work_dir, basename)
    shutil.copy(path, fsdb.working_dir())


def parse_win_stats_table(stats_str, num_lines):
  result = []
  lines = stats_str.split('\n')
  while True:
    # Find the start of the win stats table.
    assert len(lines) > 1
    if 'Black' in lines[0] and 'White' in lines[0] and 'm.lmt.' in lines[1]:
        break
    lines = lines[1:]

  # Parse the expected number of lines from the table.
  for line in lines[2:2 + num_lines]:
    result.append(WinStats(line))

  return result


async def run(*cmd):
  """Run the given subprocess command in a coroutine.

  Args:
    *cmd: the command to run and its arguments.

  Returns:
    The output that the command wrote to stdout as a list of strings, one line
    per element (stderr output is piped to stdout).

  Raises:
    RuntimeError: if the command returns a non-zero result.
  """

  stdout = await checked_run(*cmd)

  log_path = os.path.join(FLAGS.base_dir, get_cmd_name(cmd) + '.log')
  with gfile.Open(log_path, 'a') as f:
    f.write(expand_cmd_str(cmd))
    f.write('\n')
    f.write(stdout)
    f.write('\n')

  # Split stdout into lines.
  return stdout.split('\n')


def get_golden_chunk_records():
  """Return up to num_records of golden chunks to train on.

  Returns:
    A list of golden chunks up to num_records in length, sorted by path.
  """

  pattern = os.path.join(fsdb.golden_chunk_dir(), '*.zz')
  return sorted(tf.gfile.Glob(pattern), reverse=True)[:FLAGS.window_size]


# Self-play a number of games.
async def selfplay(state, flagfile='selfplay', seed_factor=0):
  """Run selfplay and write a training chunk to the fsdb golden_chunk_dir.

  Args:
    state: the RL loop State instance.
    flagfile: the name of the flagfile to use for selfplay, either 'selfplay'
        (the default) or 'boostrap'.
    seed_factor: Factor to increase seed.
  """
  output_dir = os.path.join(fsdb.selfplay_dir(), state.output_model_name)
  holdout_dir = os.path.join(fsdb.holdout_dir(), state.output_model_name)

  lines = await run(
      'bazel-bin/cc/selfplay',
      '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
      '--model={}'.format(get_ckpt_path(state.best_model_path)),
      '--output_dir={}'.format(output_dir),
      '--holdout_dir={}'.format(holdout_dir),
      '--seed={}'.format(state.seed+100*seed_factor))
  result = '\n'.join(lines[-6:])
  logging.info(result)
  result = '\n'.join(lines[-50:])
  try:
      stats = parse_win_stats_table(result, 1)[0]
      num_games = stats.total_wins
      logging.info('Black won %0.3f, white won %0.3f',
                   stats.black_wins.total / num_games,
                   stats.white_wins.total / num_games)
  except AssertionError:
    # Poplar logging might screw up lines extraction approach.
    logging.error("No results to parse: \n %s" % lines[-50:])

  if not MULTI_SP:
    # Write examples to a single record.
    pattern = os.path.join(output_dir, '*', '*.zz')
    random.seed(state.seed)
    tf.set_random_seed(state.seed)
    np.random.seed(state.seed)
    # TODO(tommadams): This method of generating one golden chunk per generation
    # is sub-optimal because each chunk gets reused multiple times for training,
    # introducing bias. Instead, a fresh dataset should be uniformly sampled out
    # of *all* games in the training window before the start of each training run.
    buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)

    # TODO(tommadams): parallel_fill is currently non-deterministic. Make it not
    # so.
    logging.info('Writing golden chunk from "{}"'.format(pattern))
    buffer.parallel_fill(tf.gfile.Glob(pattern))
    buffer.flush(os.path.join(fsdb.golden_chunk_dir(),
                              state.output_model_name + '.tfrecord.zz'))


async def selfplay_sub(state, output_dir, holdout_dir, flagfile, seed_factor):
  """Run a single selfplay C++ process"""
  if DISTRIBUTION_STRATEGY == "even":
    if seed_factor <= (FLAGS.num_ipus_selfplay-1) / 2:
      membind = 0
    else:
      membind = 1
  elif DISTRIBUTION_STRATEGY == "skewed":
    if seed_factor < 8:
      membind = 0
    else:
      membind = 1

  lines = await run(
    'numactl',
    '--cpunodebind={}'.format(membind),
    '--membind={}'.format(membind),
    'bazel-bin/cc/selfplay',
    '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
    '--model={}'.format(get_ckpt_path(state.best_model_path)),
    '--output_dir={}'.format(output_dir),
    '--holdout_dir={}'.format(holdout_dir),
    '--seed={}'.format(state.seed + 100 * seed_factor))
  return lines


async def selfplay_multi(state, num_ipus):
  """ Start *num_ipu* selfplay processes """
  output_dir = os.path.join(fsdb.selfplay_dir(), state.output_model_name)
  holdout_dir = os.path.join(fsdb.holdout_dir(), state.output_model_name)
  flagfile = 'selfplay'

  all_tasks = []
  loop = asyncio.get_event_loop()
  for i in range(num_ipus):
    all_tasks.append(loop.create_task(selfplay_sub(state, output_dir, holdout_dir, flagfile, i)))
  all_lines = await asyncio.gather(*all_tasks, return_exceptions=True)

  black_wins_total = white_wins_total = num_games = 0
  for lines in all_lines:
    if type(lines) == RuntimeError or type(lines) == OSError:
      raise lines
    result = '\n'.join(lines[-6:])
    logging.info(result)
    stats = parse_win_stats_table(result, 1)[0]
    num_games += stats.total_wins
    black_wins_total += stats.black_wins.total
    white_wins_total += stats.white_wins.total

  logging.info('Black won %0.3f, white won %0.3f',
               black_wins_total / num_games,
               white_wins_total / num_games)

  # copy paste from selfplay to aggregate results
  # potentially should be parallized to training?

  # Write examples to a single record.
  pattern = os.path.join(output_dir, '*', '*.zz')
  random.seed(state.seed)
  tf.set_random_seed(state.seed)
  np.random.seed(state.seed)
  # TODO(tommadams): This method of generating one golden chunk per generation
  # is sub-optimal because each chunk gets reused multiple times for training,
  # introducing bias. Instead, a fresh dataset should be uniformly sampled out
  # of *all* games in the training window before the start of each training run.
  buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)

  # TODO(tommadams): parallel_fill is currently non-deterministic. Make it not
  # so.
  logging.info('Writing golden chunk from "{}"'.format(pattern))
  buffer.parallel_fill(tf.gfile.Glob(pattern))
  buffer.flush(os.path.join(fsdb.golden_chunk_dir(),
                            state.output_model_name + '.tfrecord.zz'))


async def train(state, tf_records=None):
  """Run training and write a new model to the fsdb models_dir.

  Args:
    state: the RL loop State instance.
    tf_records: a list of paths to TensorFlow records to train on.
  """
  if tf_records is None:
    # Train on shuffled game data from recent selfplay rounds.
    tf_records = get_golden_chunk_records()
  model_path = os.path.join(fsdb.models_dir(), state.train_model_name)
  if DISTRIBUTION_STRATEGY == "even":
    await run(
        'numactl',
        '--cpunodebind={}'.format(1),
        '--membind={}'.format(1),
        'python3', 'train.py', *tf_records,
        '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
        '--work_dir={}'.format(fsdb.working_dir()),
        '--export_path={}'.format(model_path),
        '--training_seed={}'.format(state.seed),
        '--freeze=true')
  elif DISTRIBUTION_STRATEGY == "skewed":
    await run(
        'numactl',
        '--cpunodebind={}'.format(1),
        '--membind={}'.format(1),
        'python3', 'train.py', *tf_records,
        '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
        '--work_dir={}'.format(fsdb.working_dir()),
        '--export_path={}'.format(model_path),
        '--training_seed={}'.format(state.seed),
        '--freeze=true')
  # Append the time elapsed from when the RL was started to when this model
  # was trained.
  elapsed = time.time() - state.start_time
  timestamps_path = os.path.join(fsdb.models_dir(), 'train_times.txt')
  with gfile.Open(timestamps_path, 'a') as f:
    print('{:.3f} {}'.format(elapsed, state.train_model_name), file=f)


async def validate(state, holdout_glob):
  """Validate the trained model against holdout games.

  Args:
    state: the RL loop State instance.
    holdout_glob: a glob that matches holdout games.
  """

  if not glob.glob(holdout_glob):
    print('Glob "{}" didn\'t match any files, skipping validation'.format(
          holdout_glob))
  else:
    await run(
        'python3', 'validate.py', holdout_glob,
        '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'validate.flags')),
        '--work_dir={}'.format(fsdb.working_dir()))


def get_ckpt_path(model_path):
  return model_path[:-len(".pb")] + "_raw.ckpt"


async def evaluate_model(eval_model_path, target_model_path, sgf_dir, seed):
  """Evaluate one model against a target.

  Args:
    eval_model_path: the path to the model to evaluate.
    target_model_path: the path to the model to compare to.
    sgf_dir: directory path to write SGF output to.
    seed: random seed to use when running eval.

  Returns:
    The win-rate of eval_model against target_model in the range [0, 1].
  """
  flagfile = 'eval'
  if DISTRIBUTION_STRATEGY == "even":
    lines = await run(
        'numactl',
        '--cpunodebind={}'.format(1),
        '--membind={}'.format(1),
        'bazel-bin/cc/eval',
        '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
        '--model={}'.format(get_ckpt_path(eval_model_path)),
        '--model_two={}'.format(get_ckpt_path(target_model_path)),
        '--sgf_dir={}'.format(sgf_dir),
        '--seed={}'.format(seed))
  elif DISTRIBUTION_STRATEGY == "skewed":
    lines = await run(
        'numactl',
        '--cpunodebind={}'.format(1),
        '--membind={}'.format(1),
        'bazel-bin/cc/eval',
        '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
        '--model={}'.format(get_ckpt_path(eval_model_path)),
        '--model_two={}'.format(get_ckpt_path(target_model_path)),
        '--sgf_dir={}'.format(sgf_dir),
        '--seed={}'.format(seed))

  result = '\n'.join(lines[-7:])
  logging.info(result)
  eval_stats, target_stats = parse_win_stats_table(result, 2)
  num_games = eval_stats.total_wins + target_stats.total_wins
  win_rate = eval_stats.total_wins / num_games
  logging.info('Win rate %s vs %s: %.3f', eval_stats.model_name,
               target_stats.model_name, win_rate)
  return win_rate


async def evaluate_trained_model(state):
  """Evaluate the most recently trained model against the current best model.

  Args:
    state: the RL loop State instance.
  """

  return await evaluate_model(
      state.train_model_path, state.best_model_path,
      os.path.join(fsdb.eval_dir(), state.train_model_name), state.seed)


def rl_loop():
  """The main reinforcement learning (RL) loop."""

  state = State()

  if FLAGS.checkpoint_dir:
    # Start from a partially trained model.
    initialize_from_checkpoint(state)
  else:
    # Play the first round of selfplay games with a fake model that returns
    # random noise. We do this instead of playing multiple games using a single
    # model bootstrapped with random noise to avoid any initial bias.
    wait(selfplay(state, 'bootstrap'))

    # Train a real model from the random selfplay games.
    tf_records = get_golden_chunk_records()
    state.iter_num += 1
    wait(train(state, tf_records))

    # Select the newly trained model as the best.
    state.best_model_name = state.train_model_name
    state.gen_num += 1

    # Run selfplay using the new model.
    wait(selfplay(state))

  # Now start the full training loop.
  while state.iter_num <= FLAGS.iterations:
    # Build holdout glob before incrementing the iteration number because we
    # want to run validation on the previous generation.
    holdout_glob = os.path.join(fsdb.holdout_dir(), '%06d-*' % state.iter_num,
                                '*')
    if not PARALLEL_TRAIN:
      # Train on shuffled game data from recent selfplay rounds.
      tf_records = get_golden_chunk_records()
      state.iter_num += 1
      wait(train(state, tf_records))

    if FLAGS.parallel_post_train:

      async def evaluate_validate(state, holdout_glob):
        """First run validate and then evaluate to share IPUs"""
        await validate(state, holdout_glob)
        return await evaluate_trained_model(state)

      async def train_evaluate_validate(state, holdout_glob):
        """First run validate and then evaluate to share IPUs"""
        await train(state, tf_records=None)
        # await validate(state, holdout_glob)
        return await evaluate_trained_model(state)

      if PARALLEL_TRAIN:
        state.iter_num += 1

      if not PARALLEL_TRAIN:
        # results = wait([evaluate_trained_model(state), selfplay(state), validate(state, holdout_glob)])
        results = wait([evaluate_validate(state, holdout_glob), selfplay(state)])
        model_win_rate = results[0]
      elif PARALLEL_TRAIN and not MULTI_SP:
        results = wait([train_evaluate_validate(state, holdout_glob), selfplay(state)])
        model_win_rate = results[0]
      elif PARALLEL_TRAIN and MULTI_SP:
        results = wait(
          [train_evaluate_validate(state, holdout_glob), selfplay_multi(state, FLAGS.num_ipus_selfplay)]
        )

        model_win_rate = results[0]
    else:
      # Run eval, validation & selfplay sequentially.
      model_win_rate = wait(evaluate_trained_model(state))
      wait(validate(state, holdout_glob))
      wait(selfplay(state))

    if model_win_rate >= FLAGS.gating_win_rate:
      # Promote the trained model to the best model and increment the generation
      # number.
      state.best_model_name = state.train_model_name
      state.gen_num += 1


def main(unused_argv):
  """Run the reinforcement learning loop."""

  print('Wiping dir %s' % FLAGS.base_dir, flush=True)
  shutil.rmtree(FLAGS.base_dir, ignore_errors=True)
  dirs = [fsdb.models_dir(), fsdb.selfplay_dir(), fsdb.holdout_dir(),
          fsdb.eval_dir(), fsdb.golden_chunk_dir(), fsdb.working_dir()]
  for d in dirs:
    ensure_dir_exists(d);

  # Copy the flag files so there's no chance of them getting accidentally
  # overwritten while the RL loop is running.
  flags_dir = os.path.join(FLAGS.base_dir, 'flags')
  shutil.copytree(FLAGS.flags_dir, flags_dir)
  FLAGS.flags_dir = flags_dir

  # Copy the target model to the models directory so we can find it easily.
  for file_name in [
        "target.pb", "target_raw.ckpt.data-00000-of-00001",
        "target_raw.ckpt.index", "target_raw.ckpt.meta"]:
    shutil.copy(FLAGS.target_path[:-len("target.pb")] + file_name,
                os.path.join(fsdb.models_dir(), file_name))

  logging.getLogger().addHandler(
      logging.FileHandler(os.path.join(FLAGS.base_dir, 'rl_loop.log')))
  formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                '%Y-%m-%d %H:%M:%S')
  for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

  with logged_timer('Total time'):
    try:
      rl_loop()
    finally:
      asyncio.get_event_loop().close()


if __name__ == '__main__':
  app.run(main)
