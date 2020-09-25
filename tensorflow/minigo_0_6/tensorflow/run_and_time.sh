#!/bin/bash

# Copyright (c) 2020 Graphcore Ltd.
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

# This file has been modified by Graphcore Ltd.
# It has been modified to run the application on IPU hardware.
# The original file was provided at
# https://github.com/mlperf/training_results_v0.6/blob/master/NVIDIA/benchmarks/minigo/implementations/tensorflow/run_and_time.sh
# The original benchmark code does not provide a definition of the benchmark
# or working execution file.

# Before running, make sure that you follow the Quick start guide in the README.md
# This script requires the appropriate setup of the environment as well
# as the download of the configuration files of the benchmark.

# Exits immediately if subcommand throws error
set -e

cd minigo

BASE_DIR=$(pwd)/results/$(hostname)-$(date +%Y-%m-%d-%H-%M)

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Print trace of simple commands
set -x

# reset IPUs
for i in `seq 0 15`; do gc-reset -d $i || continue; done

echo "running benchmark loop"

# run training benchmark loop
python ml_perf/reference_implementation.py \
  --base_dir=$BASE_DIR \
  --flagfile=ml_perf/flags/9/rl_loop.flags

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# reset IPUs gain
for i in `seq 0 15`; do gc-reset -d $i || continue; done

# run eval
python ml_perf/eval_models.py \
  --base_dir=$BASE_DIR \
  --flags_dir=ml_perf/flags/9/ ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# report result
result=$(( $end - $start ))
result_name="reinforcement"


echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"

# reset to original folder to run script again
cd ..