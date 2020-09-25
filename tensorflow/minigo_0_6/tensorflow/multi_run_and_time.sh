#!/bin/bash

# Copyright (c) 2020 Graphcore Ltd.
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
#
# This file has been added by Graphcore Ltd.

# This script configures the benchmark and then executes it 10 times.

# Exits immediately if subcommand throws error
set -e

# get ipu configuration graphs
python ipu_reservation_graph.py

cd minigo
# get data and target model
python ml_perf/get_data.py

cd ..

for i in $(seq 0 9); do bash run_and_time.sh 2>&1 | tee $(pwd)/minigo/results/$(hostname)-$(date +%Y-%m-%d-%H-%M)_"$i".log; done

echo "done"