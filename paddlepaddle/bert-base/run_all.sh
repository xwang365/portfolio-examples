#!/bin/bash

# Copyright (c) 2021 Graphcore Ltd.
#
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# This file has been created by Graphcore Ltd.
# It has been created to run the application on IPU hardware.

# Pretrain Phase1
./run_stage ipu phase1 _ pretrained_model_128

# Pretrain Phase2
./run_stage ipu phase2 pretrained_model_128 pretrained_model_384

# Finetune (SQuAD)
./run_stage ipu SQuAD pretrained_model_384 finetune_model

# Validation
./run_stage ipu validation finetune_model _