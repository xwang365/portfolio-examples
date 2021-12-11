# Paddle-BERT with Graphcore IPUs

Table of contents
=================
<!--ts-->
   * [Overview](#overview)
   * [File Structure](#filestructure)
   * [Dataset](#dataset)
   * [Changelog](#changelog)
   * [Quick start guide](#gettingstarted)
   * [Result](#result)
   * [Licensing](#licensing)

<!--te-->

## Overview

This project aims to build BERT-Base pre-training and SQuAD fine-tuning task using [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) on Graphcore [IPU-POD16](https://www.graphcore.ai/products/mk2/ipu-pod16).

## File Structure

| File                         | Description                                                             |
| ---------------------------- | ----------------------------------------------------------------------- |
| `README.md`                  | How to run the model.                                                   |
| `run_pretrain.py`            | The algorithm script to run pre-training tasks (phase1 and phase2).     |
| `run_squad.py`               | The algorithm script to run SQuAD fine-tuning task.                     |
| `run_squad_infer.py`         | The algorithm script to run SQuAD validation task.                      |
| `modeling.py`                | The algorithm script to build the Bert-Base model.                      |
| `dataset_ipu.py`             | The algorithm script to load input data in pre-training.                |
| `run_stage.sh`               | Test script to run single stage (phase1, phase2, SQuAD and validation). |
| `run_all.sh`                 | Test script to run all of stages.                                       |
| `LICENSE`                    | The license of Apache.                                                  |

## Dataset

1. Pre-training dataset

   Refer to the Wikipedia dataset generator provided by Nvidia (https://github.com/NVIDIA/DeepLearningExamples.git).

   Generate sequence_length=128 and 384 datasets for pre-training phase1 and phase2 respectively.

   ```
   Code base：https://github.com/NVIDIA/DeepLearningExamples/tree/88eb3cff2f03dad85035621d041e23a14345999e/TensorFlow/LanguageModeling/BERT

   git clone https://github.com/NVIDIA/DeepLearningExamples.git

   cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT

   bash scripts/docker/build.sh

   cd data/

   vim create_datasets_from_start.sh

   Modified the line 40 `--max_seq_length 512` as `--max_seq_length 384`, and the line 41 `--max_predictions_per_seq 80` as `--max_predictions_per_seq 56`.

   cd ../

   bash scripts/data_download.sh wiki_only
   ```

2. SQuAD 1.1 dataset

   ```
   curl --create-dirs -L https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -o data/squad/train-v1.1.json

   curl --create-dirs -L https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -o data/squad/dev-v1.1.json
   ```

## Changelog

### Changed

- Modified `run_pretrain.py` (https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/language_model/bert/static/run_pretrain.py) to run the application on the Graphcore IPUs.
- Modified `modeling.py` (https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/bert/modeling.py) to support graph sharding and pipelining.

### Added

- Added `README.md` to introduce how to run the Bert-Base model.
- Added `run_squad.py` to run the SQuAD fine-tuning task.
- Added `run_squad_infer.py` to run the SQuAD validation.
- Added `dataset_ipu.py` to load input data in pre-training.
- Added `run_stage.sh` to run the single task (phase1, phase2, SQuAD and validation).
- Added `run_all.sh` to run the complete process.

## Quick Start Guide

### 1）Prepare Project Environment

#### Poplar SDK

`poplar_sdk-ubuntu_18_04-2.3.0+774-b47c577c2a`

#### PaddlePaddle

- Create Docker container

```
git clone -b paddle_bert_release https://github.com/graphcore/Paddle.git

cd Paddle

# build docker image

docker build -t paddlepaddle/paddle:dev-ipu-2.3.0 -f tools/dockerfile/Dockerfile.ipu .

# The ipu.conf is required here. if the ipu.conf is available, please make sure `${HOST_IPUOF_PATH}` is the right dir of the ipu.conf. 
# If the ipu.conf is not available, please follow the instruction below to generate a ipu.conf.

vipu create partition ipu --size 16

# Then the ipu.conf is able to be found in the dir below.

ls ~/.ipuof.conf.d/

# create container

docker run --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK \
--device=/dev/infiniband/ --ipc=host --name paddle-ipu-dev \
-v ${HOST_IPUOF_PATH}:/ipuof \
-e IPUOF_CONFIG_PATH=/ipuof/ipu.conf \
-it paddlepaddle/paddle:dev-ipu-2.3.0 bash
```

All of later processes are required to be executed in the container.

- Compile and installation

```
git clone -b paddle_bert_release https://github.com/graphcore/Paddle.git

cd Paddle

cmake -DPYTHON_EXECUTABLE=/usr/bin/python \
-DWITH_PYTHON=ON -DWITH_IPU=ON -DPOPLAR_DIR=/opt/poplar \
-DPOPART_DIR=/opt/popart -G "Unix Makefiles" -H`pwd` -B`pwd`/build

cmake --build `pwd`/build --config Release --target paddle_python -j$(nproc)

pip3.7 install -U build/python/dist/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl
```

#### PaddleNLP

```
pip3.7 install jieba h5py colorlog colorama seqeval multiprocess numpy==1.19.2 paddlefsl==1.0.0 six==1.13.0 wandb

pip3.7 install git+https://github.com/graphcore/PaddleNLP.git@paddle_bert_release
```

#### Others

```
pip3.7 install torch==1.7.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

pip3.7 install torch-xla@https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.7-cp37-cp37m-linux_x86_64.whl
```

### 2) Execution

#### Run the single task (optional)

Please check the `--input_dir` in `run_stage.sh` and make sure the dir of input data is right.

The type of the input data in `Phase1(Pre-training)` is the `tfrecord` (sequence_length = 128).

The type of the input data in `Phase2(Pre-training)` is the `tfrecord` (sequence_length = 384).

The name of the input data in `Fine-tuning` is the `train-v1.1.json`.

The name of the input data in `Validation` is the `dev-v1.1.json`.

- Run pre-training phase1 (sequence_length = 128)

```
./run_stage.sh ipu phase1 _ pretrained_128_model
```

- Run pre-training phase2 (sequence_length = 384)

```
./run_stage.sh ipu phase2 pretrained_128_model pretrained_384_model
```

- Run SQuAD fine-tuning task

```
./run_stage.sh ipu SQuAD pretrained_384_model finetune_model
```

- Run SQuAD validation

```
./run_stage.sh ipu validation finetune_model _
```

#### Run the complete process (optional)

```
./run_all.sh
```

#### Parameters

- `model_type` The type of the NLP model.
- `model_name_or_path` The model configuration.
- `input_dir` The directory of the input data.
- `output_dir` The directory of the trained models.
- `seq_len` The sequence length.
- `max_predictions_per_seq` The max number of the masked token each sentence.
- `learning_rate` The learning rate for training.
- `weight_decay` The weight decay.
- `max_steps` The max training steps.
- `warmup_steps` The warmup steps used to update learning rate with lr_schedule.
- `logging_steps` The gap steps of logging.
- `seed` The random seed.
- `device` The type of device. 'ipu': Graphcore IPU, 'cpu': CPU.
- `num_ipus` The number of IPUs.
- `num_hidden_layers` The number of encoder layers.
- `micro_batch_size` The batch size of the IPU graph.
- `ipu_enable_fp16` Enable FP16 or not.
- `scale_loss` The loss scaling.
- `save_init_onnx` Save the initial onnx graph or not.
- `save_per_n_step` Sync the weights D2H every n steps.
- `save_steps` Save the paddle model every n steps.
- `optimizer_type` The type of the optimizer.
- `enable_pipelining` Enable pipelining or not.
- `batches_per_step` The number of batches per step with pipelining.
- `enable_replica` Enable graph replication or not.
- `num_replica` The number of the graph replication.
- `enable_grad_acc` Enable gradiant accumulation or not.
- `grad_acc_factor` Update the weights every n batches.
- `batch_size` total batch size (= batches_per_step \* num_replica \* grad_acc_factor \* micro_batch_size).
- `enable_recompute`Enable recompute or not.
- `enable_half_partial` Enable matmul fp16 partial or not.
- `available_mem_proportion` The available proportion of memory used by conv or matmul.
- `check_data` Enable to check input data or not.
- `ignore_index` The ignore index for the masked position.
- `hidden_dropout_prob` The probability of the hidden dropout.
- `attention_probs_dropout_prob` The probability of the attention dropout.
- `is_training` Training or inference.

## Result

| Task   | Metric   | Result    |
| ------ | -------- | --------- |
| Phase1 | MLM Loss | 1.623     |
|        | NSP Loss | 0.02729   |
|        | MLM Acc  | 0.668     |
|        | NSP Acc  | 0.9893    |
|        | tput     | 9200      |
| Phase2 | MLM Loss | 1.527     |
|        | NSP Loss | 0.01955   |
|        | MLM Acc  | 0.6826    |
|        | NSP Acc  | 0.9927    |
|        | tput     | 2700      |
| SQuAD  | EM       | 80.48249  |
|        | F1       | 87.556685 |

## Licensing

The code presented here is licensed under the Apache License Version 2.0, see the LICENSE file in this directory.

This directory includes derived work from the following:

PaddlePaddle/PaddleNLP, https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/bert/modeling.py

PaddlePaddle/PaddleNLP, https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/language_model/bert/static/run_pretrain.py

Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

```
http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
