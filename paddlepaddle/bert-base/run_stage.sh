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

device=$1
# 1. stage=phase1 (pretrain_128)
# 2. stage=phase2 (pretrain_384) 
# 3. stage=SQuAD (finetune) 
# 4. stage=validation (validate SQuAD task)
stage=$2
in_model_name=$3
out_model_name=$4

if [ $# -eq 4 ]; then
    echo "Start from stage:${stage}..."
else
    echo "Invalid arguments."
    exit
fi

if [[ "$stage" == "phase1" ]]; then
    python3.7 run_pretrain.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --input_dir data/pretrain_128 \
        --output_dir ${out_model_name} \
        --seq_len 128 \
        --max_predictions_per_seq 20 \
        --learning_rate 0.006 \
        --weight_decay 1e-2 \
        --max_steps 7038 \
        --warmup_steps 2000 \
        --logging_steps 1 \
        --seed 42 \
        --device ${device} \
        --num_ipus 4 \
        --num_hidden_layers 12 \
        --micro_batch_size 16 \
        --ipu_enable_fp16 True \
        --scale_loss 512 \
        --save_init_onnx True \
        --save_per_n_step 7038 \
        --save_steps 7038 \
        --optimizer_type lamb \
        --enable_pipelining True \
        --batches_per_step 1 \
        --enable_replica True \
        --num_replica 4 \
        --enable_grad_acc True \
        --grad_acc_factor 1024 \
        --batch_size 65536 \
        --enable_recompute True \
        --enable_half_partial True \
        --available_mem_proportion 0.28 \
        --check_data True \
        --ignore_index 0 \
        --hidden_dropout_prob 0.1 \
        --attention_probs_dropout_prob 0.0
fi

if [[ "$stage" == "phase2" ]]; then
    in_model_params="${in_model_name}.pdparams"
    python3.7 run_pretrain.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --input_dir data/pretrain_384 \
        --output_dir ${out_model_name} \
        --seq_len 384 \
        --max_predictions_per_seq 56 \
        --learning_rate 0.002828427125 \
        --weight_decay 1e-2 \
        --max_steps 2137 \
        --warmup_steps 274 \
        --logging_steps 1 \
        --seed 42 \
        --device ${device} \
        --num_ipus 4 \
        --num_hidden_layers 12 \
        --micro_batch_size 4 \
        --ipu_enable_fp16 True \
        --scale_loss 128.0 \
        --save_init_onnx True \
        --save_per_n_step 2137 \
        --save_steps 2137 \
        --optimizer_type lamb \
        --enable_pipelining True \
        --batches_per_step 1 \
        --enable_replica True \
        --num_replica 4 \
        --enable_grad_acc True \
        --grad_acc_factor 1024 \
        --batch_size 16384 \
        --enable_recompute True \
        --enable_half_partial True \
        --available_mem_proportion 0.28 \
        --check_data True \
        --ignore_index 0 \
        --enable_load_params True \
        --load_params_path ${in_model_params} \
        --hidden_dropout_prob 0.1 \
        --attention_probs_dropout_prob 0.0
fi

if [[ "$stage" == "SQuAD" ]]; then
    in_model_params="${in_model_name}.pdparams"
    python3.7 \
        run_squad.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --input_dir data/squad \
        --output_dir ${out_model_name} \
        --seq_len 384 \
        --learning_rate 5.6e-05 \
        --weight_decay 1e-2 \
        --warmup_steps 52 \
        --logging_steps 1 \
        --seed 42 \
        --device ${device} \
        --num_ipus 2 \
        --num_hidden_layers 12 \
        --micro_batch_size 2 \
        --ipu_enable_fp16 True \
        --scale_loss 256 \
        --optimizer_type adam \
        --enable_pipelining True \
        --batches_per_step 4 \
        --enable_replica True \
        --num_replica 2 \
        --enable_grad_acc True \
        --grad_acc_factor 16 \
        --batch_size 256 \
        --enable_recompute True \
        --enable_half_partial True \
        --available_mem_proportion 0.40 \
        --check_data True \
        --layer_per_ipu 6 \
        --encoder_start_ipu 0 \
        --epochs 4 \
        --hidden_dropout_prob 0.1 \
        --attention_probs_dropout_prob 0.1 \
        --enable_load_params True \
        --load_params_path ${in_model_params}
fi

if [[ "$stage" == "validation" ]]; then
    in_model_params="${in_model_name}.pdparams"
    python3.7 \
        run_squad_infer.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --input_dir data/squad \
        --output_dir ${out_model_name} \
        --seq_len 384 \
        --learning_rate 5.6e-05 \
        --weight_decay 1e-2 \
        --warmup_steps 52 \
        --logging_steps 1 \
        --seed 42 \
        --device ${device} \
        --num_ipus 2 \
        --num_hidden_layers 12 \
        --micro_batch_size 1 \
        --ipu_enable_fp16 True \
        --scale_loss 256 \
        --optimizer_type adam \
        --enable_pipelining True \
        --batches_per_step 128 \
        --enable_replica True \
        --num_replica 4 \
        --enable_grad_acc False \
        --grad_acc_factor 1 \
        --batch_size 512 \
        --enable_recompute True \
        --enable_half_partial True \
        --available_mem_proportion 0.40 \
        --check_data True \
        --layer_per_ipu 6 \
        --encoder_start_ipu 0 \
        --hidden_dropout_prob 0.0 \
        --attention_probs_dropout_prob 0.0 \
        --enable_load_params True \
        --load_params_path ${in_model_params} \
        --is_training False
fi