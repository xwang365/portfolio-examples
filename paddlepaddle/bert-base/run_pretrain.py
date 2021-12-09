# Copyright (c) 2021 Graphcore Ltd.
#
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
# https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/language_model/bert/static/run_pretrain.py

import argparse
import os
import random
from re import A
import time
from functools import partial

import pickle
import numpy as np
import distutils.util

import paddle
import paddle.distributed.fleet as fleet
import paddle.fluid.compiler as compiler

from paddlenlp.utils.tools import TimeCostAverage
from paddlenlp.transformers import BertTokenizer
from dataset_ipu import create_data_holder, PretrainingTfRecordDataLoader
from paddlenlp.transformers import LinearDecayWithWarmup

from modeling import BertForPretraining, BertModel, BertPretrainingCriterion, BertPretrainingAccuracy

try:
    import wandb
    wandb.init(
        project="paddle_test",
        settings=wandb.Settings(console='off'),
        name='paddle_test')
except ImportError:
    wandb = None

MODEL_CLASSES = {"bert": (BertForPretraining, BertTokenizer)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input directory where the data will be read from.", )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seq_len", default=128, type=int, help="The sequence length")
    parser.add_argument(
        "--max_predictions_per_seq",
        default=20,
        type=int,
        help="The maximum total of masked tokens in input sequence")
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=10,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument(
        "--use_amp",
        type=distutils.util.strtobool,
        default=False,
        help="Enable mixed precision training.")
    parser.add_argument(
        "--enable_addto",
        type=distutils.util.strtobool,
        default=False,
        help="Whether to enable the addto strategy for gradient accumulation or not. This is only used for AMP training."
    )
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=1.0,
        help="The value of scale_loss for fp16.")
    parser.add_argument(
        "--use_pure_fp16",
        type=distutils.util.strtobool,
        default=False,
        help="Whether to use pure fp16 training.")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        help="Device for selecting for the training.")
    parser.add_argument(
        "--gradient_merge_steps",
        type=int,
        default=1,
        help="Number of merge steps before gradient update."
        "global_batch_size = gradient_merge_steps * batch_size.")
    parser.add_argument(
        "--ipu_enable_fp16",
        type=distutils.util.strtobool,
        default=False,
        help="ipu enable fp16 or not.")
    parser.add_argument(
        "--enable_pipelining",
        type=distutils.util.strtobool,
        default=False,
        help="enable pipelining or not.")
    parser.add_argument(
        "--enable_replica",
        type=distutils.util.strtobool,
        default=False,
        help="enable replicat or not.")
    parser.add_argument(
        "--num_replica", type=int, default=1, help="number of replica")
    parser.add_argument(
        "--num_ipus",
        type=int,
        default=2,
        help="Number ipu need to train bert-base")
    parser.add_argument(
        "--layer_per_ipu",
        type=int,
        default=4,
        help="Number encoder layer per ipu")
    parser.add_argument(
        "--encoder_start_ipu",
        type=int,
        default=1,
        help="Encoder start ipu index")
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help="Override config file if not None")
    parser.add_argument(
        "--enable_grad_acc",
        type=distutils.util.strtobool,
        default=False,
        help="enable gradient accumulation")
    parser.add_argument(
        "--grad_acc_factor",
        type=int,
        default=1,
        help="factor of gradient accumulation")
    parser.add_argument(
        "--micro_batch_size", type=int, default=1, help="micro batch size")
    parser.add_argument(
        "--batches_per_step", type=int, default=1, help="batches per step")
    parser.add_argument(
        "--enable_recompute",
        type=distutils.util.strtobool,
        default=False,
        help="enable recompute or not")
    parser.add_argument(
        "--save_init_onnx",
        type=distutils.util.strtobool,
        default=False,
        help="save init onnx model or not")
    parser.add_argument(
        "--enable_half_partial",
        type=distutils.util.strtobool,
        default=False,
        help="enable fp16 partial or not")
    parser.add_argument(
        "--available_mem_proportion",
        type=float,
        default=0.0,
        help="set the available memory proportion for matmul/conv")
    parser.add_argument(
        "--save_per_n_step",
        type=int,
        default=1,
        help="save weights D2H per n steps")
    parser.add_argument(
        "--optimizer_type", type=str, default='sgd', help="type of optimizer")
    parser.add_argument(
        "--is_training",
        type=distutils.util.strtobool,
        default=True,
        help="Training or not")
    parser.add_argument(
        "--check_data",
        type=distutils.util.strtobool,
        default=False,
        help="Check dataset error")
    parser.add_argument(
        "--ignore_index", type=int, default=-1, help="ignore mlm index")
    parser.add_argument(
        "--enable_load_params",
        type=distutils.util.strtobool,
        default=False,
        help="load params or not")
    parser.add_argument("--load_params_path", type=str, help="load params path")
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help="hidden_dropout_prob", )
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.0,
        help="attention_probs_dropout_prob", )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="the iteration of the whole dataset", )
    args = parser.parse_args()
    return args


def select_dataset_file_for_each_worker(files, f_start_id, worker_num,
                                        worker_index):
    """  
    Spliting the train file according to the worker index.
    """
    num_files = len(files)
    if worker_num > num_files:
        remainder = worker_num % num_files
        data_file = files[(
            f_start_id * worker_num + worker_index + remainder * f_start_id) %
                          num_files]
    else:
        data_file = files[(f_start_id * worker_num + worker_index) % num_files]
    return data_file


def reset_program_state_dict(model, state_dict):
    """
    Initialize the parameter from the bert config, and set the parameter by 
    reseting the state dict."
    """
    scale = model.initializer_range if hasattr(model, "initializer_range")\
        else model.bert.config["initializer_range"]

    new_state_dict = dict()
    for n, p in state_dict.items():
        if "layer_norm" not in p.name:
            dtype_str = "float32"
            if str(p.dtype) == "VarType.FP64":
                dtype_str = "float64"
            new_state_dict[p.name] = np.random.normal(
                loc=0.0, scale=scale, size=p.shape).astype(dtype_str)
    return new_state_dict


def create_strategy(args):
    """
    Create build strategy and exec strategy.
    """
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    build_strategy.enable_addto = args.enable_addto

    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = 10000
    return build_strategy, exec_strategy


def dist_optimizer(args, optimizer):
    """
    Create a distributed optimizer based on a normal optimizer
    """
    build_strategy, exec_strategy = create_strategy(args)

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.build_strategy = build_strategy

    dist_strategy.fuse_grad_size_in_MB = 16
    if args.use_amp:
        dist_strategy.amp = True

        custom_black_list = ['lookup_table',
                             'lookup_table_v2'] if args.use_pure_fp16 else None
        dist_strategy.amp_configs = {
            'custom_white_list': ['softmax', 'layer_norm', 'gelu'],
            'init_loss_scaling': args.scale_loss,
            'custom_black_list': custom_black_list,
            'use_pure_fp16': args.use_pure_fp16
        }
    if args.gradient_merge_steps > 1:
        dist_strategy.gradient_merge = True
        dist_strategy.gradient_merge_configs = {
            'k_steps': args.gradient_merge_steps
        }

    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    return optimizer


def set_seed(seed):
    """
    Use the same data seed(for data shuffle) for all procs to guarantee data
    consistency after sharding.
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class WorkerInitObj(object):
    "Construct the object with different seed, and the Dataloader will generate the data "
    "with different seed in each worker."

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def do_train(args):
    # Initialize the paddle and paddle fleet execute enviroment
    paddle.enable_static()
    place = paddle.set_device(args.device)
    fleet.init(is_collective=True)

    # Create the random seed for the worker
    set_seed(args.seed)

    # Define the input data in the static mode
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    data_holders = create_data_holder(args)

    [
        input_ids, segment_ids, input_mask, masked_lm_positions,
        masked_lm_labels, next_sentence_labels, masked_lm_scale, position_ids
    ] = data_holders

    # Define the model structure in static mode
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    config = model_class.pretrained_init_configuration[args.model_name_or_path]
    if config["vocab_size"] % 8 != 0:
        config["vocab_size"] += 8 - (config["vocab_size"] % 8)
    config["num_ipus"] = args.num_ipus
    config["layer_per_ipu"] = args.layer_per_ipu
    config["encoder_start_ipu"] = args.encoder_start_ipu
    config["num_hidden_layers"] = args.num_hidden_layers
    config["hidden_dropout_prob"] = args.hidden_dropout_prob
    config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob

    model = BertForPretraining(BertModel(**config))
    if args.device == "ipu":
        if args.ipu_enable_fp16:
            acc = BertPretrainingAccuracy(
                masked_lm_labels,
                next_sentence_labels,
                True,
                args.num_ipus,
                args.micro_batch_size,
                ignore_index=args.ignore_index)
        else:
            acc = BertPretrainingAccuracy(
                masked_lm_labels,
                next_sentence_labels,
                False,
                args.num_ipus,
                args.micro_batch_size,
                ignore_index=args.ignore_index)
    else:
        acc = BertPretrainingAccuracy(
            masked_lm_labels,
            next_sentence_labels,
            False,
            args.num_ipus,
            args.batch_size,
            ignore_index=args.ignore_index)
    criterion = BertPretrainingCriterion(
        model.bert.config["vocab_size"],
        args.num_ipus,
        ignore_index=args.ignore_index)
    prediction_scores, seq_relationship_score = model(
        input_ids=input_ids,
        token_type_ids=segment_ids,
        attention_mask=input_mask,
        position_ids=position_ids,
        masked_positions=masked_lm_positions)
    mlm_acc, nsp_acc = acc(prediction_scores=prediction_scores,
                           seq_relationship_score=seq_relationship_score)
    loss, masked_lm_loss, next_sentence_loss = criterion(
        prediction_scores, seq_relationship_score, masked_lm_labels,
        next_sentence_labels, masked_lm_scale)

    # Define the dynamic learing_reate scheduler and optimizer
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, args.max_steps,
                                         args.warmup_steps)
    if args.is_training:
        optimizer = paddle.optimizer.SGD(learning_rate=lr_scheduler,
                                         weight_decay=args.weight_decay)
        if args.optimizer_type == 'adam':
            optimizer = paddle.optimizer.Adam(
                learning_rate=lr_scheduler, weight_decay=args.weight_decay)
        if args.optimizer_type == 'lamb':
            optimizer = paddle.optimizer.Lamb(
                learning_rate=lr_scheduler, lamb_weight_decay=args.weight_decay)
        optimizer.minimize(loss)

    # Define the Executor for running the static model
    exe = paddle.static.Executor(place)
    exe.run(startup_program)
    state_dict = model.state_dict()
    # Use the state dict to update the parameter
    reset_state_dict = reset_program_state_dict(model, state_dict)
    paddle.static.set_program_state(main_program, reset_state_dict)

    if args.enable_load_params:
        with open(args.load_params_path, 'rb') as file:
            params = pickle.load(file)
        paddle.static.set_program_state(main_program, params)

    if args.use_amp:
        optimizer.amp_init(place)

    if args.device == "ipu":
        feed_list = [
            "input_ids", "segment_ids", "input_mask", "masked_lm_positions",
            "masked_lm_labels", "next_sentence_labels", "masked_lm_scale",
            "position_ids"
        ]
        fetch_list = [
            loss.name, masked_lm_loss.name, next_sentence_loss.name,
            mlm_acc.name, nsp_acc.name
        ]

        ipu_strategy = compiler.get_ipu_strategy()
        ipu_strategy.is_training = args.is_training
        ipu_strategy.enable_manual_shard = True
        ipu_strategy.enable_pipelining = args.enable_pipelining
        ipu_strategy.batches_per_step = args.batches_per_step
        ipu_strategy.micro_batch_size = args.micro_batch_size
        ipu_strategy.save_init_onnx = args.save_init_onnx
        ipu_strategy.save_per_n_step = args.save_per_n_step
        ipu_strategy.loss_scaling = args.scale_loss
        # Replica
        ipu_strategy.enableReplicatedGraphs = args.enable_replica and args.enable_pipelining
        ipu_strategy.replicatedGraphCount = args.num_replica if args.enable_replica and args.enable_pipelining else 1
        ipu_strategy.num_ipus = args.num_ipus * ipu_strategy.replicatedGraphCount
        # Gradacc
        ipu_strategy.enableGradientAccumulation = args.enable_grad_acc and args.enable_pipelining
        ipu_strategy.accumulationFactor = args.grad_acc_factor
        # Recomputation
        ipu_strategy.auto_recomputation = 3 if args.enable_recompute and args.enable_pipelining else 0
        # FP16
        ipu_strategy.enable_fp16 = args.ipu_enable_fp16
        # Half Partial
        ipu_strategy.enable_half_partial = args.enable_half_partial
        # Available_mem_proportion
        ipu_strategy.available_mem_proportion = args.available_mem_proportion
        # enable_stochastic_rounding
        ipu_strategy.enable_stochastic_rounding = args.is_training

        # enable patterns
        ipu_strategy.enable_pattern("TiedGather")
        ipu_strategy.enable_pattern("TiedGatherAccumulate")

        ipu_compiler = compiler.IpuCompiler(
            main_program, ipu_strategy=ipu_strategy)
        main_program = ipu_compiler.compile(feed_list, fetch_list)

    files = [
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if os.path.isfile(os.path.join(args.input_dir, f)) and "tfrecord" in f
    ]
    files.sort()
    if args.device == "ipu" and args.ipu_enable_fp16:
        data_loader = PretrainingTfRecordDataLoader(
            input_files=files,
            max_seq_length=args.seq_len,
            max_mask_tokens=args.max_predictions_per_seq,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            enable_fp16=True,
            enable_ipu=True,
            enable_check_data=args.check_data,
            ignore_index=args.ignore_index,
            shuffle=True)
    elif args.device == "ipu":
        data_loader = PretrainingTfRecordDataLoader(
            input_files=files,
            max_seq_length=args.seq_len,
            max_mask_tokens=args.max_predictions_per_seq,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            enable_fp16=False,
            enable_ipu=True,
            enable_check_data=args.check_data,
            ignore_index=args.ignore_index,
            shuffle=True)
    else:
        data_loader = PretrainingTfRecordDataLoader(
            input_files=files,
            batch_size=args.batch_size,
            micro_batch_size=args.batch_size,
            enable_check_data=args.check_data,
            ignore_index=args.ignore_index,
            shuffle=True)

    global_step = 0
    total_samples = data_loader.total_samples
    print("Total samples: %d, Total batch_size: %d, Max_steps: %d" %
          (total_samples, args.batch_size, args.max_steps))

    total_cost_avg = TimeCostAverage()
    read_cost_avg = TimeCostAverage()
    train_cost_avg = TimeCostAverage()

    batch_start = time.time()
    for batch in data_loader:
        global_step += 1
        epoch = global_step * args.batch_size / total_samples
        read_cost = time.time() - batch_start
        read_cost_avg.record(read_cost)
        train_start = time.time()
        feed = {
            "input_ids": batch[0],
            "segment_ids": batch[1],
            "input_mask": batch[2],
            "masked_lm_positions": batch[3],
            "masked_lm_labels": batch[4],
            "next_sentence_labels": batch[5],
            "masked_lm_scale": batch[6],
            "position_ids": batch[7]
        }

        lr_scheduler.step()
        loss_return = exe.run(main_program,
                              feed=feed,
                              fetch_list=[
                                  loss, masked_lm_loss, next_sentence_loss,
                                  mlm_acc, nsp_acc
                              ])

        train_cost = time.time() - train_start
        train_cost_avg.record(train_cost)
        total_cost = time.time() - batch_start
        total_cost_avg.record(total_cost)

        tput = args.batch_size / total_cost_avg.get_average()
        if wandb is not None:
            wandb.log({
                "loss/MLM": np.mean(loss_return[1]),
                "loss/NSP": np.mean(loss_return[2]),
                "accuracy/MLM": np.mean(loss_return[3]),
                "accuracy/NSP": np.mean(loss_return[4]),
                "latency/read": read_cost_avg.get_average(),
                "latency/train": train_cost_avg.get_average(),
                "latency/e2e": total_cost_avg.get_average(),
                "throughput": tput,
                "defaultLearningRate": lr_scheduler(),
                "global_step": global_step,
            })

        if global_step % args.logging_steps == 0:
            print(
                "epoch: %d, step: %d, loss: %f, "
                "total_cost: %.5f sec, read_cost: %.5f sec, train_cost: %.5f sec, throughput: %.5f seq/s"
                % (epoch, global_step, np.mean(loss_return[0]),
                   total_cost_avg.get_average(), read_cost_avg.get_average(),
                   train_cost_avg.get_average(), tput))

        read_cost_avg.reset()
        total_cost_avg.reset()
        train_cost_avg.reset()

        if global_step % args.save_steps == 0:
            if (args.device == "ipu"):
                paddle.static.save(main_program.org_program,
                                   os.path.join(args.output_dir))
            else:
                paddle.static.save(main_program, os.path.join(args.output_dir))

        if global_step >= args.max_steps:
            data_loader.release()
            del data_loader
            return
        batch_start = time.time()


if __name__ == "__main__":
    args = parse_args()
    if wandb:
        wandb_config = vars(args)
        wandb_config["global_batch_size"] = args.micro_batch_size * \
            args.num_replica * args.grad_acc_factor
        wandb.config.update(args)
    do_train(args)
