# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import argparse
import distutils.util
import logging
import os
import pickle
import random
import time
from functools import partial

import numpy as np
import paddle
import paddle.fluid
import paddle.fluid.compiler as compiler
import paddle.io
import paddle.nn
import paddle.optimizer
from paddle.io import DataLoader
from paddlenlp.data import Dict, Stack
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertTokenizer, LinearDecayWithWarmup
from paddlenlp.utils.tools import TimeCostAverage
from run_pretrain import parse_args, reset_program_state_dict, set_seed

from modeling import BertForQuestionAnswering, BertModel

try:
    import wandb
    wandb.init(
        project="squad_train",
        settings=wandb.Settings(console='off'),
        name='paddle_squad_finetune')
except ImportError:
    wandb = None

MODEL_CLASSES = {"bert": (BertForQuestionAnswering, BertTokenizer)}


def create_squad_data_holder(args):
    if args.device == "ipu":
        bs = args.micro_batch_size
    else:
        bs = args.batch_size
    input_ids = paddle.static.data(
        name="input_ids", shape=[bs, args.seq_len], dtype="int64")
    segment_ids = paddle.static.data(
        name="segment_ids", shape=[bs, args.seq_len], dtype="int64")
    position_ids = paddle.static.data(
        name="position_ids", shape=[bs, args.seq_len], dtype="int64")
    input_mask = paddle.static.data(
        name="input_mask", shape=[bs, 1, 1, args.seq_len], dtype="float32")
    start_labels = paddle.static.data(
        name="start_labels", shape=[bs], dtype="int64")
    end_labels = paddle.static.data(
        name="end_labels", shape=[bs], dtype="int64")

    return [
        input_ids, segment_ids, position_ids, input_mask, start_labels,
        end_labels
    ]


def prepare_train_features(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    #NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=args.doc_stride,
        max_seq_len=args.max_seq_length,
        pad_to_max_seq_len=True,
        return_position_ids=True,
        return_token_type_ids=True,
        return_attention_mask=True)

    # Let's label those examples!
    for i, tokenized_example in enumerate(tokenized_examples):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_example["input_ids"]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offsets = tokenized_example['offset_mapping']

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        answers = examples[sample_index]['answers']
        answer_starts = examples[sample_index]['answer_starts']

        # attention_mask to input_mask
        input_mask = (
            np.asarray(tokenized_examples[i]["attention_mask"]) - 1) * 1e3
        input_mask = np.expand_dims(input_mask, axis=(0, 1))
        if args.device == 'ipu' and args.ipu_enable_fp16:
            input_mask = input_mask.astype(np.float16)
        else:
            input_mask = input_mask.astype(np.float32)
        tokenized_examples[i]["input_mask"] = input_mask

        # If no answers are given, set the cls_index as answer.
        if len(answer_starts) == 0:
            tokenized_examples[i]["start_positions"] = cls_index
            tokenized_examples[i]["end_positions"] = cls_index
        else:
            # Start/end character index of the answer in the text.
            start_char = answer_starts[0]
            end_char = start_char + len(answers[0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # Minus one more to reach actual text
            token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[
                        token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples[i]["start_positions"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples[i]["end_positions"] = token_end_index + 1

    return tokenized_examples


class BertQACriterion(paddle.nn.Layer):
    def __init__(self, num_ipus=None):
        super(BertQACriterion, self).__init__()
        self.num_ipus = num_ipus

    def forward(self, start_logits, end_logits, start_labels, end_labels):
        ipu_index = self.num_ipus - 1
        ipu_stage = self.num_ipus - 1
        logging.info("Loss Layer - ipu_index:%d, ipu_stage:%d" %
                     (ipu_index, ipu_stage))
        with paddle.fluid.ipu_shard(ipu_index=ipu_index, ipu_stage=ipu_stage):
            with paddle.static.name_scope("SQUAD_LOSS"):
                loss0 = paddle.fluid.layers.softmax(start_logits)
                loss0 = paddle.fluid.layers.cross_entropy(loss0, start_labels)
                loss1 = paddle.fluid.layers.softmax(end_logits)
                loss1 = paddle.fluid.layers.cross_entropy(loss1, end_labels)
                loss_fin = paddle.add(loss0, loss1)
                loss_fin = paddle.mean(loss_fin)
                return loss_fin


class BertQAccuracy(paddle.nn.Layer):
    def __init__(self, num_ipus=None, use_fp16=False):
        super(BertQAccuracy, self).__init__()
        self.num_ipus = num_ipus
        self.use_fp16 = use_fp16

    def forward(self, start_logits, end_logits, start_labels, end_labels):
        ipu_index = self.num_ipus - 1
        ipu_stage = self.num_ipus - 1
        logging.info("Acc Layer - ipu_index:%d, ipu_stage:%d" %
                     (ipu_index, ipu_stage))
        with paddle.fluid.ipu_shard(ipu_index=ipu_index, ipu_stage=ipu_stage):
            with paddle.static.name_scope("SQUAD_ACC"):
                start_logits = paddle.fluid.layers.argmax(start_logits, axis=1)
                end_logits = paddle.fluid.layers.argmax(end_logits, axis=1)
                start_equal = paddle.fluid.layers.equal(start_logits,
                                                        start_labels)
                end_equal = paddle.fluid.layers.equal(end_logits, end_labels)
                dtype = 'float16' if self.use_fp16 else 'float32'
                start_equal = paddle.fluid.layers.cast(start_equal, dtype)
                end_equal = paddle.fluid.layers.cast(end_equal, dtype)
                acc0 = paddle.mean(start_equal)
                acc1 = paddle.mean(end_equal)
                return acc0, acc1


def do_train(args):
    paddle.enable_static()
    place = paddle.set_device(args.device)

    # Create the random seed for the worker
    set_seed(args.seed)

    # Define the input data in the static mode
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    data_holders = create_squad_data_holder(args)
    [
        input_ids, segment_ids, position_ids, input_mask, start_labels,
        end_labels
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

    model = BertForQuestionAnswering(BertModel(**config), args.num_ipus)
    criterion = BertQACriterion(args.num_ipus)
    acc = BertQAccuracy(args.num_ipus, args.ipu_enable_fp16)

    start_logits, end_logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=segment_ids,
        input_mask=input_mask)

    # loss
    loss = criterion(start_logits, end_logits, start_labels, end_labels)
    # acc
    acc0, acc1 = acc(start_logits, end_logits, start_labels, end_labels)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    args.train_file = f'{cur_dir}/data/squad/train-v1.1.json'
    args.max_seq_length = args.seq_len
    args.doc_stride = 128

    cache_file = f'{args.train_file}.{args.device}.{args.max_seq_length}.cache'
    if os.path.exists(cache_file):
        logging.info(f"Loading Cache {cache_file}")
        with open(cache_file, "rb") as f:
            train_ds = pickle.load(f)
    else:
        train_ds = load_dataset(
            'squad', splits='train_v1', data_files=args.train_file)
        train_ds.map(partial(
            prepare_train_features, tokenizer=tokenizer, args=args),
                     batched=True,
                     num_workers=20)
        logging.info(f"Saving Cache {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(train_ds, f)

    if args.device == "ipu":
        # bs = args.micro_batch_size
        bs = args.micro_batch_size * args.grad_acc_factor * args.batches_per_step * args.num_replica
        args.batch_size = bs
        train_batch_sampler = paddle.io.BatchSampler(
            train_ds, batch_size=bs, shuffle=True, drop_last=True)
    else:
        # bs = args.batch_size
        train_batch_sampler = paddle.io.BatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    train_batchify_fn = lambda samples, fn=Dict({
        "input_ids": Stack(),
        "token_type_ids": Stack(),
        "position_ids": Stack(),
        "input_mask": Stack(),
        "start_positions": Stack(),
        "end_positions": Stack()
    }): fn(samples)

    data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=train_batchify_fn,
        return_list=True)

    global_step = 0
    total_samples = len(train_ds)
    max_steps = total_samples // args.batch_size * args.epochs
    print("Total samples: %d, Total batch_size: %d, Max_steps: %d" %
          (total_samples, args.batch_size, max_steps))

    # Define the dynamic learing_reate scheduler and optimizer
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, max_steps,
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
        # Delete mlm and nsp weights
        params.pop("linear_73.w_0")
        params.pop("linear_73.b_0")
        paddle.static.set_program_state(main_program, params)

    if args.use_amp:
        optimizer.amp_init(place)

    fetch_list = [loss.name, acc0.name, acc1.name]

    feed_list = [
        "input_ids", "segment_ids", "position_ids", "input_mask",
        "start_labels", "end_labels"
    ]

    if args.device == "ipu":
        ipu_strategy = compiler.get_ipu_strategy()
        ipu_strategy.is_training = args.is_training
        ipu_strategy.enable_manual_shard = True
        ipu_strategy.enable_pipelining = args.enable_pipelining
        ipu_strategy.batches_per_step = args.batches_per_step
        ipu_strategy.micro_batch_size = args.micro_batch_size
        ipu_strategy.save_init_onnx = args.save_init_onnx
        ipu_strategy.save_per_n_step = max_steps
        ipu_strategy.loss_scaling = args.scale_loss
        # Replica
        ipu_strategy.enableReplicatedGraphs = args.enable_replica and args.enable_pipelining
        ipu_strategy.replicatedGraphCount = args.num_replica if args.enable_replica and args.enable_pipelining else 1
        ipu_strategy.num_ipus = args.num_ipus * ipu_strategy.replicatedGraphCount
        # Gradacc
        ipu_strategy.enableGradientAccumulation = args.enable_grad_acc
        ipu_strategy.accumulationFactor = args.grad_acc_factor
        # Recomputation
        ipu_strategy.auto_recomputation = 3 if args.enable_recompute and args.enable_pipelining else 0
        # FP16
        ipu_strategy.enable_fp16 = args.ipu_enable_fp16
        # Half Partial
        ipu_strategy.enable_half_partial = args.enable_half_partial
        # Available_mem_proportion
        ipu_strategy.available_mem_proportion = args.available_mem_proportion

        # enable patterns
        ipu_strategy.enable_pattern("TiedGather")
        ipu_strategy.enable_pattern("TiedGatherAccumulate")

        ipu_strategy.enable_fully_connected_pass = False
        ipu_strategy.enable_engine_caching = True

        ipu_compiler = compiler.IpuCompiler(
            main_program, ipu_strategy=ipu_strategy)
        main_program = ipu_compiler.compile(feed_list, fetch_list)

    total_cost_avg = TimeCostAverage()
    read_cost_avg = TimeCostAverage()
    train_cost_avg = TimeCostAverage()

    batch_start = time.time()
    for epoch in range(args.epochs):
        for step, batch in enumerate(data_loader):
            global_step += 1
            read_cost = time.time() - batch_start
            read_cost_avg.record(read_cost)
            train_start = time.time()
            feed = {
                "input_ids": batch[0],
                "segment_ids": batch[1],
                "position_ids": batch[2],
                "input_mask": batch[3],
                "start_labels": batch[4],
                "end_labels": batch[5]
            }
            lr_scheduler.step()
            loss_return = exe.run(main_program,
                                  feed=feed,
                                  fetch_list=fetch_list)
            train_cost = time.time() - train_start
            train_cost_avg.record(train_cost)
            total_cost = time.time() - batch_start
            total_cost_avg.record(total_cost)

            tput = args.batch_size / total_cost_avg.get_average()
            if wandb is not None:
                wandb.log({
                    "loss": np.mean(loss_return[0]),
                    "accuracy": np.mean(loss_return[1:]),
                    "throughput": tput,
                    "global_step": global_step,
                    "defaultLearningRate": lr_scheduler()
                })

            if global_step % args.logging_steps == 0:
                print(
                    "epoch: %d, step: %d, read_cost: %f, total_cost: %f, all_loss: %f, acc0: %f, acc1: %f, tput: %f"
                    % (epoch, global_step, read_cost_avg.get_average(),
                       total_cost_avg.get_average(), np.mean(loss_return[0]),
                       np.mean(loss_return[1]), np.mean(loss_return[2]), tput))

            read_cost_avg.reset()
            total_cost_avg.reset()
            train_cost_avg.reset()

            batch_start = time.time()

    # save fin state
    if (args.device == "ipu"):
        paddle.static.save(main_program.org_program,
                           os.path.join(args.output_dir))
    else:
        paddle.static.save(main_program, os.path.join(args.output_dir))


if __name__ == "__main__":
    args = parse_args()
    if wandb:
        wandb_config = vars(args)
        wandb_config["global_batch_size"] = args.micro_batch_size * \
            args.num_replica * args.grad_acc_factor
        wandb.config.update(args)
    do_train(args)
