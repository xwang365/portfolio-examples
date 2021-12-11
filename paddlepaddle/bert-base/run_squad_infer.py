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

import json
import logging
import os
import pickle
from functools import partial

import numpy as np
import paddle
import paddle.fluid
import paddle.fluid.compiler as compiler
import paddle.io
import paddle.nn
import paddle.optimizer
import paddle.static
from paddle.io import DataLoader
from paddlenlp.data import Dict, Stack
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics.squad import compute_prediction, squad_evaluate
from paddlenlp.transformers import BertTokenizer
from modeling import BertForQuestionAnswering, BertModel

from run_pretrain import parse_args, reset_program_state_dict, set_seed

MODEL_CLASSES = {"bert": (BertForQuestionAnswering, BertTokenizer)}


def create_squad_data_holder_infer(args):
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
    return [input_ids, segment_ids, position_ids, input_mask]


def prepare_validation_features(examples, tokenizer, args):
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

    # For validation, there is no need to compute start and end positions
    for i, tokenized_example in enumerate(tokenized_examples):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        tokenized_examples[i]["example_id"] = examples[sample_index]['id']

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples[i]["offset_mapping"] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_example["offset_mapping"])
        ]

        # attention_mask to input_mask
        input_mask = (
            np.asarray(tokenized_examples[i]["attention_mask"]) - 1) * 1e3
        input_mask = np.expand_dims(input_mask, axis=(0, 1))
        if args.device == 'ipu' and args.ipu_enable_fp16:
            input_mask = input_mask.astype(np.float16)
        else:
            input_mask = input_mask.astype(np.float32)
        tokenized_examples[i]["input_mask"] = input_mask

    return tokenized_examples


def do_infer(args):
    paddle.enable_static()
    place = paddle.set_device(args.device)

    # Create the random seed for the worker
    set_seed(args.seed)

    # Define the input data in the static mode
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    data_holders = create_squad_data_holder_infer(args)
    [input_ids, segment_ids, position_ids, input_mask] = data_holders

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
    start_logits, end_logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=segment_ids,
        input_mask=input_mask)

    args.max_seq_length = args.seq_len
    args.doc_stride = 128
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    args.dev_file = f'{cur_dir}/data/squad/dev-v1.1.json'

    cache_file = f'{args.dev_file}.{args.device}.{args.max_seq_length}.cache'
    if os.path.exists(cache_file):
        logging.info(f"Loading Cache {cache_file}")
        with open(cache_file, "rb") as f:
            train_ds = pickle.load(f)
    else:
        train_ds = load_dataset(
            'squad', splits='dev_v1', data_files=args.dev_file)
        train_ds.map(partial(
            prepare_validation_features, tokenizer=tokenizer, args=args),
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
            train_ds, batch_size=bs, shuffle=False, drop_last=False)
    else:
        # bs = args.batch_size
        train_batch_sampler = paddle.io.BatchSampler(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False)

    train_batchify_fn = lambda samples, fn=Dict({
        "input_ids": Stack(),
        "token_type_ids": Stack(),
        "position_ids": Stack(),
        "input_mask": Stack(), }): fn(samples)

    data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=train_batchify_fn,
        return_list=True)

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
        # params.pop("linear_73.w_0")
        # params.pop("linear_73.b_0")
        paddle.static.set_program_state(main_program, params)

    fetch_list = [start_logits.name, end_logits.name]

    feed_list = [
        "input_ids",
        "segment_ids",
        "position_ids",
        "input_mask",
    ]

    if args.device == "ipu":
        ipu_strategy = compiler.get_ipu_strategy()
        ipu_strategy.is_training = args.is_training
        ipu_strategy.enable_manual_shard = True
        ipu_strategy.enable_pipelining = args.enable_pipelining
        ipu_strategy.batches_per_step = args.batches_per_step
        ipu_strategy.micro_batch_size = args.micro_batch_size
        ipu_strategy.save_init_onnx = args.save_init_onnx
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

    all_start_logits = []
    all_end_logits = []
    total_samples = len(train_ds)
    max_steps = total_samples // args.batch_size + 1
    print("Total samples: %d, Total batch_size: %d, Max_steps: %d" %
          (total_samples, args.batch_size, max_steps))
    for step, batch in enumerate(data_loader):
        print(f'step: {step}')
        # input_ids, segment_ids, position_ids, input_mask = batch
        real_len = np.array(batch[0]).shape[0]
        if (real_len < args.batch_size):
            batch = [np.asarray(x) for x in batch]
            pad0 = np.zeros([args.batch_size - real_len, args.seq_len]).astype(
                batch[0].dtype)
            batch[0] = np.vstack((batch[0], pad0))
            batch[1] = np.vstack((batch[1], pad0))
            batch[2] = np.vstack((batch[2], pad0))
            pad1 = np.zeros(
                [args.batch_size - real_len, 1, 1, args.seq_len]) - 1e3
            pad1 = pad1.astype(batch[3].dtype)
            batch[3] = np.vstack((batch[3], pad1))
            print(f"batch[0].shape: {batch[0].shape}")
            print(f"batch[3].shape: {batch[3].shape}")

        feed = {
            "input_ids": batch[0],
            "segment_ids": batch[1],
            "position_ids": batch[2],
            "input_mask": batch[3],
        }
        loss_return = exe.run(main_program, feed=feed, fetch_list=fetch_list)
        start_logits, end_logits = loss_return

        start_logits = start_logits.reshape([-1, args.seq_len])
        end_logits = end_logits.reshape([-1, args.seq_len])
        for idx in range(real_len):
            all_start_logits.append(start_logits[idx])
            all_end_logits.append(end_logits[idx])

    # evaluate results
    args.version_2_with_negative = False
    args.n_best_size = 20
    args.max_answer_length = 30
    args.null_score_diff_threshold = 0.0
    print(f'len(train_ds.data): {len(train_ds.data)}')
    print(f'len(train_ds.new_data): {len(train_ds.new_data)}')
    print(f'len(all_start_logits): {len(all_start_logits)}')

    all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
        data_loader.dataset.data, data_loader.dataset.new_data,
        (all_start_logits, all_end_logits), args.version_2_with_negative,
        args.n_best_size, args.max_answer_length,
        args.null_score_diff_threshold)

    args.output_json = 'squad_prediction.json'
    with open(args.output_json, "w", encoding='utf-8') as writer:
        writer.write(
            json.dumps(
                all_predictions, ensure_ascii=False, indent=4) + "\n")

    squad_evaluate(
        examples=data_loader.dataset.data,
        preds=all_predictions,
        na_probs=scores_diff_json)


if __name__ == "__main__":
    args = parse_args()
    if not args.is_training:
        do_infer(args)
