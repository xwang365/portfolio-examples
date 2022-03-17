# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import os
import sys
import time
import logging
import argparse
from pathlib import Path

import wandb
import torch
import torch.nn as nn
import torch.onnx
import poptorch
import popdist
import popdist.poptorch
import horovod.torch as hvd

from poptorch import DataLoader
from poptorch.enums import DataLoaderMode
from torch.nn import CrossEntropyLoss
from transformers import GPT2Config, GPT2LMHeadModel

from ipu_options import get_options
from model.optimized_gpt2_attn import OptimizedGPT2Attention
from utils import (SerializedLinear, _get_layer_ipu, _WorkerInit,
                   calculate_acc, collate_fn, get_generated_datum,
                   get_lr_scheduler, get_optimizer, load_dataset,
                   outline_attribute, recomputation_checkpoint, str_to_bool,
                   sync_metrics)

MODEL_CONFIG = {'gpt2': 'config/config.json', 'gpt2-medium': 'config/config_medium.json',
                'gpt2-large': 'config/config_large.json', 'gpt2-xl': 'config/config_xl.json'}

logging.basicConfig(level=logging.INFO, format="%(message)s")


def logger(msg):
    if not popdist.isPopdistEnvSet() or popdist.getInstanceIndex() == 0:
        logging.info(msg)


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model', type=str, default='gpt2', choices=('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'),
                        help='model to train')
    parser.add_argument('--pretrained-checkpoint', default='', type=str, required=False, help='pretrained model path')
    parser.add_argument("--enable-half-partials", type=str_to_bool, nargs="?", const=True, default=False,
                        help="enable half partials or not")
    parser.add_argument('--save-model-path', default=None, type=str, required=False,
                        help='model path to save')
    parser.add_argument('--executable-cache-dir', default=None, type=str, required=False,
                        help='executable cache dir')
    parser.add_argument('--training-steps', default=10000, type=int, required=False, help='training steps')
    parser.add_argument("--compile-only", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Create an offline IPU target that can only be used for offline compilation.")
    parser.add_argument("--custom-ops", type=str_to_bool, nargs="?", const=True, default=True, help="Enable custom ops")

    # dataset
    parser.add_argument('--train-path', default='data/train.pkl', type=str, required=False, help='dataset path')
    parser.add_argument('--data-prefix', type=str, required=False, help='file prefix while using dynamic dataset')
    parser.add_argument('--tfrecord-path', nargs="+", help='tfrecord dataset path')
    parser.add_argument('--max-len', default=128, type=int, required=False, help='max length of input sequence')
    parser.add_argument("--enable-sequence-serialized", type=str_to_bool, nargs="?", const=True, default=False,
                        help="enable-sequence-serialize")
    parser.add_argument('--serialized-seq-len', default=128, type=int, required=False, help='serialized-seq-len')
    parser.add_argument('--stride', default=128, type=int, required=False, help='stride window size to sample dataset')
    parser.add_argument('--val-num', type=int, default=0, help='validate dataset length')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--num-workers', type=int, default=4, help="workers for dataloader")
    parser.add_argument("--async-dataloader", type=str_to_bool, nargs="?", const=True, default=False,
                        help="async dataloader")

    # train
    parser.add_argument('--epochs', default=1, type=int, required=False, help='epochs for training')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default = 1)')
    parser.add_argument('--optimizer', default='AdamW', type=str, required=False, help='optimizer')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False, help='weight_decay')
    parser.add_argument('--learning-rate', default=0.00001, type=float, required=False, help='learning_rate')
    parser.add_argument('--loss-scaling', default=50000.0, type=float, required=False, help='loss_scaling')
    parser.add_argument('--lr-warmup', default=0.1, type=float, required=False, help='fraction of train steps(or --lr-decay-steps) to linearly warmup learning rate over')
    parser.add_argument('--lr-warmup-steps', default=None, type=int, required=False, help='number of steps to linearly warmup learning rate over.')
    parser.add_argument('--lr-decay-steps', default=None, type=int, required=False, help='number of steps to decay learning rate over, if None defaults to train steps')
    parser.add_argument('--lr-schedule', default='constant', type=str, choices=('linear', 'constant', 'cosine'),
                        required=False, help='lr_schedule')
    parser.add_argument('--log-steps', default=1, type=int, required=False, help='log_steps')
    parser.add_argument('--save-per-epochs', default=1, type=int, required=False, help='save-per-epochs')
    parser.add_argument('--save-per-steps', default=None, type=int, required=False, help='save-per-steps')
    parser.add_argument('--gradient-accumulation', default=10, type=int, required=False, help='gradient_accumulation')
    parser.add_argument("--use-wandb", type=str_to_bool, nargs="?", const=True, default=False, help="use wandb or not")

    # mapping
    parser.add_argument('--layers-per-ipu', type=int, default=3, nargs="+",
                        help='Number of decoder layers per pipeline stage, after the 0th stage (default = 3). Can be a single number, for an equal number decoder layers per IPU.\
                                Or it can be a list of numbers, specifying number of decoder layers for each individual IPU.')
    parser.add_argument('--batches-per-step', default=4, type=int, required=False, help='batches_per_step')
    parser.add_argument('--replication-factor', default=1, type=int, required=False, help='replication_factor')
    parser.add_argument('--ipus-per-replica', default=4, type=int, required=False, help='ipus_per_replica')
    parser.add_argument("--matmul-proportion", type=float, nargs="+",
                        help="Relative IPU memory proportion size allocated for matmul")
    parser.add_argument("--recompute-checkpoint-every-layer", type=str_to_bool, nargs="?", const=True, default=False,
                        help="This controls how recomputation is handled in pipelining. "
                             "If True the output of each encoder layer will be stashed keeping the max liveness "
                             "of activations to be at most one layer. "
                             "However, the stash size scales with the number of pipeline stages so this may not always be beneficial. "
                             "The added stash + code could be greater than the reduction in temporary memory.", )
    parser.add_argument("--recompute-checkpoint-layers", type=int, nargs="+", default=None,
                        help="Decoder layers that will be checkpointed.")
    parser.add_argument("--resume-training-from-checkpoint", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Restore both the model checkpoint and training state in order to resume a training run.")
    parser.add_argument("--embedding-serialization-factor", default=1, type=int,
                        help="Matmul serialization factor the embedding layers")
    parser.add_argument("--remap-logit", type=str_to_bool, nargs="?", const=True, default=False,
                        help="remap logits or not by custom op")
    parser.add_argument("--optimizer-state-offchip", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Set the tensor storage location for optimizer state to be offchip.")
    parser.add_argument("--replicated-tensor-sharding", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enable replicated tensor sharding of optimizer state")

    args = parser.parse_args()
    # Initialise PopDist
    if popdist.isPopdistEnvSet():
        hvd.init()
        args.use_popdist = True
        if popdist.getNumTotalReplicas() != args.replication_factor:
            print(f"The number of replicas is overridden by PopRun. "
                  f"The new value is {popdist.getNumTotalReplicas()}.")
        args.replication_factor = int(popdist.getNumLocalReplicas())
        args.popdist_rank = popdist.getInstanceIndex()
        args.popdist_size = popdist.getNumInstances()

        hvd.broadcast(torch.Tensor([args.seed]), root_rank=0)
    else:
        args.use_popdist = False

    return args


class GTP2Wrapper(nn.Module):
    def __init__(self, args, model_config):
        super().__init__()

        if args.pretrained_checkpoint:  # load pretrained model checkpoint
            self.model = GPT2LMHeadModel.from_pretrained(args.pretrained_checkpoint)
        else:  # init model
            self.config = model_config
            self.model = GPT2LMHeadModel(config=self.config)

        for layer in self.model.transformer.h:
            gpt2_attn = OptimizedGPT2Attention(self.model.config, layer_idx=layer.attn.layer_idx)
            gpt2_attn.load_state_dict(layer.attn.state_dict())
            layer.attn = gpt2_attn

        if args.embedding_serialization_factor > 1:
            serialized_lmhead = SerializedLinear(self.model.config.n_embd, self.model.config.vocab_size,
                                                 args.embedding_serialization_factor,
                                                 bias=False,
                                                 mode=poptorch.MatMulSerializationMode.OutputChannels)
            serialized_lmhead.load_state_dict(self.model.lm_head.state_dict())
            self.model.lm_head = serialized_lmhead
            self.model.tie_weights()

        logger("-------------------- Device Allocation --------------------")
        logger("Embedding  --> IPU 0")
        self.model.transformer.wte = poptorch.BeginBlock(self.model.transformer.wte, "wte", ipu_id=0)
        self.model.transformer.wpe = poptorch.BeginBlock(self.model.transformer.wpe, "wpe", ipu_id=1)
        outline_attribute(self.model.transformer.ln_f, "LayerNorm")

        layer_ipu = _get_layer_ipu(args.layers_per_ipu)
        for index, layer in enumerate(self.model.transformer.h):
            ipu = layer_ipu[index]
            if args.recompute_checkpoint_every_layer:
                if (args.recompute_checkpoint_layers is None) or (index in args.recompute_checkpoint_layers):
                    recomputation_checkpoint(layer)
            self.model.transformer.h[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger(f"Layer {index:<2} --> IPU {ipu}")

        logger(f'LM_head --> IPU 0')
        self.model.lm_head = poptorch.BeginBlock(self.model.lm_head, ipu_id=0)

    def forward(self, input_ids, labels):
        transformer_outputs = self.model.transformer(input_ids=input_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.model.lm_head(hidden_states)
        if not args.enable_sequence_serialized:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss = poptorch.identity_loss(loss, reduction="none")
            acc = calculate_acc(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            return loss, acc
        else:
            lm_logits = lm_logits.view(-1, lm_logits.size(-1))
            labels = labels.view(-1)
            loss_fct = CrossEntropyLoss(reduction="sum")
            loss, acc = None, None
            loss_weights = torch.sum((labels > -1).to(torch.float), dim=-1)
            for i in range(args.serialized_seq_len, args.max_len+args.serialized_seq_len, args.serialized_seq_len):
                logit = lm_logits[i - args.serialized_seq_len:i, :]
                label = labels[i - args.serialized_seq_len:i]
                if args.remap_logit:
                    logit_remap = poptorch.custom_op([logit],
                                "RemapCE",
                                "ai.graphcore",
                                1,
                                example_outputs=[logit],
                                attributes={"grain_size": 8})
                if loss is None:
                    if args.remap_logit:
                        acc = calculate_acc(logit_remap[0], label, reduction='sum')
                    else:
                        acc = calculate_acc(logit, label, reduction='sum')
                    loss = loss_fct(logit, label).to(torch.float32) + 0*acc.detach()
                else:
                    if args.remap_logit:
                        tmp_acc = calculate_acc(logit_remap[0], label, reduction='sum')
                    else:
                        tmp_acc = calculate_acc(logit, label, reduction='sum')
                    tmp_loss = loss_fct(logit, label).to(torch.float32) + 0*tmp_acc.detach()
                    loss += tmp_loss
                    acc += tmp_acc
            mean_loss = loss / loss_weights
            total_loss = poptorch.identity_loss(mean_loss, reduction="none")
            acc /= loss_weights
            return total_loss, acc


if __name__ == "__main__":
    args = set_args()
    opts = get_options(args)

    logger("Model initializing")
    model_config = GPT2Config.from_json_file(MODEL_CONFIG[args.model])
    model = GTP2Wrapper(args, model_config).half().train()

    logger("Arguments: {}".format(args))
    logger("Model config: {}".format(model_config))
    optimizer = get_optimizer(args.optimizer, args.weight_decay, args.learning_rate, args.loss_scaling, model,
                              use_popdist=args.use_popdist)
    poptorch_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    if args.compile_only:
        # Compile model
        logger("---------- Compilation/Loading from Cache Started ---------")
        start_compile = time.perf_counter()
        datum = get_generated_datum(args, model_config.vocab_size)
        poptorch_model.compile(*datum)
        duration_compilation = time.perf_counter() - start_compile
        logger(f"Compiled/Loaded model in {duration_compilation} secs")
        logger("-----------------------------------------------------------")
        logger("Model successfully compiled. Exiting now as '--compile-only' argument was passed.")
        sys.exit(0)

    # W&B
    if args.use_wandb and (not args.use_popdist or args.popdist_rank == 0):
        wandb.init(project="torch-gpt2", settings=wandb.Settings(console="wrap"),
                   name="{}_{}_sl{}_gbs{}".format(args.model, model_config.vocab_size, args.max_len, args.batch_size * args.gradient_accumulation * args.replication_factor))
        wandb_config = vars(args)
        wandb.config.update(wandb_config)

    # Dataloader
    logger("------------------- Data Loading Started ------------------")
    start_loading = time.perf_counter()
    train_dataset, validate_dataset = load_dataset(logger, args, model_config.vocab_size)
    loader = DataLoader(opts,
                        train_dataset,
                        shuffle=True if args.train_path.endswith('.pkl') else False,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        worker_init_fn=_WorkerInit(args.seed),
                        collate_fn=collate_fn if not 'dynamic' in args.train_path else None,
                        drop_last=True,
                        auto_distributed_partitioning=not isinstance(train_dataset, torch.utils.data.IterableDataset),
                        mode=DataLoaderMode.AsyncRebatched if args.async_dataloader else DataLoaderMode.Sync)
    samples_per_epoch = int(len(train_dataset) / args.epochs) if 'dynamic' in args.train_path else len(train_dataset)
    steps_per_epoch = int(len(loader) / args.epochs) if 'dynamic' in args.train_path else len(loader)
    logger(f"Samples per epoch: {samples_per_epoch}")
    logger(f"Steps per epoch: {steps_per_epoch}")
    if steps_per_epoch < 1:
        raise RuntimeError("Not enough data in input_files for current configuration, "
                           "try reducing deviceIterations or gradientAccumulation.")
    duration_loader = time.perf_counter() - start_loading
    logger(f"Data loaded in {duration_loader} secs")
    logger("-----------------------------------------------------------")

    if args.lr_decay_steps:
        lr_decay_steps = args.lr_decay_steps
    else:
        lr_decay_steps = steps_per_epoch * args.epochs
    if args.lr_warmup_steps:
        lr_warmup_steps = args.lr_warmup_steps
    else:
        lr_warmup_steps = int(args.lr_warmup * lr_decay_steps)

    scheduler = get_lr_scheduler(optimizer, args.lr_schedule, lr_warmup_steps, lr_decay_steps)
    if args.resume_training_from_checkpoint:
        training_state = torch.load(Path(args.pretrained_checkpoint) / "training_state.pt")
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["lr_scheduler"])

    # Training loop
    logger("--------------------- Training Started --------------------")
    factor = args.gradient_accumulation * args.batches_per_step
    start_train = time.perf_counter()

    epoch = 0
    total_step = 0
    while epoch < args.epochs and total_step < steps_per_epoch * args.epochs:
        for batch_idx, batch in enumerate(loader):
            if not 'dynamic' in args.train_path:
                _input_ids, _labels = batch
                input_ids = _input_ids[:, :-1]
                labels = _labels[:, 1:]
            else:
                input_ids = batch[:, :-1]
                labels = batch[:, 1:]

            start_step = time.perf_counter()
            outputs = poptorch_model(input_ids=input_ids, labels=labels)
            scheduler.step()
            poptorch_model.setOptimizer(optimizer)
            step_length = sync_metrics(time.perf_counter() - start_step)
            outputs_sync = sync_metrics(outputs, factor)
            num_instances = args.popdist_size if args.use_popdist else 1
            step_throughput = num_instances * args.replication_factor * args.batch_size * \
                args.gradient_accumulation * args.batches_per_step / step_length
            if (batch_idx + 1) % args.log_steps == 0:
                logger("stpe {} of epoch {}, loss: {}, acc: {}, lr: {}, Throughput: {} seq/s".format(
                    batch_idx, epoch, outputs_sync[0], outputs_sync[1], scheduler.get_last_lr()[0],
                    step_throughput))

            if args.use_wandb and (not args.use_popdist or args.popdist_rank == 0):
                wandb.log({"Loss": outputs_sync[0],
                           "Acc": outputs_sync[1],
                           "LR": scheduler.get_last_lr()[0],
                           "Step": total_step,
                           "Epoch": epoch + 1,
                           "Throughput": step_throughput})

            if args.save_model_path:
                if not args.use_popdist or args.popdist_rank == 0:
                    if args.save_per_steps is not None and (total_step % args.save_per_steps == 0):
                        model_path = os.path.join(args.save_model_path, 'step_{}'.format(total_step))
                        logger('saving current model to {}'.format(model_path))
                        os.makedirs(model_path, exist_ok=True)
                        model.model.save_pretrained(model_path)
                        torch.save({
                            "step": total_step,
                            "epoch": epoch,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": scheduler.state_dict(),
                            "loss": outputs_sync[0],
                            "acc": outputs_sync[1],
                            "config": args
                        }, os.path.join(model_path, "training_state.pt"))
            total_step += 1
            if total_step % steps_per_epoch == 0:
                epoch += 1
        if args.save_model_path:
            if not args.use_popdist or args.popdist_rank == 0:
                if (epoch % args.save_per_epochs) == 0:
                    model_path = os.path.join(args.save_model_path, 'epoch_{}'.format(epoch + 1))
                    logger('saving current model to {}'.format(model_path))
                    os.makedirs(model_path, exist_ok=True)

                    model.model.save_pretrained(model_path)

                    torch.save({
                        "step": total_step,
                        "epoch": epoch,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": scheduler.state_dict(),
                        "loss": outputs_sync[0],
                        "acc": outputs_sync[1],
                        "config": args
                    }, os.path.join(model_path, "training_state.pt"))
        epoch += 1