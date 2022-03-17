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

import warnings

import torch
import poptorch
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel

from train_gpt2 import GTP2Wrapper, set_args

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)


class cpu_wrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = GPT2LMHeadModel(config=config)

    def forward(self, input_ids, labels):
        return self.model.forward(input_ids=input_ids, labels=labels)


def test_ipu_cpu_match():
    """
    Test that the GPT2 model ran on IPU approximately matches that same
    model ran on the CPU.
    """

    # Config
    args = set_args()
    args.batch_size = 1
    args.pretrained_model = None
    args.mlp_serialization_factor = 1
    args.embedding_serialization_factor = 2
    args.layers_per_ipu = [3]
    args.recompute_checkpoint_every_layer = True

    batch_size = args.batch_size
    config = GPT2Config.from_json_file('config/config.json')
    config.model = 'gpt2_test'
    config.attn_pdrop = 0.0
    config.embd_pdrop = 0.0
    config.resid_pdrop = 0.0
    config.summary_first_dropout = 0.0
    config.n_layer = 3
    config.n_embd = 384
    config.n_head = 2

    # Models and options
    opts = poptorch.Options().deviceIterations(1)
    opts.Training.gradientAccumulation(1)
    opts.replicationFactor(1)
    opts.Precision.setPartialsType(torch.float32)
    opts.anchorMode(poptorch.AnchorMode.Final)

    model_cpu = cpu_wrapper(config=config).train()
    model_ipu = GTP2Wrapper(args, config).train()
    model_ipu.load_state_dict(model_cpu.state_dict())

    # Check that copy was successful
    assert model_ipu is not model_cpu
    assert all([(a == b).all() for a, b in zip(
        model_cpu.parameters(), model_ipu.parameters())]) is True

    optimizer_cpu = torch.optim.AdamW(model_cpu.parameters(), lr=0.001)
    optimizer_ipu = poptorch.optim.AdamW(model_ipu.model.parameters(), lr=0.001, loss_scaling=1.0)
    poptorch_model = poptorch.trainingModel(model_ipu, opts, optimizer=optimizer_ipu)

    # Input
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(
        "Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute yo"
        "Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute yo"
        "Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute yo"
        "Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute",
        return_tensors="pt")
    labels = torch.ones_like(inputs['input_ids'])
    past_key_values = [(torch.tensor(
        np.random.randn(batch_size, config.n_head, 128 - 1, int(config.n_embd / config.n_head))).type(torch.float),
                        torch.tensor(np.random.randn(batch_size, config.n_head, 128 - 1,
                                                     int(config.n_embd / config.n_head))).type(torch.float))
                       for _ in range(config.n_layer)]

    batch_cpu = (inputs['input_ids'].repeat(batch_size, 1),
                 labels.repeat(batch_size, 1))

    _label = batch_cpu[1][:, 1:]
    batch_ipu = (batch_cpu[0],
                 torch.cat((_label, -100 * torch.ones((_label.size(0), 1), dtype=torch.long)), dim=1))
    # Training Loop
    for step in range(10):
        # Step CPU model
        optimizer_cpu.zero_grad()
        for b in range(batch_size):
            cpu_output = model_cpu(*batch_cpu)
            cpu_loss = cpu_output[0]
            cpu_loss.backward()
        optimizer_cpu.step()

        # Step IPU Model
        ipu_output = poptorch_model(*batch_ipu)
        ipu_loss = ipu_output[0]

        with torch.no_grad():
            print(f"CPU Loss: {cpu_loss}, IPU Loss: {ipu_loss}")
            # Check the losses are approximately equal
            assert np.allclose(cpu_loss.numpy(), ipu_loss.numpy(), rtol=1e-4)
