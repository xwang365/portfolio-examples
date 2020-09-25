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

"""Code to create ipu configuration graphs in Python

The purpose of the dumped graphs is to load them into the C++ code base
to initialize hardware initialization.
"""

import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
import os
import errno

# IpuConfigureHardware happens on CPU
device = "cpu"

try:
    os.makedirs("minigo/results")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for i in range(16):
    cfg = ipu.utils.create_ipu_config(profiling=False,
                                      profile_execution=False,
                                      use_poplar_text_report=False)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=True)
    cfg = ipu.utils.auto_select_ipus(cfg, [1]*(i+1))
    g = ops.Graph()
    with g.as_default():
        with ops.device(device):
            cfg_op = gen_ipu_ops.ipu_configure_hardware(cfg.SerializeToString())
            with tf.gfile.GFile(
                    "minigo/results/" + str(i+1) + "_ipu_init_graph_def.pb", "wb") as f:
                f.write(g.as_graph_def().SerializeToString())
