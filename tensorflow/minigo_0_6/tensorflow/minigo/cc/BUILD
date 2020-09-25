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

# Description:
# TensorFlow cc tools.

load("//tensorflow:tensorflow.bzl",
    "cc_library_with_android_deps",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_copts",
    "tf_gen_op_wrappers_cc",
    "transitive_hdrs",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

cc_library(
    name = "dual_net_selfplay",
    srcs = [
        "dual_net/tf_dual_net.cc",
        "dual_net/batching_dual_net.cc",
        "dual_net/factory.cc",
        "dual_net/reloading_dual_net.cc",
        "dual_net/dual_net.cc",
        "dual_net/random_dual_net.cc",
        "dual_net/fake_dual_net.cc",
        "mcts_player.cc",
        "mcts_node.cc",
        "game.cc",
        "position.cc",
        "coord.cc",
        "color.cc",
        "random.cc",
        "zobrist.cc",
        "file/path.cc",
        "file/utils_tf.cc",
        "init.cc",
        "sgf.cc",
        "tf_utils.cc",
        "logging.cc",
        "game_utils.cc",
        "platform/utils_linux.cc",
    ],
    hdrs = [
        "init.h",
        "platform/utils.h",
        "color.h",
        "stone.h",
        "mcts_player.h",
        "tiny_set.h",
        "constants.h",
        "logging.h",
        "file/path.h",
        "dual_net/factory.h",
        "dual_net/dual_net.h",
        "dual_net/tf_dual_net.h",
        "dual_net/fake_dual_net.h",
        "dual_net/random_dual_net.h",
        "dual_net/reloading_dual_net.h",
        "dual_net/batching_dual_net.h",
        "dual_net/inference_cache.h",
        "position.h",
        "group.h",
        "inline_vector.h",
        "zobrist.h",
        "move.h",
        "random.h",
        "symmetries.h",
        "algorithm.h",
        "game.h",
        "game_utils.h",
        "mcts_node.h",
        "thread_safe_queue.h",
        "coord.h",
        "file/utils.h",
        "sgf.h",
        "tf_utils.h",
    ],
    deps = [
        "@com_github_gflags_gflags//:gflags",
	"//tensorflow/compiler/plugin/poplar:poplar_lib",
        "//tensorflow/core:tensorflow",
    ],
    defines = [
      "MINIGO_SELFPLAY=1",
      "MINIGO_EVAL=0",
      "MY_DEVICE_COUNT=1",
      "MG_ENABLE_TF_DUAL_NET=1"
    ],
)

tf_cc_test(
    name = "selfplay",
    srcs = [
	    "selfplay.cc",
    ],
    deps = [
	    "dual_net_selfplay",
    ],
)

cc_library(
    name = "dual_net_eval",
    srcs = [
        "dual_net/tf_dual_net.cc",
        "dual_net/batching_dual_net.cc",
        "dual_net/factory.cc",
        "dual_net/reloading_dual_net.cc",
        "dual_net/dual_net.cc",
        "dual_net/random_dual_net.cc",
        "dual_net/fake_dual_net.cc",
        "mcts_player.cc",
        "mcts_node.cc",
        "game.cc",
        "position.cc",
        "coord.cc",
        "color.cc",
        "random.cc",
        "zobrist.cc",
        "file/path.cc",
        "file/utils_tf.cc",
        "init.cc",
        "sgf.cc",
        "tf_utils.cc",
        "logging.cc",
        "game_utils.cc",
        "platform/utils_linux.cc",
        "tf_bt_utils_dummy.cc",
    ],
    hdrs = [
        "init.h",
        "platform/utils.h",
        "color.h",
        "stone.h",
        "mcts_player.h",
        "tiny_set.h",
        "constants.h",
        "logging.h",
        "file/path.h",
        "dual_net/factory.h",
        "dual_net/dual_net.h",
        "dual_net/tf_dual_net.h",
        "dual_net/fake_dual_net.h",
        "dual_net/random_dual_net.h",
        "dual_net/reloading_dual_net.h",
        "dual_net/batching_dual_net.h",
        "dual_net/inference_cache.h",
        "position.h",
        "group.h",
        "inline_vector.h",
        "zobrist.h",
        "move.h",
        "random.h",
        "symmetries.h",
        "algorithm.h",
        "game.h",
        "game_utils.h",
        "mcts_node.h",
        "thread_safe_queue.h",
        "coord.h",
        "file/utils.h",
        "sgf.h",
        "tf_utils.h",
    ],
    deps = [
	"@com_github_gflags_gflags//:gflags",
	"//tensorflow/compiler/plugin/poplar:poplar_lib",
	"//tensorflow/core:tensorflow",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
    defines = [
      "MINIGO_SELFPLAY=0",
      "MINIGO_EVAL=1",
      "MY_DEVICE_COUNT=4",
      "MG_ENABLE_TF_DUAL_NET=1",
    ],
)

tf_cc_test(
    name = "eval",
    srcs = [
	"eval.cc",
    ],
    deps = [
	"dual_net_eval",
    ],
)

