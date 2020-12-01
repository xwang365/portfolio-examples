#!/bin/bash

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

# Before running the install script, make sure that you are in a virtual
# environment with Python3 activated as default.
# virtualenv minigo-venv -p python3.6
# source minigo-venv/bin/activate
# Also, in this environment, the gc_tensorflow for TensorFlow version 1.15
# should be installed follwoing the instructions of the
# Getting Started guide for your IPU system.
# For further installation instructions, refer to the main README.md.


# Exits immediately if subcommand throws error
set -e

# install minigo requirements
pip install -r  minigo/requirements.txt

# install buildtool requirements
## uncomment if applications are missing
# sudo apt install curl
# sudo apt install openjdk-8-jdk
# sudo apt install python3-numpy python3-dev python3-pip python3-wheel python3-virtualenv
# sudo apt install libc-ares-dev

export BAZEL_VERSION=0.24.1
wget -nc https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user

# provide poplar binaries
export TF_POPLAR_BASE=$POPLAR_SDK_ENABLED

# Download and setup Graphcore TensorFlow
# see also: https://github.com/graphcore/tensorflow/blob/r1.15/sdk-release-1.3/GRAPHCORE_BUILD.md
# and https://www.tensorflow.org/install/source
mkdir -p tf_build/install
cd tf_build

if cd tensorflow; then git pull; cd ..; else git clone -b r1.15/sdk-release-1.3 https://github.com/graphcore/tensorflow.git tensorflow; fi

cd tensorflow
export PYTHON_BIN_PATH=$(which python3)
export PYTHON_LIB_PATH=$($PYTHON_BIN_PATH -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
export TF_ENABLE_XLA=1
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_DOWNLOAD_CLANG=0
export CC_OPT_FLAGS='-march=skylake-avx512 -Wno-sign-compare'
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_IPU_HOROVOD=0
export TF_NEED_MPI=0
./configure
cd ..

# setup softlinks to integrate minigo into tensorflow
cd tensorflow/tensorflow/cc/
ln -sf ../../../../minigo/cc cc
cd ../..

# buid selfplay and eval within tensorflow
bazel build --copt="-Ofast" --cxxopt="-Ofast" --linkopt="-Ofast" --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --use_ipu_model --max_compilation_threads=1" --test_size_filters=small,medium,large --test_timeout=240,360,900,3600 --verbose_failures --test_output=errors --define=board_size=9 --config=opt //tensorflow/cc/cc:eval
bazel build --copt="-Ofast" --cxxopt="-Ofast" --linkopt="-Ofast"  --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --use_ipu_model --max_compilation_threads=1" --test_size_filters=small,medium,large --test_timeout=240,360,900,3600 --verbose_failures --test_output=errors --define=board_size=9 --config=opt //tensorflow/cc/cc:selfplay

# link build files back to minigo
cd ../../minigo/
mkdir -p bazel-bin
cd bazel-bin
ln -sf ../../tf_build/tensorflow/bazel-bin/tensorflow/cc/cc cc

# run tests
cd ../../
pytest test_ipu_minigo.py