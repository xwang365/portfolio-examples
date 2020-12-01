```
// Copyright (c) 2020 Graphcore Ltd.
// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// This file has been modified by Graphcore Ltd
// It has been modified to run the application on IPU hardware.
```



# Minigo 

This README describes how to run the IPU version of the MiniGo training application.
This has been developed using Poplar SDK 1.3.

## Overview

MiniGo is a representative benchmark for reinforcement learning. 
Many applications use reinforcement learning, such as robotics, 
autonomous driving, traffic control systems, finance and games. 

This task benchmarks on policy reinforcement learning 
for the 9x9 version of the board game Go. 
The model plays games against itself and uses these games to improve play.

## Graphcore MiniGo model

In the example given below we describe how to run the IPU version 
of the Minigo training benchmark derived from: 

(https://github.com/mlperf/training/tree/e8237dc295deab8f9064dc2e0651c8234710c0d5).

This example re-implements AlphaGo on a 9x9 array with a network 
that has only 9 residual layers instead of the original 42.

## Dataset

Data is obtained using the `get_data.py` script.
It provides data for the start network as well as the parameters
of the final network that needs to be beaten.
Performance is measured by letting the newly trained network
play 100 games against the reference network.
The benchmark time is measured until the time point
where the trained network beats the reference network
in more than 49 games. 

## File Structure

The original benchmark code has been copied into the file structure 
given in the table below.
Some adaptions were made to run the benchmark on the IPU and optimise it. 
These changes are described in the `File Changes` section.

| File                         | Description                                          |
| ---------------------------- | ---------------------------------------------------- |
| `README.md`                  | How to run the model                                 |
| `install.sh`                 | Setup bash script (**see Quick start guide**)        |
| `multi_run_and_time.sh`      | Run configuration scripts and full benchmark (bash)  |
| `ipu_reservation_graph`      | Creates graph configuration files to reserve IPUs    |
| `run_and_time.sh`            | Starts a single benchmark round out of ten (bash)    |
| `reference_implementation.py`| Orchestrates the whole benchmark                     |
| `dual_net.py`                | Defines the network architecture                     |
| `train.py`                   | Starts the training of the network                   |
| `requirements.txt`           | Required Python 3.6 packages                         |
| `preprocessing.py`           | Prepares the data for training and validation        |
| `tf_dual_net.cc`             | TensorFlow C++ interface, adapted to map to the IPU  |
| `selfplay.cc`                | Runs a number of games in parallel                   |
| `eval.cc`                    | Two networks play against each other                 |
| `eval_models.py`             | Calls eval.cc                                        | 
| `get_data.py`                | Pulls reference data including reference network     |
| `ml_perf/flags/9`             | Directory containing config files                    |
| `mcts_player.cc`             | Main game playing engine                             |
| `cc`                         | Main C++ files directory                             |
| `cc/BUILD`                   | Main Bazel build file                                |
| `test_ipu_minigo.py`          | Run reduced version of benchmark for testing         |


## Quick start guide

### 1) Download the Poplar SDK

Install Poplar SDK 1.3 following the instructions in the 
Getting Started guide for your IPU system. 
The Getting Started guides can be found on the support portal here: 
https://www.graphcore.ai/support.
Make sure to source the `enable.sh` script for Poplar.

### 2) Package installation

For assigning processes and memory to specific NUMA nodes,
we need the `numactl` package to be installed.
If not already installed, use the following command to install it:

```
sudo apt-get install numactl
```

Also, make sure that the virtualenv package is installed for Python 3.6.

### 3) Prepare the TensorFlow environment

Activate a Python virtual environment with `gc_tensorflow` 
installed for TensorFlow version 1.15 as follows:

```
virtualenv minigo-venv -p python3.6
source minigo-venv/bin/activate
pip install <path to gc_tensorflow_1.15.whl>
```

The use of Python 3.6 is crucial, because our code
assumes that the `python` command leads to
*Python 3.6* in the virtual environment
and `pip` leads to `pip3` in the installation script. 

### 4) Environment variable setting

#### Caching

When the code has not already been compiled for the IPU, the performance figures are adversely affected (since they will include the compile time). Avoid recompilation by caching the compiled code in a disk, using the following environment variable:

```
export TF_POPLAR_FLAGS="--executable_cache_path=<path to large storage>/ipu_cache/"
```

The `<path to large storage>` can be `/localdata/<your username>` or `/mnt/data/<your username>`for example.
Whichever path you choose, make sure you have read and write permissions for it
and around 5 GB of free memory.

#### Logging (optional)

To get Poplar output use the following command:

```
export POPLAR_LOG_LEVEL=INFO
```

By doing this you will get more information about how the model is running - for example you will be able to see if compilations are running correctly and how the IPU is being used.
For more details refer to 
[the poplar user guide](https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/env-vars.html).
You can remove the logging with the following command:

```
unset POPLAR_LOG_LEVEL=INFO
```

#### File system (optional)

Make sure that the temporary directory has at least 1 TB of free memory  
and that you have a fast hard drive such as an SSD for data management.
If your default temporary directory is too small, redirect it using:

```
mkdir -p /mnt/data/USERNAME/temp/
export TMPDIR=/mnt/data/USERNAME/temp/
```

For the result storage, 
which is within the repository in the `tensorflow/minigo/results` folder,
you should have at least 40 GB available.

### 5) Run the installation and test script 

To run the installation
(after the setup of the virtual environment) do the following:

```
$: cd tensorflow
$: bash install.sh
```

This installs the required packages,
compiles the C++ code,
pulls the respective reference data,
and eventually executes the test script.

To run the tests separately, run

```
pytest test_ipu_minigo.py
```

in the `tensorflow` directory within the `minigo` directory.
The tests are also executed in the installation process.

The test requires eight IPUs on an IPU-Server, 
a set of four for training, and four single IPUs for inference (selfplay)
which is running in parallel. 

The tests cover the setup of the environment,
running one single loop consisting of training, selfplay and evaluation,
followed by running another single loop together with the evaluation that compares the
resulting trained network against a reference network.
Some tests will take a bit longer to run.
Only execution is tested 
and the expectation is that no errors are raised.

### 6) Execution

The whole benchmark (10 runs) is executed using:

```
bash multi_run_and_time.sh
```

First, the script downloads the data and creates some C++ IPU configuration files. 
Then it executes the benchmark 10 times.
This will take around six hours and requires an **IPU-Server with 16 IPUs**.
Again, four IPUs will be used for training but this time the remaining 12
IPUs will be occupied by independent inference (selfplay) processes.
Using `sh` instead of `bash` will result in an error after the first
benchmark round.

### 7) Results

A results summary can be obtained with a simple bash command:

```
for i in minigo/results/*.log ; do echo -n $i" : "  ; tail -n4 "$i" | head -n 1 ; done > benchmark_output.txt
more benchmark_output.txt
```

You might have to adapt the path to the log files and filter out the ones
that correspond to your latest run.
One line in the `benchmark_output.txt` might look like:

```
IPU1234-2019-12-24-13-31_6.log : Model 000020-000015 beat target after 2345.678s
```

 - The most important information here is the total execution time 
   of the benchmark of 2345.678s.
   Note that this time will always be lower
   than the actual execution time of the benchmark.
   First, the benchmark script executes a predefined number of 
   iterations of data generation and training.
   Afterwards, it checks which iteration was actually better than
   the reference implementation and reports the time.
 - `IPU1234-2019-12-24-13-31_6.log` is the log file name.
 - `IPU1234` is the name of the machine the benchmark was executed on.
 - `2019-12-24` is the execution data (format: YYYY-MM-DD).
 - `13-31` is the execution time in hours and minutes.
 - The final `6` in the log file name 
   stands for the iteration index of the benchmark execution
   and can be any number between 0 and 9.
 - `Model 000020-000015 beat target` means that after 20 iterations
   within the benchmark loop of training, evaluation,
   and data generation (selfplay), the model defeated the reference network.
   In 15 cases, the evaluation step determined that the
   trained model is actually better than the previous model and should be
   used in the next iteration.
   In the remaining cases, the previous model was reused.
   If there is sufficient training these numbers should be very similar.
   In our implementation, training 
   and data generation are running in parallel and 
   a difference of 25% is expected. 

According to the benchmark rules, 
the fastest and slowest run have to be removed from the statistics.
A large variation of the results is expected.

If your log file does not provide a number in this format,
the execution was not successful.
It might be necessary to increase the number of iterations 
in the configuration files and run the benchmark again, or fix any
errors reported in the log file.


## Model
### Publication/attribution

This application is based on a fork of the minigo project (https://github.com/tensorflow/minigo); which is inspired by the work done by Deepmind with ["Mastering the Game of Go with Deep Neural Networks and
Tree Search"](https://www.nature.com/articles/nature16961), ["Mastering the Game of Go without Human
Knowledge"](https://www.nature.com/articles/nature24270), and ["Mastering Chess and Shogi by
Self-Play with a General Reinforcement Learning
Algorithm"](https://arxiv.org/abs/1712.01815). Note that minigo is an
independent effort from AlphaGo, and that this fork is minigo is independent from minigo itself.

### Reinforcement setup

This benchmark includes both the environment and training for 9x9 go. There are four primary phases in this benchmark, these phases are repeated in order:

 - Selfplay: the *current best* model plays games against itself to produce board positions for training.
 - Training: train the neural networks selfplay data from several recent models.
 - Target Evaluation: the termination criteria for the benchmark is checked using the provided record of professional games.
 - Model Evaluation: the *current best* and the most recently trained model play a series of games. In order to become the new *current best*, the most recently trained model must win 55% of the games.

### Structure

This task has a non-trivial network structure, including a search tree. A good overview of the structure can be found here: https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0.

### Weight and bias initialization and loss function
Network weights are initialized randomly. Initialization and loss are described here;
["Mastering the Game of Go with Deep Neural Networks and Tree Search"](https://www.nature.com/articles/nature16961)

### Optimizer
We use a MomentumOptimizer to train the primary network.

## File modifications

Most files are used unchanged, especially the C++ source code.
Some modifications were made to run the model efficiently on the IPU.
These modifications mean that the code will no longer run on CPU, GPU or TPU.

The list below details the changes that were made: 

* multi_run_and_time.sh:
                    ​Run configuration scripts and run series of ten benchmarks (new file).
                    For the competition, the best and worst run are removed, and the arithmetic mean is taken.

* ipu_reservation_graph:
​                    Creates graph configuration files to reserve IPUs in TensorFlow C++ (new file).

* run_and_time.sh:  
                    Starts a single benchmark.
                    ​It mixes the original file, submission files from the competition, and our own adaptions like an IPU reset.

* reference_implementation.py:
​                    Orchestrates the whole benchmark.
​                    Several changes optimise the distribution of the
​                    processes onto CPUs and IPUs.
​                    Training and evaluation happen in parallel
​                    to data generation (selfplay).
​                    The number of 1000 iterations (steps) is normalised
​                    to the original batch size of 4096 for comparability.

* dual_net.py:      
                    Defines the network architecture.
​                    Code freezing was changed to checkpointing instead.
​                    IPU interfaces were defined.
​                    Unnecessary hooks and metrics were removed.
* train.py:         
                    Starts the training of the network. Major changes are
​                    interfacing the IPU estimator as well as
​                    data preparation for IPUs.

* requirements.txt: 
                    Required Python packages.

* preprocessing.py: 
                    Prepares the data for training (and validation).
​                    For the IPU, we return a dataset instead of an iterator.
​                    Additionally, we make sure that data filtering, shuffling,
​                    and random rotation are deactivated to optimise data throughput.

* tf_dual_net.cc:   
                    TensorFlow C++ interface, adapted to map to the IPU.
​                    The IPU hardware is reserved.
​                    A checkpoint file is loaded instead of the original
​                    protobuf file. The graph is mapped to the IPU,
​                    attached to a session, and alternately executed
​                    by two workers. The original approach created two
​                    separate sessions.
​                    Additionally, we added switches for selfplay.cc and eval.cc.
​                    For eval, we get two different networks and respective sessions.
​                    Since those result in different programs for the IPU,
​                    we keep them on separate IPUs to avoid loading of the executables.
​                    We attach sessions to workers instead of graphs that are
​                    used by the workers to initialize sessions.

* selfplay.cc:     
                    Run a number of games in parallel and orchestrate
​                    the mapping to the accelerators.
​                    Minor adaptions have been made to make the code compile.

* eval.cc:         
                    Two networks play against each other.
​                    Minor adaptions for code compilation.

* eval_models.py:   
                    Calls eval.cc on the different results from the loop iteration
​                    to determine when the reference network was beaten.
​                    It is crucial for determining the time of convergence but
​                    its execution is not included in the convergence time.

* get_data.py:      
                    Script to pull reference data, especially the reference
​                    network. Directory names and file names had to be corrected to
​                    fit to the public data structure, which was changed to
​                    also support 19x19 Go fields.

* mlperf/flags/9:   
                    Directory containing configuration files.
​                    Our minor changes are described in a later section.
​                    Parts that were core to the benchmark remained unchanged.

* mcts_player.cc:   
                    Main game playing engine (unchanged).

* cc:              
                    The original game engine was too slow and largely limiting
​                    overall performance. Hence it was ported to C++ with the files
​                    in this directory.
​                    Apart from the aforementioned changes, we adapted imports
​                    to enable compilation.

* cc/BUILD:        
                    Main Bazel build file.
​                    File was adapted to include IPU libraries and
​                    parameters, specific to our implementation.

* test_ipu_minigo.py:
​                    Run a reduced version of the benchmark for testing.


We deleted several files that were no longer needed.


## Configuration modifications

For transparency, we describe below the minor changes that were made to the configuration of
the benchmark to fine tune the benchmark for the IPU.

### Data pipeline

We removed random rotations because they did not improve performance,
reduced data throughput, and had issues with autograph.

Additionally, we made sure samples are not randomly reduced by 50% or
randomised.
This part of the implementation created a very slow data pipeline and is not
necessary. When the data is stored, it is already randomised.
So for the data reduction, a smaller number of training steps could be used
instead.

### Training length

The whole dataset roughly needs 1000 steps for one episode.
Whereas the original implementation iterated only over
50% of the data which is equivalent to 500 steps,
we use around 1500. This number is optimised to ensure that training and
evaluation takes as much time as selfplay.

Due to data caching, the IPU training can benefit at least partially from
better data throughput.

Due to the optimal use of training, convergence is almost guaranteed,
and the validation step in the original implementation can be skipped.

Similar adaptions have been made in the original competition.

### Batch sizes

For training, we used a batch size of 32 to ensure that the whole training
happens within the IPU without external communication.

For inference, we made sure that the batch size is not adapted to the
data input but is fixed. Otherwise different IPU code would have been required
and loading this code would have slowed down inference.
The maximum batch size can be precalculated by multiplying the
number of parallel games by the number of virtual losses (8),
divided by the number of IPUs that the data is distributed on,
and divided by the number of parallel workers (2).

### Loop iterations

We reduced the total number of loop iterations,
since convergence happened much earlier.
This does not affect the time measurement.
