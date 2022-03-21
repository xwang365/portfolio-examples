# PyTorch GPT2

This directory contains an implementation of GPT2 models in PyTorch for the IPU, leveraging the HuggingFace Transformers library. 

There are two examples for GPT2, the one is for pre-training: `train_gpt2.py` and the second one is text generation `text_generator_fast_one_token.py`

## Environment setup

First, install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` scripts for Poplar and PopART.

SDK version: 2.5

Then, create a virtual environment, install the required packages and build the custom ops.

```
virtualenv venv -p python3.6
source venv/bin/activate
pip3 install -r requirements.txt
make
```

## Quick start with generated mock dataset

Setup your environment as explained above and run the example with the configuration of your choice.

```
python train_gpt2.py \
    --model gpt2 \
    --optimizer AdamW \
    --layers-per-ipu 2 10 \
    --gradient-accumulation 512 \
    --batches-per-step 4 \
    --batch-size 8 \
    --matmul-proportion 0.2 0.2 \
    --ipus-per-replica 2 \
    --enable-half-partials True \
    --embedding-serialization-factor 6 \
    --recompute-checkpoint-every-layer True \
    --train-path generated
```

## Generate pretraining dataset

The dataset used for pretraining is WIKI-103. It can be generated from a RAW dump of Wikipedia following a five step process.

### 1. Download

Use the `wikipedia_download.sh` script to download the latest Wikipedia dump, about 20GB in size.

```
./data/wikipedia_download.sh <chosen-path-for-dump-file>
```

Dumps are available from <https://dumps.wikimedia.org/> (and mirrors) and are licensed under CC BY-SA 3.0 and GNU Free Documentation Licenses.

### 2. Extraction

In order to create the pre-training data we need to extract the Wikipedia dump and put it in this form:

```text
<doc id = article1>
Title of article 1

Body of article 1

</doc>

<doc id = article2>
Title of article 2

Body of article 2
</doc>
```

and so on.

One of the tools that can be used to do so is WikiExtractor, <https://github.com/attardi/wikiextractor>.
Install the WikiExtractor package with `pip3 install wikiextractor`.

In order not to encounter a `UnicodeEncodeError` at this step, you may want to run these two commands first:

```
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
```

You can then use the the `wikipedia_extract.sh` script to use WikiExtractor to extract the data dump.

```
./data/wikipedia_extract.sh <chosen-path-for-dump-file>/wikidump.xml <chosen-folder-for-extracted-files>
```

The result should be a folder containing directories named `AA`, `AB`, ...
Note that the number of directories depends on the parameters of the `wikipedia_extract.sh` script, and is not to be confused with alphabetical ordering of the wikipedia articles.
In other words you should probably not expect all of `AC`, `AD`, ... `ZX`, `ZY`, `ZZ` to be created by the script.

### 3. Pre-processing

Use the `wikipedia_preprocess.py` script to preprocess and tokenize the extracted files and get the `.pkl` data.
```
python3 ./data/wikipedia_preprocess.py --input-file-path <chosen-folder-for-extracted-files> --output-file-path <chosen-folder-for-preprocessed-files>
```

## Run the pre-training application of GPT2-small
**Notice**: The default scripts are used to get benchmarks for throughput only. You must passing path to processed data files to `--train-path` to start the actual pre-training, you may also need to specify the `--save-model-path` to save checkpoints. Further arguments are described in the source file `train_gpt2.py` and hyperparameters for model are listed in json files in `config/`.

This script runs the 117M parameter GPT2 pre-training.
```
bash run/pretraining_small.sh
```

## Run the pre-training application of GPT2-medium
This script runs the 345M parameter GPT2 pre-training.
```
bash run/pretraining_medium.sh
```

## Run the pre-training application of GPT2-large(SL=512)
This script runs the 762M parameter GPT2 pre-training, with sequence length=512.
```
bash run/pretraining_large.sh
```

## Run the pre-training application of GPT2-large by poprun
This script runs the 762M parameter GPT2 distributed pre-training using PopRun, which can scale the application from POD16 to POD256 and further.

We advise you to first read through the [User Guide](https://docs.graphcore.ai/projects/poprun-user-guide/en/latest/index.html) for PopRun before running this script.
```
bash run/pretraining_large_poprun.sh
```

## Run the tests (optional)

Setup your environment and generate the sample dataset as explained above and run `python3 -m pytest` from the root folder.

## TFRecord dataset (optional)
In order to use the multi-threaded `dataloader`, `tfrecord` files need to be generated.
```
cd <chosen-folder-for-preprocessed-files>
mkdir tfrecords
python write_into_tfrecord.py

cd tfrecords
for f in *.tfrecord; do python3 -m tfrecord.tools.tfrecord2idx $f `basename $f .tfrecord`.index; done
```
and use the tfrecord datasets by
```
python train_gpt2.py \
    --model gpt2-medium \
    --optimizer LAMB \
    --learning-rate 0.006 \
    --lr-schedule linear \
    --layers-per-ipu 1 7 8 8 \
    --matmul-proportion 0.2 0.15 0.15 0.15 \
    --ipus-per-replica 4 \
    --replication-factor 1 \
    --gradient-accumulation 1024 \
    --batches-per-step 8 \
    --batch-size 4 \
    --embedding-serialization-factor 6 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'tfrecord' \
    --tfrecord-path ./data/tfrecords/*.tfrecord \
    --epochs 3
```

## Megatron dataset (optional)
We also support mmap dataset which is used by NVIDIA's Megatron, you should first follow the [instruction](https://github.com/ningchaoar/Megatron-LM#data-preprocessing) to get `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`.
Then you need to set `--train-path dynamic` and `--data-prefix <path>/my-gpt2_text_document` in the training scripts.

If you have trained model using Megatron's dataset, you will need to set `--tokenizer-type 1` in `tasks/run_evaluate.sh` for evalutaion.

## Evaluation
### WikiText Perplexity Evaluation
we evaluate perplexity on the word-level [WikiText-103 test dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), 
and appropriately compute perplexity given the change in tokens 
when using our generated BPE tokenizer.

We use the following command to run WikiText-103 evaluation on pretrained model.
```
bash tasks/run_evaluate.sh wiki
```

### LAMBADA Cloze Accuracy
To compute LAMBADA cloze accuracy (the accuracy of predicting the last token given the preceeding tokens) 
we utilize a detokenized, processed version of the [LAMBADA dataset](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl).

We use the following command to run LAMBADA evaluation on a pretrained model.
```
bash tasks/run_evaluate.sh lmbd
```

##  Text Generation (Chinese Poem)
```
bash tasks/run_text_generator.sh
```
or
```
python tasks/text_generator_fast_one_token.py \
      --model-path uer/gpt2-chinese-poem \
      --fp16 true \
      --single-ipu true \
      --poptorch_loop true \
      --batch-size 1 \
      --input-len  100 \
      --output-len 924 \
```
We generate Chinese ancient poems using the pretrained GPT2-small model. 
This task is interactive. You can enter one sentence such as "[CLS]梅 山 如 积 翠 ，" and it will generate the rest ancient poems.  

| Arguments | Meaning |
| ---- | ---- | 
| --model-path | Loading pretrained GPT2-small checkpoint from huggingface.co. The default is uer/gpt2-chinese-poem |
| --fp16 | Whether to use fp16 model for this task|
| --single-ipu |Whether to use single IPU for this task |
| --poptorch_loop| Whether to use poptorch_loop to optimize the latency|
| --batch-size| batch size for inference. The defult is 1|
| --input-len| The maximum input length you want to set for this task|
| --output-len| The maximum output length you want to set for this task|
## Benchmark
We have pretrained GPT2-Medium on both GPU and IPU using the Wikipedia dataset.

|model| device | vocab size | sequence length | optimizer | global batch size | pretraining steps | LM loss | throughput (samples/s) | lambada acc (strict) | lambada acc (non-strict) | wiki (ppl) | wiki (adjusted ppl) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| GPT2-Medium | 8*V100 | 50256 | 1024 | Adam | 1024 | 34k | 3.07 | 112 | 19.35% | 36.04% | 29.16 | 40.97 |
| GPT2-Medium | 8*V100 | 50256 | 1024 | Adam | 1024 | 55k | 2.94 | 112 | 21.21% | 37.55% | 26.03 | 36.16 |
| GPT2-Medium | POD16 | 50272 | 1024 | AdamW | 1024 | 22k | 3.04 | 232 | 19.56% | 36.10% | 28.42 | 39.83 |
| GPT2-Medium | POD16 | 50272 | 1024 | AdamW | 1024 | 36k | 2.89 | 232 | 22.16% | 38.29% | 25.14 | 34.80 |


## Licensing

The code presented here is licensed under the Apache License Version 2.0, see the LICENSE file in this directory.

This directory includes derived work from the following:

GPT2, https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py

Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
Copyright (c) 2021 Graphcore Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

