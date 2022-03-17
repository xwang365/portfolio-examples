#!/bin/bash

echo "Evaluate using $1 dataset"

if [ $1 == "wiki" ]
then
  python tasks/evaluate_wiki.py \
        --valid-data data/wikitext-103/wiki.test.tokens \
        --pretrained-checkpoint $CHECKPOINT_PATH \
        --fp16 \
        --seq-length 1024 \
        --tokenizer-type 1 \
        --batches-per-step 1 \
        --layers-per-ipu 7 8 8 1 \
        --matmul-proportion 0.4 0.4 0.4 0.2
elif [ $1 == "lmbd" ]
then
  python tasks/evaluate_lambada.py \
        --valid-data data/lambada_test.jsonl \
        --pretrained-checkpoint $CHECKPOINT_PATH \
        --fp16 \
        --seq-length 1024 \
        --tokenizer-type 1 \
        --batches-per-step 1 \
        --layers-per-ipu 7 8 8 1 \
        --matmul-proportion 0.4 0.4 0.4 0.2 \
        --strict-lambada false
else
  echo "Dataset should be 'wiki' or 'lmbd'"
fi
