#!/bin/bash
python tasks/text_generator_fast_one_token.py \
      --model-path uer/gpt2-chinese-poem \
      --fp16 true \
      --single-ipu true \
      --poptorch_loop true \
      --batch-size 1 \
      --input-len  100 \
      --output-len 924 \
