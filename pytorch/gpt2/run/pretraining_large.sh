#!/bin/bash
# run gpt2-large on POD16
python train_gpt2.py \
    --model gpt2-large \
    --max-len 512 \
    --optimizer AdamW \
    --learning-rate 0.00015 \
    --lr-schedule cosine \
    --lr-warmup 0.01 \
    --layers-per-ipu 0 5 5 5 5 5 5 6 \
    --matmul-proportion 0.40 0.12 0.15 0.15 0.15 0.15 0.15 0.10 \
    --ipus-per-replica 8 \
    --replication-factor 2 \
    --epochs 5 \
    --gradient-accumulation 256 \
    --batches-per-step 8 \
    --batch-size 1 \
    --enable-sequence-serialized False \
    --embedding-serialization-factor 8 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'generated' \
    --replicated-tensor-sharding True \
    --compile-only False