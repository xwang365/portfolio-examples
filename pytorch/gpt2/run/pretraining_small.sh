#!/bin/bash
# run gpt2-small on POD16
python train_gpt2.py \
    --model gpt2 \
    --max-len 1024 \
    --optimizer AdamW \
    --learning-rate 0.00004 \
    --lr-schedule cosine \
    --lr-warmup 0.01 \
    --layers-per-ipu 0 4 4 4 \
    --matmul-proportion 0.05 0.10 0.10 0.10 \
    --ipus-per-replica 4 \
    --replication-factor 4 \
    --epochs 5 \
    --gradient-accumulation 128 \
    --batches-per-step 8 \
    --batch-size 1 \
    --enable-sequence-serialized True \
    --embedding-serialization-factor 8 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'generated' \
    --replicated-tensor-sharding True \
    --compile-only False