#!/bin/bash

# Runs the "340M" parameter model with Distributed Muon
# See more details at: https://github.com/MoonshotAI/Moonlight/blob/master/Moonlight.pdf

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4
export HF_ENDPOINT=https://hf-mirror.com

GPUS_PER_NODE=${NGPU:-8}
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NUM_NODES=${NNODES:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/data/Megatron-LM/local/checkpoint
TENSORBOARD_LOGS_PATH=/data/Megatron-LM/local/checkpoint
# data is preprocessed as described in Megatron-LM' readme
DATA_PATH=/data/Megatron-LM/local/dataset/17_20241108_004951_input_ids_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --disable-bias-linear
    --num-layers 56
    --hidden-size 1920
    --ffn-hidden-size 4800
    --num-attention-heads 30
    --num-query-groups 6
    --seq-length 4096
    --max-position-embeddings 4096
    --init-method-std 0.01
    --rotary-base 100000
    --normalization RMSNorm
    --position-embedding-type rope
    --group-query-attention
    --swiglu
    --transformer-impl transformer_engine
)

MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 4
    --moe-router-load-balancing-type none
    --moe-router-enable-expert-bias
    --moe-router-score-function sigmoid
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-topk 2
    --moe-token-dispatcher-type flex  # deepep
    --moe-enable-deepep
)

TRAINING_ARGS=(
    --optimizer muon
    --micro-batch-size 1
    --global-batch-size 64
    --train-iters 5000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr 1e-3
    --lr-decay-style cosine
    --min-lr 1e-4
    --muon-matched-adamw-rms 0.2
    --lr-warmup-fraction 0.02
    --lr-decay-iters 5000
    --use-distributed-optimizer
    --ckpt-format torch

)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model yulan-team/YuLan-Mini
    --split 990,7,3
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000
    --eval-interval 1000
    --save $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
