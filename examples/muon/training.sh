#!/bin/bash
cd /data/Megatron-LM
source /data/Megatron-LM/.venv/bin/activate
if ! command -v envsubst &> /dev/null; then
  bash /data/install/setup.sh
fi
set -x

# export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=3
NCCL_IB_GID_INDEX=3
NCCL_IB_TC=106
NCCL_CONNECT_TIMEOUT=5
NCCL_RETRY_CNT=100
NCCL_RETRY_WAIT=100
NCCL_IB_TIMEOUT=5
NCCL_IB_RETRY_CNT=100

# Load from kubectl
GPUS_PER_NODE=${NGPU:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-0}
NUM_NODES=${NNODES:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))  # All DP

# Logs and datasets
CHECKPOINT_PATH=/data/Megatron-LM/local/exp/deepseek_moe/${JOB_ID:-standalone}
TENSORBOARD_LOGS_PATH=$CHECKPOINT_PATH
DATA_PATH=/data/Megatron-LM/local/dataset/17_20241108_004951_input_ids_document
mkdir -p $TENSORBOARD_LOGS_PATH

# Parallelism
TP=1
PP=1
MICRO_BATCH_SIZE=8
ACC=1
GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$WORLD_SIZE/$TP/$PP*$ACC))

if [[ $NUM_NODES -gt 1 ]]; then
    DISTRIBUTED_ARGS=(
        --rdzv_backend static
        --nproc_per_node $GPUS_PER_NODE
        --nnodes $NUM_NODES
        --master_addr $MASTER_ADDR
        --master_port $MASTER_PORT
        --node_rank $NODE_RANK
    )
else
    DISTRIBUTED_ARGS=(
        --rdzv_backend static
        --nproc_per_node $GPUS_PER_NODE
        --standalone
    )
fi

GPT_MODEL_ARGS=(
    --disable-bias-linear
    --add-qkv-bias
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
    --attention-backend flash
)

MOE_ARGS=(
    --num-experts 2
    --moe-router-topk 1
    --expert-model-parallel-size 1
    --moe-router-load-balancing-type none
    --moe-router-enable-expert-bias
    --moe-router-score-function sigmoid
    # --moe-grouped-gemm
    # --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
    # --moe-token-dispatcher-type flex  # deepep
    # --moe-enable-deepep
)
MOE_ARGS=()

TRAINING_ARGS=(
    --optimizer muon
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters 5000
    --eval-iters 0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-15
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr 1e-3
    --lr-decay-style cosine
    --min-lr 1e-4
    --muon-matched-adamw-rms 0.2
    --lr-warmup-fraction 0.25
    --lr-decay-iters 5000
    --ckpt-format torch_dist
    # --cross-entropy-loss-fusion
    # --recompute-activations
    --recompute-method uniform
    --recompute-num-layers 56
    --recompute-granularity full
    # --no-persist-layer-norm
    # --check-for-spiky-loss
    # --check-for-large-grads
    --use-distributed-optimizer
    --use-custom-fsdp
    --data-parallel-sharding-strategy optim_grads
    # --use-torch-fsdp2
    --no-gradient-accumulation-fusion
    --untie-embeddings-and-output-weights
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --distributed-timeout-minutes 5
    # --overlap-grad-reduce
    # --overlap-param-gather
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model yulan-team/YuLan-Mini
    --split 1000,0,0
    --num-workers 8
    --reset-position-ids
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000
    --eval-interval 1000
    --save $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --disable-gloo-process-groups
    --use-persistent-ckpt-worker
    --async-save
    --ckpt-fully-parallel-load
    --ckpt-assume-constant-structure
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} |& tee $TENSORBOARD_LOGS_PATH/train-${NODE_RANK}.log
