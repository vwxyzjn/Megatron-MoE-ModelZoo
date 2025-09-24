#!/bin/bash
set -euxo pipefail

# Environment setup
source /usr/local/gib/scripts/set_nccl_env.sh
export NCCL_SOCKET_IFNAME="eth0,eth1"
export NCCL_TUNER_CONFIG_PATH=/usr/local/gib/configs/tuner_config_a4.txtpb

export TRITON_CACHE_DIR="/tmp/triton-cache/"
#export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=20
export NVTE_BWD_LAYERNORM_SM_MARGIN=20
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
export DEEPEP_COMM_TIMEOUT_MS=30000
export TOKENIZERS_PARALLELISM=false # avoid HF warnings

chmod +x /home/Megatron-LM/pretrain_gpt.py

# Optional defaults (can override externally)
# MASTER_ADDR=${MASTER_ADDR:-"localhost"}
# MASTER_PORT=${MASTER_PORT:-"6000"}
# NNODES=${SLURM_NNODES:-"1"}
# NODE_RANK=${RANK:-"0"}

# ------------------------------
# Argument Groups
# ------------------------------

DISTRIBUTED_ARGS=(
    --nproc_per_node 8
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --rdzv_id="${JOB_IDENTIFIER}"
    --rdzv_backend static
)

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model deepseek-ai/DeepSeek-V3
    --make-vocab-size-divisible-by 3232
)

DATA_ARGS=(
    --split 99,1,0
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --num-workers 6
    --mock-data
    --train-iters 10000
    --lr-decay-iters 10000
    --lr-warmup-iters 1000
)

PERF_ARGS=(
    # Parallelism
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 8
    --expert-model-parallel-size 32
    --context-parallel-size 1
    --expert-tensor-parallel-size 1

    # Pipeline layout
    --pipeline-model-parallel-layout "Et*3|(tt|)*29|mL"

    # Recompute args (activation checkpointing)
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    # Instead of the above, you can use selective recomputation
    # but this doesn't really work well with our EFA setup
    # --recompute-granularity selective
    # --recompute-modules mla_up_proj moe mlp layernorm

    # Offload
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d

    # Overlap comm/compute
    --overlap-grad-reduce
    --overlap-param-gather
    --overlap-moe-expert-parallel-comm
)

TRAINING_ARGS=(
    --seq-length 4096
    --micro-batch-size 1
    --global-batch-size 2048

    # Optimizer
    --lr-warmup-init 3.9e-7
    --lr 3.9e-7
    --min-lr 3.9e-7
    --lr-decay-style cosine
    --adam-beta1 0.9
    --adam-beta2 0.95

    # Precision + optimizer configs
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --main-grads-dtype fp32
    --main-params-dtype fp32
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16

    # Memory + activation
    --sequence-parallel
    --use-flash-attn


    # Misc
    --no-save-optim
    --no-check-for-nan-in-loss-and-grad
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --manual-gc
    --manual-gc-interval 10
    --transformer-impl transformer_engine
    # Regularization args
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --clip-grad 1.0
    --weight-decay 0.1
    --qk-layernorm

    # Initialization args
    --init-method-std 0.02

    # Add mixed precision args
    --bf16

    # Enable experimental args
    --enable-experimental

    # Misc args
    --distributed-timeout-minutes 60
)

CHECKPOINTING_ARGS=(
    --load /gcs-dir/Megatron-MoE-ModelZoo/output/mcore-benchmarking-vyour_own_megatron_version/DeepSeek-V3-TP1PP8EP32VPP4CP1-MBS1GBS8192/checkpoints
    --save /gcs-dir/Megatron-MoE-ModelZoo/output/mcore-benchmarking-vyour_own_megatron_version/DeepSeek-V3-TP1PP8EP32VPP4CP1-MBS1GBS8192/checkpoints
    --save-interval 10000000
    --no-load-optim
    --no-load-rng
    --auto-detect-ckpt-format
    --dist-ckpt-strictness log_all
)

LOGGING_ARGS=(
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --log-interval 1
    --logging-level 40
    --tensorboard-dir /gcs-dir/Megatron-MoE-ModelZoo-workspace/Megatron-MoE-ModelZoo/output/mcore-benchmarking-vyour_own_megatron_version/DeepSeek-V3-TP1PP8EP32VPP4CP1-MBS1GBS8192/tensorboard
)

EVAL_ARGS=(
    --eval-iters 32
    --eval-interval 10000000
)

NETWORK_ARGS=(
    --disable-bias-linear
    --num-layers 61
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --kv-channels 128
    --max-position-embeddings 4096
    --position-embedding-type rope
    --rotary-base 10000
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --multi-latent-attention
)

MOE_ARGS=(
    --num-experts 256
    --moe-layer-freq "([0]*3+[1]*58)"
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 8
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-4
    --moe-router-group-topk 4
    --moe-router-num-groups 8
    --moe-router-topk-scaling-factor 2.5
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-router-fusion
    --moe-router-padding-for-fp8
)

MLA_ARGS=(
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 40
    --mscale 1.0
    --mscale-all-dim 1.0
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
)

FP8_ARGS=(
    --fp8-recipe mxfp8
    --fp8-format e4m3
    --delay-wgrad-compute
)

# ------------------------------
# Launch
# ------------------------------

NVSHMEM_DEBUG=INFO \
OMP_NUM_THREADS=8 \
PYTHON_PATH=/home/Megatron-LM \
TORCH_NCCL_ENABLE_MONITORING=0 \
torchrun \
    ${DISTRIBUTED_ARGS[@]} /home/Megatron-LM/pretrain_gpt.py \
    ${DATA_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${NETWORK_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${FP8_ARGS[@]} \
    "$@"