#!/bin/bash

##SBATCH --nodes=32
##SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=8
#SBATCH --ntasks=32
#SBATCH --account=root
#SBATCH --partition=h200-high
#SBATCH --job-name=root-moe--benchmarking-250909_000201
#SBATCH --dependency=singleton
##SBATCH --output=/mnt/home/costa/periodic-mono/thirdparty/Megatron-MoE-ModelZoo-workspace/Megatron-MoE-ModelZoo/slurm.log
#SBATCH --exclusive
#SBATCH --requeue

# Use the first host in the allocation as the rendez-vous master address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)

# Use a port correlated to the job ID, in the range of 20000-29999.
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
export TRITON_CACHE_DIR="/tmp/triton-cache/"

# deepseek v3
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

srun \
    --container-image=876143322689.dkr.ecr.us-east-1.amazonaws.com/periodic-mono:megatron-moe-modelzoo-no-efa-te-transformer_engine_2.8.0.dev0-603dbf7 \
    --container-mounts=/mnt/:/mnt/ \
    --container-workdir=/mnt/home/costa/periodic-mono/thirdparty/Megatron-MoE-ModelZoo-workspace/Megatron-LM \
    bash -c 'torchrun \
        --nproc_per_node 8 \
        --nnodes $SLURM_NNODES \
        --node_rank $SLURM_NODEID \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT /mnt/home/costa/periodic-mono/thirdparty/Megatron-MoE-ModelZoo-workspace/Megatron-LM/pretrain_gpt.py  \
        --distributed-timeout-minutes 60 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 8 \
        --expert-model-parallel-size 32 \
        --context-parallel-size 1 \
        --expert-tensor-parallel-size 1 \
        --use-distributed-optimizer  \
        --use-mcore-models  \
        --sequence-parallel  \
        --use-flash-attn  \
        --disable-bias-linear  \
        --micro-batch-size 1 \
        --global-batch-size 4096 \
        --train-samples 65528000 \
        --no-save-optim  \
        --no-check-for-nan-in-loss-and-grad  \
        --cross-entropy-loss-fusion  \
        --cross-entropy-fusion-impl te \
        --manual-gc  \
        --manual-gc-interval 10 \
        --transformer-impl transformer_engine \
        --seq-length 4096 \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model deepseek-ai/DeepSeek-V3 \
        --mock-data  \
        --split 99,1,0 \
        --no-mmap-bin-files  \
        --no-create-attention-mask-in-dataloader  \
        --num-workers 6 \
        --num-layers 61 \
        --hidden-size 7168 \
        --ffn-hidden-size 18432 \
        --num-attention-heads 128 \
        --kv-channels 128 \
        --max-position-embeddings 4096 \
        --position-embedding-type rope \
        --rotary-base 10000 \
        --make-vocab-size-divisible-by 3232 \
        --normalization RMSNorm \
        --norm-epsilon 1e-6 \
        --swiglu  \
        --untie-embeddings-and-output-weights  \
        --multi-latent-attention  \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --qk-layernorm  \
        --lr-decay-samples 584765624 \
        --lr-warmup-samples 1536000 \
        --lr-warmup-init 3.9e-7 \
        --lr 3.9e-7 \
        --min-lr 3.9e-7 \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --num-experts 256 \
        --moe-layer-freq "([0]*3+[1]*58)" \
        --moe-ffn-hidden-size 2048 \
        --moe-shared-expert-intermediate-size 2048 \
        --moe-router-load-balancing-type seq_aux_loss \
        --moe-router-topk 8 \
        --moe-token-dispatcher-type flex \
        --moe-enable-deepep  \
        --moe-router-pre-softmax  \
        --moe-grouped-gemm  \
        --moe-aux-loss-coeff 1e-4 \
        --moe-router-group-topk 4 \
        --moe-router-num-groups 8 \
        --moe-router-topk-scaling-factor 2.5 \
        --moe-router-score-function sigmoid \
        --moe-router-enable-expert-bias  \
        --moe-router-bias-update-rate 1e-3 \
        --moe-router-dtype fp32 \
        --moe-permute-fusion  \
        --q-lora-rank 1536 \
        --kv-lora-rank 512 \
        --qk-head-dim 128 \
        --qk-pos-emb-head-dim 64 \
        --v-head-dim 128 \
        --rotary-scaling-factor 40 \
        --mscale 1.0 \
        --mscale-all-dim 1.0 \
        --mtp-num-layers 1 \
        --mtp-loss-scaling-factor 0.1 \
        --eval-iters 32 \
        --eval-interval 10000000 \
        --no-load-optim  \
        --no-load-rng  \
        --auto-detect-ckpt-format  \
        --load /path/to/DeepSeek-V3-dist/torch_dist/ \
        --save /mnt/home/costa/periodic-mono/thirdparty/Megatron-MoE-ModelZoo-workspace/Megatron-MoE-ModelZoo/output/mcore-benchmarking-vyour_own_megatron_version/DeepSeek-V3-TP1PP8EP32VPP4CP1-MBS1GBS8192/checkpoints \
        --save-interval 10000000 \
        --dist-ckpt-strictness log_all \
        --init-method-std 0.02 \
        --log-memory-to-tensorboard  \
        --log-validation-ppl-to-tensorboard  \
        --log-throughput  \
        --log-interval 1 \
        --logging-level 40 \
        --tensorboard-dir /mnt/home/costa/periodic-mono/thirdparty/Megatron-MoE-ModelZoo-workspace/Megatron-MoE-ModelZoo/output/mcore-benchmarking-vyour_own_megatron_version/DeepSeek-V3-TP1PP8EP32VPP4CP1-MBS1GBS8192/tensorboard \
        --bf16  \
        --enable-experimental   \
        --recompute-granularity selective \
        --recompute-modules mla_up_proj moe mlp layernorm \
        --pipeline-model-parallel-layout "Et*2|(tt|)*22t|(tt|)*7mL" \
        --fp8-recipe blockwise \
        --fp8-format e4m3 \
        --use-precision-aware-optimizer \
        --main-grads-dtype fp32 \
        --main-params-dtype fp32 \
        --exp-avg-dtype bf16 \
        --exp-avg-sq-dtype bf16 \
        --moe-router-padding-for-fp8 \
        --overlap-grad-reduce \
        --overlap-param-gather'