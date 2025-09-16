export CLUSTER=
export CONTAINER_IMAGE=
export MEGATRON_PATH=
export MCORE_RELEASE_VERSION=

export MODEL=DeepSeek-V3
export WANDB_PROJECT=
export RUN_NAME="${MODEL}-benchmarking"

# # Training parameters
export PROFILE=0 # whether to profile the model with nsys profile
export PRETRAIN=0 # whether train the model from scratch
export MBS=1
export SEQ_LEN=4096
export MOE_GROUPED_GEMM=true

export RUN_TIME=00:30:00
export COMMENT=


# best config on 1024 H100 GPUs with 4096 sequence lengths
A2A_OVERLAP=1 PP=8 VPP=4 TP=2 EP=64 NNODES=128 GBS=8192 PR=fp8 bash sbatch_benchmarking.sh \
    --recompute-granularity selective \
    --recompute-modules mla_up_proj mlp \
    --pipeline-model-parallel-layout "Et|(tt|)*30mL"
