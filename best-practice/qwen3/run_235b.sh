# Cluster variables
export CONTAINER_IMAGE=
export ACCOUNT=
export MEGATRON_PATH=
export PARTITION=
export RUN_NAME="${MODEL}-benchmarking"
export CONTAINER_MOUNTS=
export CLUSTER=CW

# Model selection parameters
export MODEL=Qwen3-235B-A22B
export WANDB_PROJECT=
export OUTPUT_PATH=

# # Training parameters
export PROFILE=0 # whether to profile the model with nsys profile
export PRETRAIN=0 # whether train the model from scratch
export MBS=1
export SEQ_LEN=4096
export MOE_GROUPED_GEMM=true

export RUN_TIME=00:30:00
export COMMENT=baseline


# H100 baseline config
A2A_OVERLAP=1 TP=2 PP=8 VPP=4 EP=32 NNODES=32 GBS=2048 bash ./sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules moe_act layernorm --moe-router-force-load-balancing

# B200 baseline config
A2A_OVERLAP=1 TP=1 PP=4 VPP=24 EP=16 NNODES=16 GBS=1024 bash ./sbatch_benchmarking.sh --moe-router-force-load-balancing

# GB200 baseline config 128 GPUs
## EP32 with DeepEP dispatcher
TP=1 PP=4 VPP=12 EP=32 NNODES=32 GBS=1024 bash ./sbatch_benchmarking.sh --moe-router-force-load-balancing --external-cuda-graph --cuda-graph-scope attn --te-rng-tracker --recompute-granularity selective --recompute-modules moe_act layernorm

## EP64 with A2A dispatcher
TP=1 PP=2 VPP=24 EP=64 NNODES=32 MBS=2 GBS=1024 bash ./sbatch_benchmarking.sh --moe-router-force-load-balancing --external-cuda-graph --cuda-graph-scope attn --te-rng-tracker --recompute-granularity selective --recompute-modules moe 